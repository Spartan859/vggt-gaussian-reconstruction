#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import threading
import time
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from vggt_gaussian_reconstruction.colmap import camera_matrix, has_model, read_model
from vggt_gaussian_reconstruction.gaussian_trainer import _normalize_model
from vggt_gaussian_reconstruction.geometry import qvec_to_rotmat, rotmat_to_qvec


RENDER_MODE_MAP = {
    "rgb": "RGB",
    "depth(accumulated)": "D",
    "depth(expected)": "ED",
    "alpha": "RGB",
}

CAMERA_PATH_TYPES = ("train", "interp", "ellipse", "spiral")
DEFAULT_VIEWER_SCENES = ("scene", "data1_baseline", "data2_baseline")


class ViewerCheckpointError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a browser-based real-time gsplat viewer.")
    parser.add_argument("--scene", type=Path, default=None, help="Scene directory containing Gaussian outputs.")
    parser.add_argument("--mode", choices=["vggt", "ba"], default="ba")
    parser.add_argument("--checkpoint", "--ckpt", type=Path, default=None, help="Path to checkpoint.pt.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing gaussians_<mode>/checkpoint.pt.")
    parser.add_argument(
        "--extra-scene",
        type=Path,
        action="append",
        default=[],
        help="Additional scene directory exposed in the viewer scene dropdown.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind-host", default="0.0.0.0", help="Interface address for the viewer server.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for viewer captures.")
    parser.add_argument("--host", default=None, help="Optional display host printed in the viewer URL.")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.scene, args.mode, args.checkpoint, args.run_dir)
    scene_options = discover_viewer_scenes(args.scene, args.mode, args.extra_scene)
    checkpoint_options = discover_checkpoints(args.scene, args.mode, args.checkpoint, args.run_dir)
    output_dir = args.output_dir or checkpoint.parent / "viewer"
    launch_viewer(
        checkpoint=checkpoint,
        checkpoint_options=checkpoint_options,
        scene_options=scene_options,
        scene=args.scene,
        mode=args.mode,
        checkpoint_arg=args.checkpoint,
        run_dir=args.run_dir,
        output_dir=output_dir,
        device_name=args.device,
        port=args.port,
        bind_host=args.bind_host,
        display_host=args.host,
    )


def resolve_checkpoint(
    scene: Path | None,
    mode: str,
    checkpoint: Path | None = None,
    run_dir: Path | None = None,
) -> Path:
    if checkpoint is not None:
        return _require_checkpoint(checkpoint)
    if run_dir is not None:
        return _require_checkpoint(run_dir / f"gaussians_{mode}" / "checkpoint.pt")
    if scene is None:
        raise ViewerCheckpointError("Pass --checkpoint, --run-dir, or --scene.")

    candidates = [scene / f"gaussians_{mode}" / "checkpoint.pt"]
    runs_dir = scene / "runs"
    if runs_dir.exists():
        candidates.extend(sorted(runs_dir.glob(f"*/gaussians_{mode}/checkpoint.pt"), key=lambda p: p.stat().st_mtime))
    for candidate in reversed(candidates):
        if candidate.exists():
            return candidate
    searched = "\n".join(str(p) for p in candidates)
    raise ViewerCheckpointError(f"No Gaussian checkpoint found. Searched:\n{searched}")


def discover_checkpoints(
    scene: Path | None,
    mode: str,
    checkpoint: Path | None = None,
    run_dir: Path | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    if scene is not None:
        candidates.append(scene / f"gaussians_{mode}" / "checkpoint.pt")
        runs_dir = scene / "runs"
        if runs_dir.exists():
            candidates.extend(runs_dir.glob(f"*/gaussians_{mode}/checkpoint.pt"))
    if run_dir is not None:
        candidates.append(run_dir / f"gaussians_{mode}" / "checkpoint.pt")
    if checkpoint is not None:
        candidates.append(checkpoint)

    seen = set()
    existing = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        key = candidate.resolve()
        if key in seen:
            continue
        seen.add(key)
        existing.append(candidate)
    return sorted(existing, key=lambda p: p.stat().st_mtime, reverse=True)


def discover_viewer_scenes(
    scene: Path | None,
    mode: str,
    extra_scenes: list[Path] | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    if scene is not None:
        candidates.append(scene)
        parent = scene.parent
        for name in DEFAULT_VIEWER_SCENES:
            candidates.append(parent / name)
    candidates.extend(extra_scenes or [])

    seen = set()
    existing = []
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        key = candidate.resolve()
        if key in seen:
            continue
        if not discover_checkpoints(candidate, mode):
            continue
        seen.add(key)
        existing.append(candidate)
    return existing


def _scene_label(path: Path) -> str:
    parent_name = path.parent.name
    if parent_name:
        return f"{parent_name}/{path.name}"
    return path.name


def _scene_choices(paths: list[Path]) -> tuple[list[str], dict[str, Path]]:
    labels: list[str] = []
    by_label: dict[str, Path] = {}
    for path in paths:
        label = _scene_label(path)
        if label in by_label and by_label[label].resolve() != path.resolve():
            label = str(path)
        labels.append(label)
        by_label[label] = path
    return labels, by_label


def _scene_for_checkpoint(checkpoint: Path, scene_options: list[Path]) -> Path | None:
    checkpoint_resolved = checkpoint.resolve()
    best: Path | None = None
    best_len = -1
    for scene in scene_options:
        try:
            scene_resolved = scene.resolve()
            checkpoint_resolved.relative_to(scene_resolved)
        except ValueError:
            continue
        path_len = len(scene_resolved.parts)
        if path_len > best_len:
            best = scene
            best_len = path_len
    return best


def _checkpoint_label(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 3 and parts[-2].startswith("gaussians_"):
        return f"{parts[-3]}/{parts[-2]}"
    if len(parts) >= 2:
        return str(Path(parts[-2]) / parts[-1])
    return str(path)


def _checkpoint_choices(paths: list[Path]) -> tuple[list[str], dict[str, Path]]:
    labels: list[str] = []
    by_label: dict[str, Path] = {}
    for path in paths:
        label = _checkpoint_label(path)
        if label in by_label and by_label[label].resolve() != path.resolve():
            label = str(path)
        labels.append(label)
        by_label[label] = path
    return labels, by_label


def launch_viewer(
    checkpoint: Path,
    checkpoint_options: list[Path] | None,
    scene_options: list[Path] | None,
    scene: Path | None,
    mode: str,
    checkpoint_arg: Path | None,
    run_dir: Path | None,
    output_dir: Path,
    device_name: str,
    port: int,
    bind_host: str = "0.0.0.0",
    display_host: str | None = None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import viser
        from nerfview import CameraState, RenderTabState, Viewer
        from nerfview._renderer import RenderTask, Renderer
        from gsplat.rendering import rasterization
    except ImportError as exc:
        raise SystemExit(
            "Missing viewer dependency. Install viser, nerfview, and gsplat in the CUDA environment."
        ) from exc

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for real-time gsplat viewing, but torch.cuda.is_available() is false.")
    device = torch.device(device_name)
    splats, sh_degree = load_splats(checkpoint, device)
    scene_options = scene_options or ([scene] if scene is not None else [])
    active_scene = _scene_for_checkpoint(checkpoint, scene_options) or scene
    camera_poses = load_scene_camera_poses(active_scene, mode)
    state = {
        "checkpoint": checkpoint,
        "scene": active_scene,
        "checkpoint_mtime": checkpoint.stat().st_mtime,
        "splats": splats,
        "sh_degree": sh_degree,
        "filter_key": None,
        "filtered_splats": None,
        "filtered_count": int(splats["means"].shape[0]),
        "camera_poses": camera_poses,
    }
    checkpoint_options = checkpoint_options or [checkpoint]
    scene_labels, scene_by_label = _scene_choices(scene_options)
    current_scene_label = _scene_label(active_scene) if active_scene is not None else "custom checkpoint"
    if active_scene is not None and current_scene_label not in scene_by_label:
        scene_labels.insert(0, current_scene_label)
        scene_by_label[current_scene_label] = active_scene
    checkpoint_labels, checkpoint_by_label = _checkpoint_choices(checkpoint_options)
    current_label = _checkpoint_label(checkpoint)
    if current_label not in checkpoint_by_label:
        checkpoint_labels.insert(0, current_label)
        checkpoint_by_label[current_label] = checkpoint
    auto_load_newest_on_refresh = checkpoint_arg is None and run_dir is None

    class GsplatRenderTabState(RenderTabState):
        total_gs_count: int = 0
        filtered_gs_count: int = 0
        rendered_gs_count: int = 0
        min_opacity: float = 0.0
        max_scale: float = 0.0
        max_radius: float = 0.0
        max_gaussians: int = 0
        max_sh_degree: int = sh_degree
        near_plane: float = 1e-2
        far_plane: float = 1e2
        radius_clip: float = 0.0
        eps2d: float = 0.3
        backgrounds: tuple[float, float, float] = (0.0, 0.0, 0.0)
        render_mode: Literal["rgb", "depth(accumulated)", "depth(expected)", "alpha"] = "rgb"
        normalize_nearfar: bool = False
        inverse: bool = False
        colormap: Literal["turbo", "viridis", "magma", "inferno", "cividis", "gray"] = "turbo"
        rasterize_mode: Literal["classic", "antialiased"] = "classic"
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
        camera_path_type: Literal["train", "interp", "ellipse", "spiral"] = "interp"
        camera_path_fps: int = 30
        camera_path_duration: float = 8.0
        camera_path_export_width: int = 1280
        camera_path_export_height: int = 720

    class GsplatViewer(Viewer):
        def __init__(self, server: viser.ViserServer, render_fn: Callable, output_dir: Path):
            self._render_modes = ("rgb", "depth(accumulated)", "depth(expected)", "alpha")
            self._path_stops: dict[int, threading.Event] = {}
            self._path_threads: dict[int, threading.Thread] = {}
            self._path_active: set[int] = set()
            self._export_thread: threading.Thread | None = None
            super().__init__(server, render_fn, output_dir, "rendering")
            server.gui.set_panel_label("gsplat viewer")

        def _init_rendering_tab(self):
            self.render_tab_state = GsplatRenderTabState()
            self._rendering_tab_handles = {}
            self._rendering_folder = self.server.gui.add_folder("Rendering")

        def _populate_rendering_tab(self):
            server = self.server
            with self._rendering_folder:
                with server.gui.add_folder("Gsplat"):
                    scene_handle = None
                    if scene_labels:
                        scene_handle = server.gui.add_dropdown(
                            "Scene",
                            tuple(scene_labels),
                            initial_value=current_scene_label,
                            hint="Scene directory to inspect. Checkpoints refresh when this changes.",
                        )
                    checkpoint_handle = server.gui.add_dropdown(
                        "Checkpoint",
                        tuple(checkpoint_labels),
                        initial_value=current_label,
                        hint="Loaded Gaussian checkpoint.",
                    )
                    refresh_handle = server.gui.add_button("Refresh Checkpoints")
                    total_handle = server.gui.add_number("Total", initial_value=0, disabled=True)
                    filtered_handle = server.gui.add_number("Kept", initial_value=0, disabled=True)
                    rendered_handle = server.gui.add_number("Rendered", initial_value=0, disabled=True)
                    min_opa_handle = server.gui.add_number(
                        "Min Opacity",
                        initial_value=self.render_tab_state.min_opacity,
                        min=0.0,
                        max=1.0,
                        step=0.005,
                    )
                    max_scale_handle = server.gui.add_number(
                        "Max Scale",
                        initial_value=self.render_tab_state.max_scale,
                        min=0.0,
                        max=10.0,
                        step=0.005,
                    )
                    max_radius_handle = server.gui.add_number(
                        "Max Radius",
                        initial_value=self.render_tab_state.max_radius,
                        min=0.0,
                        max=100.0,
                        step=0.05,
                    )
                    max_gaussians_handle = server.gui.add_number(
                        "Max Count",
                        initial_value=self.render_tab_state.max_gaussians,
                        min=0,
                        max=10_000_000,
                        step=1000,
                    )
                    clean_preset_handle = server.gui.add_button("Clean Preset")
                    no_clean_handle = server.gui.add_button("No Clean")
                    sh_handle = server.gui.add_number("Max SH", initial_value=sh_degree, min=0, max=sh_degree, step=1)
                    near_far_handle = server.gui.add_vector2(
                        "Near/Far",
                        initial_value=(self.render_tab_state.near_plane, self.render_tab_state.far_plane),
                        min=(1e-3, 1e1),
                        max=(1e1, 1e4),
                        step=1e-3,
                    )
                    radius_handle = server.gui.add_number("Radius Clip", initial_value=0.0, min=0.0, max=100.0, step=1.0)
                    eps_handle = server.gui.add_number("2D Epsilon", initial_value=0.3, min=0.0, max=1.0, step=0.01)
                    bg_handle = server.gui.add_rgb("Background", initial_value=(0, 0, 0))
                    mode_handle = server.gui.add_dropdown("Render Mode", self._render_modes, initial_value="rgb")
                    normalize_handle = server.gui.add_checkbox("Normalize Near/Far", initial_value=False, disabled=True)
                    inverse_handle = server.gui.add_checkbox("Inverse", initial_value=False, disabled=True)
                    colormap_handle = server.gui.add_dropdown(
                        "Colormap", ("turbo", "viridis", "magma", "inferno", "cividis", "gray"), initial_value="turbo"
                    )
                    aa_handle = server.gui.add_dropdown("Anti-Aliasing", ("classic", "antialiased"), initial_value="classic")
                    camera_handle = server.gui.add_dropdown("Camera", ("pinhole", "ortho", "fisheye"), initial_value="pinhole")
                    suppress_scene_update = False
                    suppress_checkpoint_update = False

                with server.gui.add_folder("Camera Path"):
                    path_handle = server.gui.add_dropdown(
                        "Path",
                        CAMERA_PATH_TYPES,
                        initial_value=self.render_tab_state.camera_path_type,
                    )
                    path_fps_handle = server.gui.add_number(
                        "FPS", initial_value=self.render_tab_state.camera_path_fps, min=1, max=120, step=1
                    )
                    path_duration_handle = server.gui.add_number(
                        "Duration", initial_value=self.render_tab_state.camera_path_duration, min=1.0, max=120.0, step=0.5
                    )
                    export_width_handle = server.gui.add_number(
                        "Export Width",
                        initial_value=self.render_tab_state.camera_path_export_width,
                        min=64,
                        max=4096,
                        step=16,
                    )
                    export_height_handle = server.gui.add_number(
                        "Export Height",
                        initial_value=self.render_tab_state.camera_path_export_height,
                        min=64,
                        max=4096,
                        step=16,
                    )
                    play_path_handle = server.gui.add_button("Play Path")
                    stop_path_handle = server.gui.add_button("Stop Path")
                    export_path_handle = server.gui.add_button("Export MP4")
                    path_status_handle = server.gui.add_text(
                        "Status",
                        initial_value=_camera_path_status(camera_poses),
                        disabled=True,
                    )

                    def load_checkpoint(selected: Path, event=None, force: bool = False) -> None:
                        nonlocal suppress_scene_update
                        selected_mtime = selected.stat().st_mtime
                        if (
                            not force
                            and selected.resolve() == state["checkpoint"].resolve()
                            and selected_mtime <= float(state["checkpoint_mtime"])
                        ):
                            return
                        new_splats, new_sh_degree = load_splats(selected, device)
                        state["checkpoint"] = selected
                        state["checkpoint_mtime"] = selected_mtime
                        state["splats"] = new_splats
                        state["sh_degree"] = new_sh_degree
                        next_scene = _scene_for_checkpoint(selected, scene_options) or state["scene"]
                        if next_scene is not None and next_scene != state["scene"]:
                            state["scene"] = next_scene
                            state["camera_poses"] = load_scene_camera_poses(next_scene, mode)
                            path_status_handle.value = _camera_path_status(state["camera_poses"])
                            if scene_handle is not None:
                                next_scene_label = _scene_label(next_scene)
                                if next_scene_label in scene_by_label:
                                    suppress_scene_update = True
                                    try:
                                        scene_handle.value = next_scene_label
                                    finally:
                                        suppress_scene_update = False
                        state["filter_key"] = None
                        state["filtered_splats"] = None
                        state["filtered_count"] = int(new_splats["means"].shape[0])
                        self.render_tab_state.max_sh_degree = min(self.render_tab_state.max_sh_degree, new_sh_degree)
                        self.render_tab_state.total_gs_count = int(new_splats["means"].shape[0])
                        self.render_tab_state.filtered_gs_count = int(new_splats["means"].shape[0])
                        sh_handle.max = new_sh_degree
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        print(f"Loaded {new_splats['means'].shape[0]} Gaussians from {selected}", flush=True)
                        if event is not None:
                            self.rerender(event)

                    def refresh_checkpoints(event=None, load_newest: bool = False) -> None:
                        nonlocal checkpoint_labels, checkpoint_by_label, suppress_checkpoint_update
                        current_scene = state["scene"]
                        refreshed = discover_checkpoints(current_scene, mode)
                        if current_scene is None:
                            refreshed = discover_checkpoints(None, mode, checkpoint_arg, run_dir)
                        if not refreshed:
                            refreshed = [state["checkpoint"]]
                        checkpoint_labels, checkpoint_by_label = _checkpoint_choices(refreshed)
                        active_label = None
                        active_resolved = state["checkpoint"].resolve()
                        for label, path in checkpoint_by_label.items():
                            if path.resolve() == active_resolved:
                                active_label = label
                                break
                        if active_label is None:
                            active_label = _checkpoint_label(state["checkpoint"])
                            if active_label in checkpoint_by_label:
                                active_label = str(state["checkpoint"])
                            checkpoint_labels.insert(0, active_label)
                            checkpoint_by_label[active_label] = state["checkpoint"]
                        target_label = checkpoint_labels[0] if load_newest else active_label
                        suppress_checkpoint_update = True
                        try:
                            checkpoint_handle.options = tuple(checkpoint_labels)
                            checkpoint_handle.value = target_label
                        finally:
                            suppress_checkpoint_update = False
                        selected = checkpoint_by_label[target_label]
                        load_checkpoint(selected, event, force=load_newest)

                    if scene_handle is not None:

                        @scene_handle.on_update
                        def _(_) -> None:
                            if suppress_scene_update:
                                return
                            selected_scene = scene_by_label[scene_handle.value]
                            state["scene"] = selected_scene
                            state["camera_poses"] = load_scene_camera_poses(selected_scene, mode)
                            path_status_handle.value = _camera_path_status(state["camera_poses"])
                            refresh_checkpoints(_, load_newest=True)

                    @checkpoint_handle.on_update
                    def _(_) -> None:
                        if suppress_checkpoint_update:
                            return
                        selected = checkpoint_by_label[checkpoint_handle.value]
                        load_checkpoint(selected, _, force=True)

                    @refresh_handle.on_click
                    def _(_) -> None:
                        refresh_checkpoints(_, load_newest=auto_load_newest_on_refresh)

                    @server.on_client_connect
                    def _(_) -> None:
                        refresh_checkpoints(load_newest=auto_load_newest_on_refresh)

                    def update_filter(event=None) -> None:
                        self.render_tab_state.min_opacity = float(min_opa_handle.value)
                        self.render_tab_state.max_scale = float(max_scale_handle.value)
                        self.render_tab_state.max_radius = float(max_radius_handle.value)
                        self.render_tab_state.max_gaussians = int(max_gaussians_handle.value)
                        if event is not None:
                            self.rerender(event)

                    @min_opa_handle.on_update
                    def _(_) -> None:
                        update_filter(_)

                    @max_scale_handle.on_update
                    def _(_) -> None:
                        update_filter(_)

                    @max_radius_handle.on_update
                    def _(_) -> None:
                        update_filter(_)

                    @max_gaussians_handle.on_update
                    def _(_) -> None:
                        update_filter(_)

                    @clean_preset_handle.on_click
                    def _(_) -> None:
                        min_opa_handle.value = 0.05
                        max_scale_handle.value = 0.08
                        max_radius_handle.value = 2.8
                        max_gaussians_handle.value = 120_000
                        update_filter(_)

                    @no_clean_handle.on_click
                    def _(_) -> None:
                        min_opa_handle.value = 0.0
                        max_scale_handle.value = 0.0
                        max_radius_handle.value = 0.0
                        max_gaussians_handle.value = 0
                        update_filter(_)

                    @sh_handle.on_update
                    def _(_) -> None:
                        self.render_tab_state.max_sh_degree = int(sh_handle.value)
                        self.rerender(_)

                    @near_far_handle.on_update
                    def _(_) -> None:
                        self.render_tab_state.near_plane = near_far_handle.value[0]
                        self.render_tab_state.far_plane = near_far_handle.value[1]
                        self.rerender(_)

                    @radius_handle.on_update
                    def _(_) -> None:
                        self.render_tab_state.radius_clip = radius_handle.value
                        self.rerender(_)

                    @eps_handle.on_update
                    def _(_) -> None:
                        self.render_tab_state.eps2d = eps_handle.value
                        self.rerender(_)

                    @bg_handle.on_update
                    def _(_) -> None:
                        self.render_tab_state.backgrounds = bg_handle.value
                        self.rerender(_)

                    @mode_handle.on_update
                    def _(_) -> None:
                        depth_enabled = "depth" in mode_handle.value
                        normalize_handle.disabled = not depth_enabled
                        inverse_handle.disabled = not depth_enabled
                        self.render_tab_state.render_mode = mode_handle.value
                        self.rerender(_)

                    @normalize_handle.on_update
                    def _(_) -> None:
                        self.render_tab_state.normalize_nearfar = normalize_handle.value
                        self.rerender(_)

                    @inverse_handle.on_update
                    def _(_) -> None:
                        self.render_tab_state.inverse = inverse_handle.value
                        self.rerender(_)

                    @colormap_handle.on_update
                    def _(_) -> None:
                        self.render_tab_state.colormap = colormap_handle.value
                        self.rerender(_)

                    @aa_handle.on_update
                    def _(_) -> None:
                        self.render_tab_state.rasterize_mode = aa_handle.value
                        self.rerender(_)

                    @camera_handle.on_update
                    def _(_) -> None:
                        self.render_tab_state.camera_model = camera_handle.value
                        self.rerender(_)

                    def update_path_state() -> None:
                        self.render_tab_state.camera_path_type = path_handle.value
                        self.render_tab_state.camera_path_fps = int(path_fps_handle.value)
                        self.render_tab_state.camera_path_duration = float(path_duration_handle.value)
                        self.render_tab_state.camera_path_export_width = int(export_width_handle.value)
                        self.render_tab_state.camera_path_export_height = int(export_height_handle.value)

                    def make_path() -> list[dict[str, np.ndarray | float]]:
                        update_path_state()
                        frame_count = max(
                            1,
                            int(round(self.render_tab_state.camera_path_fps * self.render_tab_state.camera_path_duration)),
                        )
                        return generate_camera_path(
                            state["camera_poses"],
                            self.render_tab_state.camera_path_type,
                            frame_count,
                        )

                    @path_handle.on_update
                    def _(_) -> None:
                        update_path_state()

                    @path_fps_handle.on_update
                    def _(_) -> None:
                        update_path_state()

                    @path_duration_handle.on_update
                    def _(_) -> None:
                        update_path_state()

                    @export_width_handle.on_update
                    def _(_) -> None:
                        update_path_state()

                    @export_height_handle.on_update
                    def _(_) -> None:
                        update_path_state()

                    @play_path_handle.on_click
                    def _(_) -> None:
                        path = make_path()
                        if not path:
                            path_status_handle.value = "no camera path"
                            return
                        self.start_camera_path_playback(_.client, path, self.render_tab_state.camera_path_fps, path_status_handle)

                    @stop_path_handle.on_click
                    def _(_) -> None:
                        self.stop_camera_path_playback(_.client.client_id)
                        path_status_handle.value = "stopped"

                    @export_path_handle.on_click
                    def _(_) -> None:
                        path = make_path()
                        if not path:
                            path_status_handle.value = "no camera path"
                            return
                        self.export_camera_path_video(
                            path,
                            self.render_tab_state.camera_path_fps,
                            self.render_tab_state.camera_path_export_width,
                            self.render_tab_state.camera_path_export_height,
                            path_status_handle,
                        )

            self._rendering_tab_handles.update(
                {
                    "total": total_handle,
                    "filtered": filtered_handle,
                    "rendered": rendered_handle,
                }
            )
            super()._populate_rendering_tab()

        def _after_render(self):
            self._rendering_tab_handles["total"].value = self.render_tab_state.total_gs_count
            self._rendering_tab_handles["filtered"].value = self.render_tab_state.filtered_gs_count
            self._rendering_tab_handles["rendered"].value = self.render_tab_state.rendered_gs_count

        def _connect_client(self, client):
            client_id = client.client_id
            self._renderers[client_id] = Renderer(viewer=self, client=client, lock=self.lock)
            self._renderers[client_id].start()

            @client.camera.on_update
            def _(_camera):
                if client_id in self._path_active:
                    return
                self._last_move_time = time.time()
                with self.server.atomic():
                    camera_state = self.get_camera_state(client)
                    self._renderers[client_id].submit(RenderTask("move", camera_state))

        def _disconnect_client(self, client):
            client_id = client.client_id
            self.stop_camera_path_playback(client_id)
            self._path_active.discard(client_id)
            if client_id in self._renderers:
                self._renderers[client_id].running = False
                self._renderers.pop(client_id)

        def start_camera_path_playback(self, client, path: list[dict[str, np.ndarray | float]], fps: int, status_handle) -> None:
            client_id = int(client.client_id)
            self.stop_camera_path_playback(client_id)
            stop_event = threading.Event()
            self._path_stops[client_id] = stop_event
            self._path_active.add(client_id)

            def worker() -> None:
                status_handle.value = "playing"
                delay = 1.0 / max(fps, 1)
                try:
                    for pose in path:
                        if stop_event.is_set():
                            break
                        with self.server.atomic():
                            apply_camera_pose_to_client(client, pose)
                        try:
                            aspect = float(client.camera.aspect)
                        except AssertionError:
                            aspect = 4.0 / 3.0
                        try:
                            camera_state = CameraState(float(pose["fov"]), aspect, pose["c2w"].astype(np.float32))
                            self._renderers[client_id].submit(RenderTask("move", camera_state))
                        except KeyError:
                            break
                        time.sleep(delay)
                    status_handle.value = "idle" if not stop_event.is_set() else "stopped"
                    time.sleep(0.25)
                finally:
                    self._path_active.discard(client_id)

            self._path_threads[client_id] = threading.Thread(target=worker, daemon=True)
            self._path_threads[client_id].start()

        def stop_camera_path_playback(self, client_id: int | None = None) -> None:
            if client_id is None:
                for stop_event in self._path_stops.values():
                    stop_event.set()
                return
            stop_event = self._path_stops.get(int(client_id))
            if stop_event is not None:
                stop_event.set()

        def export_camera_path_video(
            self,
            path: list[dict[str, np.ndarray | float]],
            fps: int,
            width: int,
            height: int,
            status_handle,
        ) -> None:
            if self._export_thread is not None and self._export_thread.is_alive():
                status_handle.value = "export already running"
                return

            def worker() -> None:
                status_handle.value = "exporting"
                video_dir = state["checkpoint"].parent / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                output = video_dir / f"traj_{self.render_tab_state.camera_path_type}.mp4"
                old_preview = self.render_tab_state.preview_render
                old_width = self.render_tab_state.render_width
                old_height = self.render_tab_state.render_height
                try:
                    import imageio.v2 as imageio

                    self.render_tab_state.preview_render = True
                    self.render_tab_state.render_width = int(width)
                    self.render_tab_state.render_height = int(height)
                    aspect = float(width) / max(float(height), 1.0)
                    with self.lock, imageio.get_writer(output, fps=max(fps, 1), quality=8) as writer:
                        for index, pose in enumerate(path):
                            camera_state = CameraState(float(pose["fov"]), aspect, pose["c2w"].astype(np.float32))
                            frame = render_fn(camera_state, self.render_tab_state)
                            frame_u8 = (np.clip(frame[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
                            writer.append_data(frame_u8)
                            if index % max(fps, 1) == 0:
                                status_handle.value = f"exporting {index + 1}/{len(path)}"
                    status_handle.value = f"saved {output.name}"
                    print(f"Camera path video saved to {output}", flush=True)
                except Exception as exc:
                    status_handle.value = f"export failed: {exc}"
                    print(f"Camera path export failed: {exc}", flush=True)
                finally:
                    self.render_tab_state.preview_render = old_preview
                    self.render_tab_state.render_width = old_width
                    self.render_tab_state.render_height = old_height

            self._export_thread = threading.Thread(target=worker, daemon=True)
            self._export_thread.start()

    @torch.inference_mode()
    def render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        splats = filtered_splats_for_render(state, render_tab_state)
        active_sh_degree = int(state["sh_degree"])
        width = render_tab_state.render_width if render_tab_state.preview_render else render_tab_state.viewer_width
        height = render_tab_state.render_height if render_tab_state.preview_render else render_tab_state.viewer_height
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)
        background = torch.tensor([render_tab_state.backgrounds], dtype=torch.float32, device=device) / 255.0
        colors, alphas, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"].exp(),
            opacities=splats["opacities"].sigmoid(),
            colors=splats["colors"],
            viewmats=torch.linalg.inv_ex(c2w[None]).inverse,
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, active_sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=background,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
        )
        render_tab_state.total_gs_count = state["splats"]["means"].shape[0]
        render_tab_state.filtered_gs_count = splats["means"].shape[0]
        radii = info.get("radii")
        render_tab_state.rendered_gs_count = int((radii > 0).all(-1).sum().item()) if radii is not None else 0
        if render_tab_state.render_mode == "rgb":
            return colors[0, ..., :3].clamp(0, 1).cpu().numpy()
        if render_tab_state.render_mode in ("depth(accumulated)", "depth(expected)"):
            depth = colors[0, ..., :1]
            near = render_tab_state.near_plane if render_tab_state.normalize_nearfar else depth.min()
            far = render_tab_state.far_plane if render_tab_state.normalize_nearfar else depth.max()
            depth_norm = torch.clip((depth - near) / (far - near + 1e-10), 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            return apply_float_colormap(depth_norm, render_tab_state.colormap).cpu().numpy()
        alpha = alphas[0, ..., :1]
        if render_tab_state.inverse:
            alpha = 1 - alpha
        return apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    server = viser.ViserServer(host=bind_host, port=port, verbose=False)
    GsplatViewer(server=server, render_fn=render_fn, output_dir=output_dir)
    host = display_host or "localhost"
    print(f"Loaded {state['splats']['means'].shape[0]} Gaussians from {checkpoint}")
    print(f"Viewer running at http://{host}:{port} . Press Ctrl+C to exit.")
    while True:
        time.sleep(3600)


def filtered_splats_for_render(state: dict, render_tab_state) -> dict[str, torch.Tensor]:
    splats = state["splats"]
    min_opacity = float(getattr(render_tab_state, "min_opacity", 0.0))
    max_scale = float(getattr(render_tab_state, "max_scale", 0.0))
    max_radius = float(getattr(render_tab_state, "max_radius", 0.0))
    max_gaussians = int(getattr(render_tab_state, "max_gaussians", 0))
    key = (
        state["checkpoint"],
        float(state["checkpoint_mtime"]),
        int(splats["means"].shape[0]),
        min_opacity,
        max_scale,
        max_radius,
        max_gaussians,
    )
    if state.get("filter_key") == key and state.get("filtered_splats") is not None:
        return state["filtered_splats"]

    filtered = _filter_splats(splats, min_opacity, max_scale, max_radius, max_gaussians)
    state["filter_key"] = key
    state["filtered_splats"] = filtered
    state["filtered_count"] = int(filtered["means"].shape[0])
    return filtered


def _camera_path_status(camera_poses: list[dict[str, np.ndarray | float]]) -> str:
    return f"{len(camera_poses)} source cameras" if camera_poses else "no sparse cameras"


def _filter_splats(
    splats: dict[str, torch.Tensor],
    min_opacity: float = 0.0,
    max_scale: float = 0.0,
    max_radius: float = 0.0,
    max_gaussians: int = 0,
) -> dict[str, torch.Tensor]:
    if min_opacity <= 0 and max_scale <= 0 and max_radius <= 0 and max_gaussians <= 0:
        return splats

    means = splats["means"]
    opacities = splats["opacities"].sigmoid()
    scales = splats["scales"].exp().max(dim=-1).values
    radius = torch.linalg.norm(means, dim=-1)
    keep = torch.isfinite(means).all(dim=-1)
    keep &= torch.isfinite(opacities) & torch.isfinite(scales) & torch.isfinite(radius)
    if min_opacity > 0:
        keep &= opacities >= min_opacity
    if max_scale > 0:
        keep &= scales <= max_scale
    if max_radius > 0:
        keep &= radius <= max_radius

    if not bool(keep.any()):
        keep[torch.argmax(opacities)] = True

    if max_gaussians > 0 and int(keep.sum().item()) > max_gaussians:
        score = opacities / scales.clamp_min(1e-6)
        score[~keep] = -torch.inf
        ids = torch.topk(score, k=max_gaussians, largest=True).indices
        capped = torch.zeros_like(keep)
        capped[ids] = True
        keep = capped

    return {
        key: value[keep].contiguous() if torch.is_tensor(value) and value.shape[:1] == keep.shape else value
        for key, value in splats.items()
    }


def load_scene_camera_poses(scene: Path | None, mode: str) -> list[dict[str, np.ndarray | float]]:
    if scene is None:
        return []
    sparse = scene / mode / "sparse" / "0"
    if not has_model(sparse):
        return []
    try:
        model = _normalize_model(read_model(sparse))
    except Exception as exc:
        print(f"Could not load camera path from {sparse}: {exc}", flush=True)
        return []

    poses: list[dict[str, np.ndarray | float]] = []
    for image in sorted(model.images.values(), key=lambda item: item.name):
        camera = model.cameras.get(image.camera_id)
        if camera is None:
            continue
        viewmat = np.eye(4, dtype=np.float64)
        viewmat[:3, :3] = qvec_to_rotmat(image.qvec)
        viewmat[:3, 3] = image.tvec.astype(np.float64)
        c2w = np.linalg.inv(viewmat)
        try:
            intr = camera_matrix(camera)
            fy = float(intr[1, 1])
            fov = 2.0 * math.atan(float(camera.height) / max(2.0 * fy, 1e-8))
        except Exception:
            fov = math.radians(60.0)
        poses.append({"c2w": c2w.astype(np.float64), "fov": float(fov), "name": image.name})
    return poses


def generate_camera_path(
    source_poses: list[dict[str, np.ndarray | float]],
    path_type: str,
    frame_count: int,
) -> list[dict[str, np.ndarray | float]]:
    if not source_poses or frame_count <= 0:
        return []
    if path_type == "train":
        return _sample_train_path(source_poses, frame_count)
    if path_type == "interp":
        return _interpolate_camera_path(source_poses, frame_count)
    if path_type in {"ellipse", "spiral"}:
        return _orbit_camera_path(source_poses, frame_count, spiral=path_type == "spiral")
    return _interpolate_camera_path(source_poses, frame_count)


def _sample_train_path(source_poses: list[dict[str, np.ndarray | float]], frame_count: int) -> list[dict[str, np.ndarray | float]]:
    if frame_count <= len(source_poses):
        ids = np.linspace(0, len(source_poses) - 1, frame_count).round().astype(int)
        return [_copy_pose(source_poses[int(idx)]) for idx in ids]
    return _interpolate_camera_path(source_poses, frame_count)


def _interpolate_camera_path(
    source_poses: list[dict[str, np.ndarray | float]],
    frame_count: int,
) -> list[dict[str, np.ndarray | float]]:
    if len(source_poses) == 1:
        return [_copy_pose(source_poses[0]) for _ in range(frame_count)]
    c2ws = [np.asarray(pose["c2w"], dtype=np.float64) for pose in source_poses]
    fovs = [float(pose["fov"]) for pose in source_poses]
    out = []
    for t in np.linspace(0.0, len(c2ws) - 1, frame_count):
        lo = int(np.floor(t))
        hi = min(lo + 1, len(c2ws) - 1)
        u = float(t - lo)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 3] = (1.0 - u) * c2ws[lo][:3, 3] + u * c2ws[hi][:3, 3]
        q0 = rotmat_to_qvec(c2ws[lo][:3, :3])
        q1 = rotmat_to_qvec(c2ws[hi][:3, :3])
        c2w[:3, :3] = qvec_to_rotmat(_slerp_quat(q0, q1, u))
        fov = (1.0 - u) * fovs[lo] + u * fovs[hi]
        out.append({"c2w": c2w, "fov": float(fov)})
    return out


def _orbit_camera_path(
    source_poses: list[dict[str, np.ndarray | float]],
    frame_count: int,
    spiral: bool,
) -> list[dict[str, np.ndarray | float]]:
    c2ws = [np.asarray(pose["c2w"], dtype=np.float64) for pose in source_poses]
    positions = np.stack([c2w[:3, 3] for c2w in c2ws], axis=0)
    center = np.median(positions, axis=0)
    offsets = positions - center[None, :]
    up = _normalize(np.mean(np.stack([-c2w[:3, 1] for c2w in c2ws], axis=0), axis=0), np.array([0.0, 0.0, 1.0]))
    radii = np.linalg.norm(offsets, axis=-1)
    radius = float(np.median(radii[radii > 1e-6])) if np.any(radii > 1e-6) else 1.0
    x_axis = offsets[int(np.argmax(radii))] if np.any(radii > 1e-6) else np.array([1.0, 0.0, 0.0])
    x_axis = x_axis - float(x_axis @ up) * up
    x_axis = _normalize(x_axis, _orthogonal_axis(up))
    y_axis = _normalize(np.cross(up, x_axis), _orthogonal_axis(x_axis))
    fov = float(np.median([float(pose["fov"]) for pose in source_poses]))
    z_amp = radius * 0.25 if spiral else 0.0

    out = []
    for idx, theta in enumerate(np.linspace(0.0, 2.0 * math.pi, frame_count, endpoint=False)):
        z_offset = z_amp * math.sin(theta * 2.0)
        position = center + radius * math.cos(theta) * x_axis + radius * math.sin(theta) * y_axis + z_offset * up
        c2w = look_at_c2w(position, center, up)
        out.append({"c2w": c2w, "fov": fov, "target": center.copy()})
    return out


def apply_camera_pose_to_client(client, pose: dict[str, np.ndarray | float]) -> None:
    if getattr(client.camera._state, "update_timestamp", 0.0) == 0.0:
        return
    c2w = np.asarray(pose["c2w"], dtype=np.float64)
    position = c2w[:3, 3]
    look_direction = _normalize(c2w[:3, 2], np.array([0.0, 0.0, 1.0]))
    up_direction = _normalize(-c2w[:3, 1], np.array([0.0, 0.0, 1.0]))
    target = pose.get("target")
    if target is not None:
        look_at = np.asarray(target, dtype=np.float64)
    else:
        look_distance = max(float(np.linalg.norm(position)), 1.0)
        look_at = position + look_direction * look_distance
    client.camera.up_direction = up_direction
    client.camera.position = position
    client.camera.look_at = look_at
    client.camera.fov = float(pose["fov"])


def look_at_c2w(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = _normalize(np.asarray(up, dtype=np.float64), np.array([0.0, 0.0, 1.0]))
    z_axis = _normalize(target - position, np.array([0.0, 0.0, 1.0]))
    y_axis = -up
    y_axis = y_axis - float(y_axis @ z_axis) * z_axis
    y_axis = _normalize(y_axis, _orthogonal_axis(z_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis), _orthogonal_axis(z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis), y_axis)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    c2w[:3, 3] = position
    return c2w


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = _normalize(np.asarray(q0, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))
    q1 = _normalize(np.asarray(q1, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))
    dot = float(q0 @ q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _normalize((1.0 - t) * q0 + t * q1, q0)
    theta_0 = math.acos(max(min(dot, 1.0), -1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return _normalize(s0 * q0 + s1 * q1, q0)


def _copy_pose(pose: dict[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    return {"c2w": np.asarray(pose["c2w"], dtype=np.float64).copy(), "fov": float(pose["fov"])}


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _orthogonal_axis(axis: np.ndarray) -> np.ndarray:
    axis = _normalize(axis, np.array([0.0, 0.0, 1.0]))
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(float(candidate @ axis)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    return _normalize(candidate - float(candidate @ axis) * axis, np.array([1.0, 0.0, 0.0]))


def load_splats(checkpoint: Path, device: torch.device) -> tuple[torch.nn.ParameterDict, int]:
    data = torch.load(checkpoint, map_location=device)
    raw = data.get("splats")
    if raw is None:
        raw = _legacy_checkpoint_to_splats(data)
    required = {"means", "quats", "scales", "opacities"}
    missing = required - set(raw)
    if missing:
        raise ViewerCheckpointError(f"Checkpoint is missing required splat tensors: {sorted(missing)}")
    if "colors" in raw:
        colors = raw["colors"]
    elif "sh0" in raw and "shN" in raw:
        colors = torch.cat([raw["sh0"], raw["shN"]], dim=-2)
    elif "sh0" in raw:
        colors = raw["sh0"]
    else:
        raise ViewerCheckpointError("Checkpoint is missing SH color tensors: expected colors or sh0/shN.")
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    if (sh_degree + 1) ** 2 != colors.shape[-2]:
        raise ViewerCheckpointError(f"Invalid SH coefficient count: {colors.shape[-2]}")
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(raw["means"].float().to(device).contiguous(), requires_grad=False),
            "quats": torch.nn.Parameter(F.normalize(raw["quats"].float().to(device), p=2, dim=-1).contiguous(), requires_grad=False),
            "scales": torch.nn.Parameter(raw["scales"].float().to(device).contiguous(), requires_grad=False),
            "opacities": torch.nn.Parameter(raw["opacities"].float().to(device).contiguous(), requires_grad=False),
            "colors": torch.nn.Parameter(colors.float().to(device).contiguous(), requires_grad=False),
        }
    )
    return splats, sh_degree


def _legacy_checkpoint_to_splats(data: dict) -> dict[str, torch.Tensor]:
    if "colors" not in data:
        raise ViewerCheckpointError("Checkpoint has no 'splats' state and no legacy 'colors' tensor.")
    colors = data["colors"]
    if colors.ndim == 2:
        colors = _rgb_to_sh(colors).unsqueeze(1)
    return {
        "means": data["means"],
        "quats": data["quats"],
        "scales": data["scales"].clamp_min(1e-8).log(),
        "opacities": torch.logit(data["opacities"].clamp(1e-6, 1.0 - 1e-6)),
        "colors": colors,
    }


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    img = torch.nan_to_num(img, 0.0).clamp(0.0, 1.0)
    if img.shape[-1] != 1:
        raise ValueError(f"Expected single-channel image, got shape {tuple(img.shape)}")
    if colormap == "gray":
        return img.repeat_interleave(3, dim=-1)

    stops = _colormap_stops(colormap).to(device=img.device, dtype=img.dtype)
    x = img[..., 0] * (stops.shape[0] - 1)
    lower = torch.floor(x).long().clamp(0, stops.shape[0] - 1)
    upper = (lower + 1).clamp(0, stops.shape[0] - 1)
    weight = (x - lower.to(x.dtype))[..., None]
    return stops[lower] * (1.0 - weight) + stops[upper] * weight


def _colormap_stops(name: str) -> torch.Tensor:
    palettes = {
        "turbo": [
            (48, 18, 59),
            (45, 97, 191),
            (36, 188, 169),
            (246, 230, 70),
            (220, 50, 32),
        ],
        "viridis": [
            (68, 1, 84),
            (59, 82, 139),
            (33, 145, 140),
            (94, 201, 98),
            (253, 231, 37),
        ],
        "magma": [
            (0, 0, 4),
            (80, 18, 123),
            (182, 54, 121),
            (251, 136, 97),
            (252, 253, 191),
        ],
        "inferno": [
            (0, 0, 4),
            (87, 15, 109),
            (187, 55, 84),
            (249, 142, 8),
            (252, 255, 164),
        ],
        "cividis": [
            (0, 32, 76),
            (46, 80, 110),
            (101, 121, 111),
            (165, 162, 104),
            (253, 234, 69),
        ],
    }
    values = palettes.get(name, palettes["turbo"])
    return torch.tensor(values, dtype=torch.float32) / 255.0


def _require_checkpoint(path: Path) -> Path:
    if not path.exists():
        raise ViewerCheckpointError(f"Missing Gaussian checkpoint: {path}")
    return path


if __name__ == "__main__":
    main()
