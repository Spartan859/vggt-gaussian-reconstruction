from __future__ import annotations

import os
import random
import shutil
import urllib.error
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
from vggt.dependency.track_predict import predict_tracks


@dataclass
class VggtConfig:
    scene: Path
    device: str = "cuda"
    seed: int = 42
    conf_threshold: float = 5.0
    max_points: int = 100_000
    max_reproj_error: float = 8.0
    vis_thresh: float = 0.2
    query_frame_num: int = 8
    max_query_pts: int = 4096
    camera_type: str = "SIMPLE_PINHOLE"
    shared_camera: bool = False
    fine_tracking: bool = True
    query_frame_strategy: str = "linspace"
    extra_args: list[str] = field(default_factory=list)


def run_vggt_package(config: VggtConfig) -> Path:
    _seed_everything(config.seed)
    _configure_torch_hub()
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VGGT package inference requires CUDA for the full pipeline.")

    image_dir = config.scene / "images"
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    model = VGGT()
    weights_url = os.environ.get("VGGT_WEIGHTS_URL", "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt")
    weights_path = Path(os.environ.get("VGGT_WEIGHTS_PATH", ""))
    if weights_path.exists():
        print(f"Loading VGGT weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
    else:
        print(f"Downloading VGGT weights to torch cache from {weights_url}")
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                weights_url,
                model_dir=str(weights_path.parent) if str(weights_path) else None,
                file_name=weights_path.name if str(weights_path) else None,
                map_location="cpu",
            )
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(
                "Failed to download VGGT weights. Enable network/proxy on the GPU job, "
                f"set VGGT_WEIGHTS_URL to an accessible mirror, or place model.pt at {weights_path}."
            ) from exc
    model.load_state_dict(state_dict)
    model.eval().to(device)

    fixed_resolution = 518
    load_resolution = 1024
    images, original_coords = load_and_preprocess_images_square([str(p) for p in image_paths], load_resolution)
    images = images.to(device)
    original_coords = original_coords.cpu().numpy()

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        infer_images = F.interpolate(images, size=(fixed_resolution, fixed_resolution), mode="bilinear", align_corners=False)
        aggregated_tokens, ps_idx = model.aggregator(infer_images[None])
        pose_enc = model.camera_head(aggregated_tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, infer_images.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens, infer_images[None], ps_idx)

    extrinsic_np = extrinsic.detach().squeeze(0).cpu().numpy()
    intrinsic_np = intrinsic.detach().squeeze(0).cpu().numpy()
    depth_map_np = depth_map.detach().squeeze(0).cpu().numpy()
    points_3d_map = unproject_depth_map_to_point_map(depth_map_np, extrinsic_np, intrinsic_np)
    depth_conf_for_tracks, points_3d_for_tracks = _prepare_tracker_inputs(depth_conf, points_3d_map)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        _configure_vggsfm_tracker_loader(image_paths=image_paths, strategy=config.query_frame_strategy)
        pred_tracks, pred_vis_scores, _pred_confs, points_3d, points_rgb = predict_tracks(
            images,
            conf=depth_conf_for_tracks,
            points_3d=points_3d_for_tracks,
            masks=None,
            max_query_pts=config.max_query_pts,
            query_frame_num=config.query_frame_num,
            keypoint_extractor="aliked+sp",
            fine_tracking=config.fine_tracking,
        )
        torch.cuda.empty_cache()

    image_size = np.array(images.shape[-2:])
    intrinsic_np[:, :2, :] *= load_resolution / fixed_resolution
    track_masks = pred_vis_scores > config.vis_thresh

    reconstruction, inlier_mask = batch_np_matrix_to_pycolmap(
        points_3d,
        extrinsic_np,
        intrinsic_np,
        pred_tracks,
        image_size,
        masks=track_masks,
        max_reproj_error=config.max_reproj_error,
        shared_camera=config.shared_camera,
        camera_type=config.camera_type,
        points_rgb=points_rgb,
    )
    if reconstruction is None:
        raise RuntimeError(
            "VGGT did not produce enough multi-view observations for BA. "
            f"Generated {points_3d.shape[0]} tracked points, but no valid reconstruction was built."
        )
    _rename_and_rescale(
        reconstruction,
        [p.name for p in image_paths],
        original_coords,
        load_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=config.shared_camera,
    )

    sparse_zero = config.scene / "vggt" / "sparse" / "0"
    _write_pycolmap_reconstruction(reconstruction, sparse_zero)
    return sparse_zero


def _prepare_tracker_inputs(depth_conf: torch.Tensor, points_3d_map: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    # VGGT's tracker indexes these with NumPy arrays and then uses the resulting
    # mask on NumPy colors, so keep them on CPU to avoid implicit CUDA->NumPy conversion.
    conf = depth_conf.detach().squeeze(0).float().cpu()
    points = torch.as_tensor(points_3d_map, dtype=torch.float32, device="cpu")
    return conf, points


def _configure_torch_hub() -> None:
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        torch.hub.set_dir(str(Path(torch_home) / "hub"))


def _configure_vggsfm_tracker_loader(image_paths: Sequence[Path] | None = None, strategy: str = "linspace") -> None:
    from vggt.dependency import track_predict, vggsfm_utils
    from vggt.dependency.vggsfm_tracker import TrackerPredictor

    if strategy not in {"linspace", "quality", "anchor"}:
        raise ValueError("--query-frame-strategy must be one of: linspace, quality, anchor")

    def generate_rank_without_dino(
        images,
        query_frame_num,
        image_size=336,
        model_name="dinov2_vitb14_reg",
        device="cuda",
        spatial_similarity=False,
    ):
        frame_count = int(images.shape[0])
        if frame_count <= 1 or query_frame_num <= 0:
            return []
        count = min(query_frame_num, frame_count)
        if strategy == "linspace" or image_paths is None:
            return np.linspace(0, frame_count - 1, num=count, dtype=int).tolist()
        return _rank_query_frames(image_paths[:frame_count], count, strategy)

    def build_vggsfm_tracker(model_path=None):
        tracker = TrackerPredictor()
        weights_path = Path(
            model_path
            or os.environ.get(
                "VGGSFM_TRACKER_WEIGHTS_PATH",
                str(Path(torch.hub.get_dir()) / "checkpoints" / "vggsfm_v2_tracker.pt"),
            )
        )
        if weights_path.exists():
            print(f"Loading VGGSfM tracker weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu")
        else:
            url = os.environ.get(
                "VGGSFM_TRACKER_WEIGHTS_URL",
                "https://hf-mirror.com/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt",
            )
            print(f"Downloading VGGSfM tracker weights to torch cache from {url}")
            try:
                state_dict = torch.hub.load_state_dict_from_url(
                    url,
                    model_dir=str(weights_path.parent),
                    file_name=weights_path.name,
                    map_location="cpu",
                )
            except (OSError, urllib.error.URLError) as exc:
                raise RuntimeError(
                    "Failed to download VGGSfM tracker weights. Run setup_a800_env.sh weights stage, "
                    f"set VGGSFM_TRACKER_WEIGHTS_URL to an accessible mirror, or place weights at {weights_path}."
                ) from exc
        tracker.load_state_dict(state_dict)
        tracker.eval()
        return tracker

    vggsfm_utils.build_vggsfm_tracker = build_vggsfm_tracker
    track_predict.build_vggsfm_tracker = build_vggsfm_tracker
    vggsfm_utils.generate_rank_by_dino = generate_rank_without_dino
    track_predict.generate_rank_by_dino = generate_rank_without_dino


def _rank_query_frames(image_paths: Sequence[Path], query_frame_num: int, strategy: str) -> list[int]:
    if query_frame_num <= 0 or not image_paths:
        return []
    count = min(query_frame_num, len(image_paths))
    scores = []
    thumbnails = [_gray_thumbnail(path) for path in image_paths]
    novelty = []
    for i, gray in enumerate(thumbnails):
        prev_diff = float(np.mean(np.abs(gray - thumbnails[i - 1]))) if i > 0 else 0.0
        next_diff = float(np.mean(np.abs(gray - thumbnails[i + 1]))) if i + 1 < len(thumbnails) else 0.0
        novelty.append(max(prev_diff, next_diff))
    max_novelty = max(max(novelty), 1e-6)
    center = (len(image_paths) - 1) / 2.0
    for i, path in enumerate(image_paths):
        with Image.open(path) as im:
            gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        blur = _laplacian_variance(gray)
        brightness = float(gray.mean())
        exposure_penalty = abs(brightness - 0.5) * 0.5
        quality = float(np.log1p(blur) - exposure_penalty)
        novelty_score = float(novelty[i] / max_novelty)
        center_bias = 1.0 - abs(i - center) / max(center, 1.0)
        if strategy == "anchor":
            rank = quality + 0.35 * center_bias + 0.25 * novelty_score
        else:
            rank = quality + 0.7 * novelty_score
        scores.append((rank, i))
    ranked = sorted(scores, reverse=True)
    chosen = sorted(i for _rank, i in ranked[:count])
    return chosen


def _gray_thumbnail(path: Path, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    with Image.open(path) as im:
        gray = im.convert("L").resize(size)
    return np.asarray(gray, dtype=np.float32) / 255.0


def _laplacian_variance(gray: np.ndarray) -> float:
    center = -4.0 * gray[1:-1, 1:-1]
    lap = center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def _select_ba_points(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    points_rgb: np.ndarray,
    *,
    max_points: int,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if world_points.ndim != 4:
        raise ValueError(f"Expected world_points to have shape (S, H, W, 3); got {world_points.shape}")
    if world_points_conf.shape != world_points.shape[:-1]:
        raise ValueError(
            f"Expected world_points_conf to have shape {world_points.shape[:-1]}; got {world_points_conf.shape}"
        )
    points_rgb = _normalize_points_rgb(points_rgb, world_points.shape)

    mask = np.isfinite(world_points_conf) & np.all(np.isfinite(world_points), axis=-1)
    mask &= world_points_conf >= conf_threshold
    if not np.any(mask):
        fallback = np.isfinite(world_points_conf) & np.all(np.isfinite(world_points), axis=-1)
        if not np.any(fallback):
            raise RuntimeError("VGGT did not produce any finite world points.")
        mask = fallback

    flat_idx = np.flatnonzero(mask.reshape(-1))
    if flat_idx.size > max_points:
        rng = np.random.default_rng(0)
        flat_idx = rng.choice(flat_idx, size=max_points, replace=False)
    flat_idx.sort()

    s, h, w = world_points.shape[:3]
    frames = flat_idx // (h * w)
    rem = flat_idx % (h * w)
    ys = rem // w
    xs = rem % w

    points_3d = world_points[frames, ys, xs].astype(np.float64)
    points_rgb = points_rgb[frames, ys, xs].astype(np.uint8)
    stats = {"selected": int(points_3d.shape[0]), "frames": int(s), "height": int(h), "width": int(w)}
    return points_3d, points_rgb, stats


def _normalize_points_rgb(points_rgb: np.ndarray, world_points_shape: tuple[int, ...]) -> np.ndarray:
    if points_rgb.ndim != 4:
        raise ValueError(f"Expected points_rgb to be 4D; got {points_rgb.shape}")
    expected_shape = world_points_shape[:-1] + (3,)
    if points_rgb.shape == expected_shape:
        return points_rgb
    if points_rgb.shape[0] == world_points_shape[0] and points_rgb.shape[1] == 3:
        return np.transpose(points_rgb, (0, 2, 3, 1))
    raise ValueError(f"Expected points_rgb to have shape {expected_shape} or (S, 3, H, W); got {points_rgb.shape}")


def _write_pycolmap_reconstruction(reconstruction, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(output_dir))


def _rename_and_rescale(
    reconstruction,
    image_names: list[str],
    original_coords: np.ndarray,
    image_size: int,
    *,
    shift_point2d_to_original_res: bool = False,
    shared_camera: bool = False,
) -> None:
    rescale_camera = True
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        image.name = image_names[image_id - 1]
        real_width, real_height = original_coords[image_id - 1, -2:]
        resize_ratio = max(real_width, real_height) / image_size
        if rescale_camera:
            params = np.array(camera.params, copy=True)
            params *= resize_ratio
            params[-2:] = np.array([real_width / 2.0, real_height / 2.0])
            camera.params = params
            camera.width = int(real_width)
            camera.height = int(real_height)

        if shift_point2d_to_original_res:
            top_left = original_coords[image_id - 1, :2]
            for point2d in image.points2D:
                point2d.xy = (point2d.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
