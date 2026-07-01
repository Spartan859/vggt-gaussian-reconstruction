from __future__ import annotations

import os
import random
import shutil
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class VggtConfig:
    scene: Path
    device: str = "cuda"
    seed: int = 42
    conf_threshold: float = 5.0
    max_points: int = 100_000
    extra_args: list[str] = field(default_factory=list)


def run_vggt_package(config: VggtConfig) -> Path:
    _seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    if device.type != "cuda":
        raise RuntimeError("VGGT package inference requires CUDA for the full pipeline.")

    image_dir = config.scene / "images"
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
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

    extrinsic_np = extrinsic.squeeze(0).cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).cpu().numpy()
    depth_np = depth_map.squeeze(0).cpu().numpy()
    conf_np = depth_conf.squeeze(0).cpu().numpy()
    points_3d = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)

    num_frames, height, width, _ = points_3d.shape
    points_rgb = F.interpolate(images, size=(fixed_resolution, fixed_resolution), mode="bilinear", align_corners=False)
    points_rgb_np = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    conf_mask = randomly_limit_trues(conf_np >= config.conf_threshold, config.max_points)

    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d[conf_mask],
        points_xyf[conf_mask],
        points_rgb_np[conf_mask],
        extrinsic_np,
        intrinsic_np,
        np.array([fixed_resolution, fixed_resolution]),
        shared_camera=False,
        camera_type="PINHOLE",
    )
    _rename_and_rescale(reconstruction, [p.name for p in image_paths], original_coords, fixed_resolution)

    sparse_zero = config.scene / "vggt" / "sparse" / "0"
    _write_pycolmap_reconstruction(reconstruction, sparse_zero)
    return sparse_zero


def _write_pycolmap_reconstruction(reconstruction, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(output_dir))


def _rename_and_rescale(reconstruction, image_names: list[str], original_coords: np.ndarray, image_size: int) -> None:
    for image_id in reconstruction.images:
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        image.name = image_names[image_id - 1]
        real_width, real_height = original_coords[image_id - 1, -2:]
        resize_ratio = max(real_width, real_height) / image_size
        params = np.array(camera.params, copy=True)
        params *= resize_ratio
        params[-2:] = np.array([real_width / 2.0, real_height / 2.0])
        camera.params = params
        camera.width = int(real_width)
        camera.height = int(real_height)

        top_left = original_coords[image_id - 1, :2]
        for point2d in image.points2D:
            point2d.xy = (point2d.xy - top_left) * resize_ratio


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
