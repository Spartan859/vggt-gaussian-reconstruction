from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from pycolmap import Reconstruction as PycolmapReconstruction
except Exception:  # pragma: no cover - optional runtime dependency path
    PycolmapReconstruction = None


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    point3d_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: list[tuple[int, int]]


@dataclass
class Reconstruction:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points3d: dict[int, Point3D]


def read_model(path: Path | str) -> Reconstruction:
    path = Path(path)
    if _has_colmap_bin(path):
        if PycolmapReconstruction is not None:
            model = PycolmapReconstruction(str(path))
            return _from_pycolmap(model)
        raise RuntimeError(
            f"Detected COLMAP binary model in {path}, but pycolmap is not available to read it."
        )
    return Reconstruction(
        cameras=read_cameras_text(path / "cameras.txt"),
        images=read_images_text(path / "images.txt"),
        points3d=read_points3d_text(path / "points3D.txt"),
    )


def write_model(model: Reconstruction, path: Path | str) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    write_cameras_text(model.cameras.values(), path / "cameras.txt")
    write_images_text(model.images.values(), path / "images.txt")
    write_points3d_text(model.points3d.values(), path / "points3D.txt")


def _has_colmap_bin(path: Path) -> bool:
    return any((path / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin"))


def _from_pycolmap(model) -> Reconstruction:
    cameras = {}
    for cam_id, cam in model.cameras.items():
        cam_dict = cam.todict()
        cameras[int(cam_id)] = Camera(
            id=int(cam_id),
            model=str(cam_dict["model"]).split(".")[-1],
            width=int(cam_dict["width"]),
            height=int(cam_dict["height"]),
            params=np.array(cam_dict["params"], dtype=np.float64),
        )

    images = {}
    for image_id, image in model.images.items():
        image_dict = image.todict()
        points2d = image_dict.get("points2D", [])
        xys = np.array([[p["xy"][0], p["xy"][1]] for p in points2d], dtype=np.float64)
        point3d_ids = np.array([p.get("point3D_id", -1) for p in points2d], dtype=np.int64)
        if xys.size == 0:
            xys = np.zeros((0, 2), dtype=np.float64)
        if point3d_ids.size == 0:
            point3d_ids = np.zeros((0,), dtype=np.int64)
        cam_from_world = image_dict["cam_from_world"]
        qvec = np.array(cam_from_world["rotation"]["quat"], dtype=np.float64)
        tvec = np.array(cam_from_world["translation"], dtype=np.float64)
        images[int(image_id)] = Image(
            id=int(image_id),
            qvec=qvec,
            tvec=tvec,
            camera_id=int(image_dict["camera_id"]),
            name=str(image_dict["name"]),
            xys=xys,
            point3d_ids=point3d_ids,
        )

    points3d = {}
    for point_id, point in model.points3D.items():
        point_dict = point.todict()
        track = []
        for elem in point_dict.get("track", {}).get("elements", []):
            track.append((int(elem["image_id"]), int(elem["point2D_idx"])))
        points3d[int(point_id)] = Point3D(
            id=int(point_id),
            xyz=np.array(point_dict["xyz"], dtype=np.float64),
            rgb=np.array(point_dict["color"], dtype=np.uint8),
            error=float(point_dict["error"]),
            track=track,
        )

    return Reconstruction(cameras=cameras, images=images, points3d=points3d)


def read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            cameras[cam_id] = Camera(
                id=cam_id,
                model=parts[1],
                width=int(parts[2]),
                height=int(parts[3]),
                params=np.array([float(x) for x in parts[4:]], dtype=np.float64),
            )
    return cameras


def write_cameras_text(cameras: Iterable[Camera], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam in sorted(cameras, key=lambda x: x.id):
            params = " ".join(_fmt(x) for x in cam.params)
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params}\n")


def read_images_text(path: Path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")]
    i = 0
    while i < len(lines):
        pose = lines[i].split()
        image_id = int(pose[0])
        qvec = np.array([float(x) for x in pose[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in pose[5:8]], dtype=np.float64)
        camera_id = int(pose[8])
        name = " ".join(pose[9:])
        pts_line = lines[i + 1].split() if i + 1 < len(lines) else []
        triples = [pts_line[j : j + 3] for j in range(0, len(pts_line), 3)]
        xys = np.array([[float(t[0]), float(t[1])] for t in triples], dtype=np.float64)
        point_ids = np.array([int(t[2]) for t in triples], dtype=np.int64)
        images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, point_ids)
        i += 2
    return images


def write_images_text(images: Iterable[Image], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for im in sorted(images, key=lambda x: x.id):
            q = " ".join(_fmt(x) for x in im.qvec)
            t = " ".join(_fmt(x) for x in im.tvec)
            f.write(f"{im.id} {q} {t} {im.camera_id} {im.name}\n")
            entries = []
            for xy, pid in zip(im.xys, im.point3d_ids):
                entries.extend([_fmt(xy[0]), _fmt(xy[1]), str(int(pid))])
            f.write(" ".join(entries) + "\n")


def read_points3d_text(path: Path) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    if not path.exists():
        return points
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point_id = int(parts[0])
            xyz = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
            rgb = np.array([int(x) for x in parts[4:7]], dtype=np.uint8)
            error = float(parts[7])
            track_vals = [int(x) for x in parts[8:]]
            track = [(track_vals[i], track_vals[i + 1]) for i in range(0, len(track_vals), 2)]
            points[point_id] = Point3D(point_id, xyz, rgb, error, track)
    return points


def write_points3d_text(points: Iterable[Point3D], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pt in sorted(points, key=lambda x: x.id):
            xyz = " ".join(_fmt(x) for x in pt.xyz)
            rgb = " ".join(str(int(x)) for x in pt.rgb)
            track = " ".join(f"{im_id} {idx}" for im_id, idx in pt.track)
            suffix = f" {track}" if track else ""
            f.write(f"{pt.id} {xyz} {rgb} {_fmt(pt.error)}{suffix}\n")


def camera_matrix(camera: Camera) -> np.ndarray:
    model = camera.model.upper()
    p = camera.params.astype(np.float64)
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif model == "SIMPLE_RADIAL":
        f, cx, cy = p[:3]
        fx = fy = f
    elif model in {"OPENCV", "RADIAL"}:
        fx, fy, cx, cy = p[:4]
    else:
        raise ValueError(f"Unsupported camera model for BA/eval: {camera.model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"
