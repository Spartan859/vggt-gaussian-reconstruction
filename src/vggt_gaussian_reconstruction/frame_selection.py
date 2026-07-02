from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class FrameScore:
    path: Path
    index: int
    blur: float
    brightness: float
    score: float
    novelty: float = 0.0
    anchor_score: float = 0.0


def extract_candidate_frames(video: Path, out_dir: Path, candidate_count: int, overwrite: bool = False) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old in out_dir.glob("candidate_*.png"):
            old.unlink()
    existing = sorted(out_dir.glob("candidate_*.png"))
    if len(existing) >= candidate_count:
        return existing[:candidate_count]

    duration = ffprobe_duration(video)
    fps = max(candidate_count / max(duration, 1e-6), 0.1)
    pattern = out_dir / "candidate_%05d.png"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps:.8f},scale='min(1280,iw)':-2",
        "-frames:v",
        str(candidate_count),
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("candidate_*.png"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames from {video}")
    return frames


def select_frames(video: Path, image_dir: Path, num_frames: int, strategy: str, candidate_multiplier: int = 4) -> dict:
    if strategy not in {"uniform", "quality", "anchor", "parallax"}:
        raise ValueError("--strategy must be one of: uniform, quality, anchor, parallax")
    image_dir.mkdir(parents=True, exist_ok=True)
    candidate_count = max(num_frames, num_frames * candidate_multiplier)
    with tempfile.TemporaryDirectory(prefix="frames_") as tmp:
        candidates = extract_candidate_frames(video, Path(tmp), candidate_count, overwrite=True)
        if strategy == "uniform":
            selected = uniform_subset(candidates, num_frames)
            scores = [score_frame(p, i) for i, p in enumerate(candidates)]
        elif strategy == "quality":
            scores = [score_frame(p, i) for i, p in enumerate(candidates)]
            selected = quality_subset(scores, num_frames)
        else:
            scores = score_sequence(candidates)
            if strategy == "anchor":
                selected = anchor_subset(scores, num_frames)
            else:
                selected = parallax_subset(scores, num_frames)

        selected_paths = []
        for out_idx, src in enumerate(selected):
            dst = image_dir / f"frame_{out_idx:04d}.png"
            with Image.open(src) as im:
                im.convert("RGB").save(dst)
            selected_paths.append(dst)

    metadata = {
        "video": str(video),
        "strategy": strategy,
        "num_frames": len(selected_paths),
        "frames": [p.name for p in selected_paths],
        "candidate_scores": [
            {
                "index": s.index,
                "blur": s.blur,
                "brightness": s.brightness,
                "score": s.score,
                "novelty": s.novelty,
                "anchor_score": s.anchor_score,
            }
            for s in scores
        ],
    }
    return metadata


def write_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def uniform_subset(paths: list[Path], num_frames: int) -> list[Path]:
    if len(paths) <= num_frames:
        return paths
    idxs = np.linspace(0, len(paths) - 1, num_frames).round().astype(int)
    return [paths[int(i)] for i in idxs]


def quality_subset(scores: list[FrameScore], num_frames: int) -> list[Path]:
    if len(scores) <= num_frames:
        return [s.path for s in scores]
    min_gap = max(1, len(scores) // max(num_frames * 2, 1))
    selected: list[FrameScore] = []
    for score in sorted(scores, key=lambda x: x.score, reverse=True):
        if all(abs(score.index - old.index) >= min_gap for old in selected):
            selected.append(score)
        if len(selected) == num_frames:
            break
    if len(selected) < num_frames:
        used = {s.index for s in selected}
        for score in sorted(scores, key=lambda x: x.index):
            if score.index not in used:
                selected.append(score)
            if len(selected) == num_frames:
                break
    return [s.path for s in sorted(selected, key=lambda x: x.index)]


def anchor_subset(scores: list[FrameScore], num_frames: int) -> list[Path]:
    """Choose a stable anchor first, then fill with quality-spaced frames."""
    if len(scores) <= num_frames:
        return [s.path for s in scores]
    anchor = max(scores, key=lambda x: x.anchor_score)
    selected: list[FrameScore] = [anchor]
    remaining = [s for s in scores if s.index != anchor.index]
    selected.extend(_diverse_quality_subset(remaining, num_frames - 1, prefer_novelty=False))
    return [s.path for s in sorted(selected, key=lambda x: x.index)]


def parallax_subset(scores: list[FrameScore], num_frames: int) -> list[Path]:
    """Prefer sharp frames that add temporal/appearance novelty."""
    if len(scores) <= num_frames:
        return [s.path for s in scores]
    selected = _diverse_quality_subset(scores, num_frames, prefer_novelty=True)
    return [s.path for s in sorted(selected, key=lambda x: x.index)]


def score_sequence(paths: list[Path]) -> list[FrameScore]:
    scores = [score_frame(path, i) for i, path in enumerate(paths)]
    thumbnails = [_gray_thumbnail(path) for path in paths]
    if not thumbnails:
        return scores
    novelty = []
    for i, gray in enumerate(thumbnails):
        prev_diff = float(np.mean(np.abs(gray - thumbnails[i - 1]))) if i > 0 else 0.0
        next_diff = float(np.mean(np.abs(gray - thumbnails[i + 1]))) if i + 1 < len(thumbnails) else 0.0
        novelty.append(max(prev_diff, next_diff))
    max_novelty = max(max(novelty), 1e-6)
    center = (len(scores) - 1) / 2.0
    for score, nov in zip(scores, novelty):
        score.novelty = float(nov / max_novelty)
        center_bias = 1.0 - abs(score.index - center) / max(center, 1.0)
        score.anchor_score = float(score.score + 0.35 * center_bias + 0.25 * score.novelty)
    return scores


def _diverse_quality_subset(scores: list[FrameScore], num_frames: int, *, prefer_novelty: bool) -> list[FrameScore]:
    if num_frames <= 0:
        return []
    min_gap = max(1, len(scores) // max(num_frames * 2, 1))
    selected: list[FrameScore] = []

    def rank(score: FrameScore) -> float:
        novelty_weight = 0.7 if prefer_novelty else 0.25
        return score.score + novelty_weight * score.novelty

    for score in sorted(scores, key=rank, reverse=True):
        if all(abs(score.index - old.index) >= min_gap for old in selected):
            selected.append(score)
        if len(selected) == num_frames:
            break
    if len(selected) < num_frames:
        used = {s.index for s in selected}
        for score in sorted(scores, key=lambda x: x.index):
            if score.index not in used:
                selected.append(score)
            if len(selected) == num_frames:
                break
    return selected


def score_frame(path: Path, index: int) -> FrameScore:
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    blur = laplacian_variance(gray)
    brightness = float(gray.mean())
    exposure_penalty = abs(brightness - 0.5) * 0.5
    score = float(np.log1p(blur) - exposure_penalty)
    return FrameScore(path=path, index=index, blur=float(blur), brightness=brightness, score=score)


def _gray_thumbnail(path: Path, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    with Image.open(path) as im:
        gray = im.convert("L").resize(size)
    return np.asarray(gray, dtype=np.float32) / 255.0


def laplacian_variance(gray: np.ndarray) -> float:
    center = -4.0 * gray[1:-1, 1:-1]
    lap = center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def ffprobe_duration(video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)
