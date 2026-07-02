from __future__ import annotations

import numpy as np
from PIL import Image

from vggt_gaussian_reconstruction.frame_selection import (
    anchor_subset,
    parallax_subset,
    quality_subset,
    score_frame,
    score_sequence,
    uniform_subset,
)


def test_uniform_subset_keeps_requested_count(tmp_path):
    paths = []
    for i in range(10):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full((8, 8, 3), i, dtype=np.uint8)).save(path)
        paths.append(path)
    selected = uniform_subset(paths, 4)
    assert len(selected) == 4
    assert selected[0] == paths[0]
    assert selected[-1] == paths[-1]


def test_quality_subset_scores_frames(tmp_path):
    paths = []
    for i in range(6):
        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        arr[:, ::2] = i * 30
        path = tmp_path / f"{i}.png"
        Image.fromarray(arr).save(path)
        paths.append(path)
    scores = [score_frame(path, i) for i, path in enumerate(paths)]
    selected = quality_subset(scores, 3)
    assert len(selected) == 3
    assert selected == sorted(selected)


def test_anchor_and_parallax_subset_keep_requested_count(tmp_path):
    paths = []
    for i in range(8):
        arr = np.zeros((24, 24, 3), dtype=np.uint8)
        arr[:, i % 4 :: 4] = 255
        path = tmp_path / f"{i}.png"
        Image.fromarray(arr).save(path)
        paths.append(path)

    scores = score_sequence(paths)

    anchor_selected = anchor_subset(scores, 4)
    parallax_selected = parallax_subset(scores, 4)

    assert len(anchor_selected) == 4
    assert len(parallax_selected) == 4
    assert anchor_selected == sorted(anchor_selected)
    assert parallax_selected == sorted(parallax_selected)
    assert any(score.anchor_score != 0.0 for score in scores)
