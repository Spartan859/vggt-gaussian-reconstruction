from __future__ import annotations

import numpy as np

from vggt_gaussian_reconstruction.colmap import (
    _point2d_point3d_id,
    _point2d_xy,
    _track_element_image_id,
    _track_element_point2d_idx,
)


class DummyPoint2D:
    xy = np.array([1.5, 2.5])
    point3D_id = 7


class DummyTrackElement:
    image_id = 3
    point2D_idx = 11


def test_pycolmap_object_helpers_accept_objects_and_dicts() -> None:
    assert _point2d_xy(DummyPoint2D()) == [1.5, 2.5]
    assert _point2d_xy({"xy": [4.0, 5.0]}) == [4.0, 5.0]
    assert _point2d_point3d_id(DummyPoint2D()) == 7
    assert _point2d_point3d_id({"point3D_id": 13}) == 13

    assert _track_element_image_id(DummyTrackElement()) == 3
    assert _track_element_point2d_idx(DummyTrackElement()) == 11
    assert _track_element_image_id({"image_id": 17}) == 17
    assert _track_element_point2d_idx({"point2D_idx": 19}) == 19
