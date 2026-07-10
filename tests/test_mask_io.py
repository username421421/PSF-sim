import numpy as np
import pytest

from psf_cli import read_mask, resize_mask


def test_resize_accepts_rotated_negative_stride_array():
    rotated = np.rot90(np.arange(6, dtype=np.float64).reshape(2, 3))
    assert any(stride < 0 for stride in rotated.strides)
    out = resize_mask(rotated, (5, 7))
    assert out.shape == (5, 7)


def test_numeric_mask_defaults_to_physical_transmission(tmp_path):
    path = tmp_path / "mask.npy"
    expected = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    np.save(path, expected)
    actual = read_mask(path)
    assert np.array_equal(actual, expected)


def test_uint_mask_scales_by_dtype_range(tmp_path):
    path = tmp_path / "mask.npy"
    np.save(path, np.array([[0, 255]], dtype=np.uint8))
    assert np.array_equal(read_mask(path), np.array([[0.0, 1.0]]))


def test_multipage_grayscale_tiff_is_not_treated_as_rgb(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "pages.tif"
    pages = np.stack(
        [
            np.zeros((5, 6), dtype=np.uint8),
            np.full((5, 6), 127, dtype=np.uint8),
            np.full((5, 6), 255, dtype=np.uint8),
        ]
    )
    tifffile.imwrite(path, pages, photometric="minisblack")
    actual = read_mask(path)
    assert actual.shape == (5, 6)
    assert np.array_equal(actual, np.zeros((5, 6)))
