import numpy as np
import pytest

from psf_cli import read_mask, resize_mask


@pytest.mark.xfail(strict=True, reason="C7: rotated arrays can retain negative strides")
def test_resize_accepts_rotated_negative_stride_array():
    rotated = np.rot90(np.arange(6, dtype=np.float64).reshape(2, 3))
    assert any(stride < 0 for stride in rotated.strides)
    out = resize_mask(rotated, (5, 7))
    assert out.shape == (5, 7)


@pytest.mark.xfail(strict=True, reason="C8: numeric masks are min-max normalized")
def test_numeric_mask_defaults_to_physical_transmission(tmp_path):
    path = tmp_path / "mask.npy"
    expected = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    np.save(path, expected)
    actual = read_mask(path)
    assert np.array_equal(actual, expected)
