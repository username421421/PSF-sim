import numpy as np
import pytest

from PSF_simulator import propagate_fresnel_field


@pytest.mark.xfail(strict=True, reason="C5: public propagation accepts unstable negative distance")
def test_public_fresnel_rejects_negative_distance():
    with pytest.raises(ValueError, match="nonnegative"):
        propagate_fresnel_field(
            np.ones((8, 8), dtype=np.float64),
            size_x=1e-6,
            size_y=1e-6,
            z=-1.0,
            wavelength=780e-9,
            device="cpu",
        )


def test_zero_distance_preserves_float64_amplitude():
    intensity = np.linspace(0.0, 1.0, 64, dtype=np.float64).reshape(8, 8)
    field = propagate_fresnel_field(intensity, 1e-3, 1e-3, 0.0, 780e-9, device="cpu")
    assert field.dtype == np.complex128
    assert np.allclose(np.abs(field) ** 2, intensity, rtol=0, atol=1e-15)
