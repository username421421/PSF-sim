import numpy as np
import pytest

from PSF_simulator import propagate_complex_field, propagate_fresnel_field


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


def test_bandlimited_backpropagation_round_trip_for_propagating_spectrum():
    rng = np.random.default_rng(4)
    spectrum = np.zeros((32, 32), dtype=np.complex128)
    spectrum[:3, :3] = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    field = np.fft.ifft2(spectrum)
    forward = propagate_complex_field(
        field, 1e-3, 1e-3, 0.01, 780e-9, device="cpu", bandlimit_evanescent=True
    )
    backward = propagate_complex_field(
        forward, 1e-3, 1e-3, -0.01, 780e-9, device="cpu", bandlimit_evanescent=True
    )
    assert np.linalg.norm(backward - field) / np.linalg.norm(field) <= 1e-10


def test_uniform_input_intensity_scales_raw_psf_not_normalized_psf():
    from PSF_simulator import CartesianPSFSimulator

    sim = CartesianPSFSimulator(torch_device="cpu")
    kwargs = dict(
        size_x=8e-3,
        size_y=8e-3,
        entrance_pupil_diameter=5.6e-3,
        fresnel_distance=0.0,
        range_microns=1.0,
        num_points=11,
    )
    one = sim.simulate_full_pipeline(np.ones((32, 32)), **kwargs)
    quarter = sim.simulate_full_pipeline(np.full((32, 32), 0.25), **kwargs)
    assert np.allclose(quarter["I_xy"], 0.25 * one["I_xy"], rtol=1e-12, atol=0.0)
    assert np.allclose(quarter["I_xy_norm"], one["I_xy_norm"], rtol=0.0, atol=1e-12)
