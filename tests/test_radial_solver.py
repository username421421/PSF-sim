import numpy as np
import pytest

from PSF_simulator import AxialSymmetricPSFSimulator


@pytest.mark.xfail(strict=True, reason="C3: radial coordinate is normalized to effective theta")
def test_radial_axis_uses_nominal_pupil_radius():
    sim = AxialSymmetricPSFSimulator(
        NA=1.4,
        n_i=1.515,
        n_s=1.33,
        theta_samples=31,
        phi_samples=60,
        torch_device="cpu",
    )
    pre = sim.prepare_axial_symmetric_precomputed(
        np.ones((48, 48), dtype=np.complex128),
        8e-3,
        8e-3,
        5.6e-3,
        range_microns=1.0,
        num_points=11,
    )
    radial = pre["radial_norm_axis"].detach().cpu().numpy()
    assert radial[-1] == pytest.approx(sim.NA_eff / sim.NA, rel=0, abs=1e-14)


@pytest.mark.xfail(strict=True, reason="C4: radial profiles are resized by index, not radius")
def test_radial_profile_is_interpolated_by_canonical_radius():
    sim = AxialSymmetricPSFSimulator(theta_samples=9, phi_samples=16)
    source_r = np.linspace(0.0, 1.0, 5)
    source = source_r**2
    target_r = np.array([0.0, 0.1, 0.4, 0.9])
    got = sim._interpolate_canonical_radial_profile(source, target_r)
    assert np.allclose(got.detach().cpu().numpy(), target_r**2, atol=2e-2)
