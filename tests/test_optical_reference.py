import numpy as np
import pytest

from PSF_simulator import AxialSymmetricPSFSimulator, CartesianPSFSimulator


def _axial_precomputed():
    sim = AxialSymmetricPSFSimulator(
        wavelength=780e-9,
        NA=1.4,
        n_i=1.515,
        n_g=1.515,
        n_s=1.474,
        theta_samples=31,
        phi_samples=60,
        torch_device="cpu",
    )
    field = np.ones((48, 48), dtype=np.complex128)
    return sim, sim.prepare_axial_symmetric_precomputed(
        field,
        size_x=8e-3,
        size_y=8e-3,
        entrance_pupil_diameter=5.6e-3,
        range_microns=1.0,
        num_points=21,
    )


@pytest.mark.xfail(strict=True, reason="C1: effective NA currently stretches the pupil")
def test_effective_na_truncates_nominal_pupil_without_remapping():
    sim = CartesianPSFSimulator(
        NA=1.4, n_i=1.515, n_g=1.515, n_s=1.33, torch_device="cpu"
    )
    field = np.ones((64, 64), dtype=np.complex128)
    tables = sim._prepare_integration_tables_cartesian_jacobian(
        field, 8e-3, 8e-3, 5.6e-3
    )
    rows, cols = np.unravel_index(
        tables["flat_indices"].detach().cpu().numpy(), field.shape
    )
    x = -4e-3 + rows * (8e-3 / 64)
    y = -4e-3 + cols * (8e-3 / 64)
    selected_fraction = np.max(np.hypot(x, y)) / 2.8e-3
    assert selected_fraction <= sim.NA_eff / sim.NA + 2 / 64


@pytest.mark.xfail(strict=True, reason="C2: axial x/y operators are circularized")
def test_axial_x_linear_y_operator_has_no_longitudinal_component():
    _sim, pre = _axial_precomputed()
    assert np.max(np.abs(pre["mz_y"].detach().cpu().numpy())) <= 1e-14
    assert not np.allclose(
        pre["mx_x"].detach().cpu().numpy(),
        pre["mx_y"].detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-14,
    )


def test_cartesian_is_the_library_default():
    assert CartesianPSFSimulator().integration_mode == "cartesian_jacobian"
