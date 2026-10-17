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


@pytest.mark.parametrize("aberration_model", ["legacy", "gibson_lanni"])
def test_corrected_axial_matches_full_polar_reference(aberration_model):
    kwargs = dict(
        wavelength=780e-9,
        NA=1.4,
        n_i=1.515,
        n_g=1.515,
        n_s=1.474,
        theta_samples=41,
        phi_samples=80,
        torch_device="cpu",
        integration_mode="polar_interp",
        aberration_model=aberration_model,
    )
    field = np.ones((64, 64), dtype=np.complex128)
    full = CartesianPSFSimulator(**kwargs).simulate_psf_planes_from_pupil_field(
        field, 8e-3, 8e-3, 5.6e-3, range_microns=1.0, num_points=21
    )
    axial = AxialSymmetricPSFSimulator(**kwargs)
    pre = axial.prepare_axial_symmetric_precomputed(
        field, 8e-3, 8e-3, 5.6e-3, range_microns=1.0, num_points=21
    )
    fast = axial.simulate_psf_planes_from_radial_profiles(
        np.ones(41), precomputed=pre
    )
    for key in ("I_xy_norm", "I_xz_norm", "I_yz_norm"):
        assert np.max(np.abs(full[key] - fast[key])) <= 5e-4
    for key in ("fwhm_x_um", "fwhm_y_um", "fwhm_z_um"):
        assert abs(full[key] - fast[key]) / full[key] < 2e-3


@pytest.mark.cuda
@pytest.mark.parametrize("integration_mode", ["cartesian_jacobian", "polar_interp"])
def test_cpu_cuda_plane_parity(integration_mode):
    kwargs = dict(
        integration_mode=integration_mode,
        theta_samples=31,
        phi_samples=60,
    )
    field = np.ones((32, 32), dtype=np.complex128)
    outputs = []
    for device in ("cpu", "cuda"):
        outputs.append(
            CartesianPSFSimulator(torch_device=device, **kwargs).simulate_psf_planes_from_pupil_field(
                field, 8e-3, 8e-3, 5.6e-3, range_microns=1.0, num_points=11
            )
        )
    for key in ("I_xy_norm", "I_xz_norm", "I_yz_norm"):
        assert np.max(np.abs(outputs[0][key] - outputs[1][key])) <= 1e-10


def test_design_angle_apodization_matches_definition():
    sim = CartesianPSFSimulator(n_i=1.515, n_id=1.50, torch_device="cpu")
    theta = np.array([0.0, 0.2, 0.5], dtype=np.float64)
    theta_design = np.arcsin(sim.n_i * np.sin(theta) / sim.n_id)
    expected = np.sqrt(np.cos(theta_design))
    actual = sim._design_apodization(theta).detach().cpu().numpy()
    assert np.allclose(actual, expected, rtol=0, atol=1e-14)


def test_public_outputs_are_finite_nonnegative_and_normalized():
    result = CartesianPSFSimulator(torch_device="cpu").simulate_full_pipeline(
        np.ones((32, 32)),
        8e-3,
        8e-3,
        5.6e-3,
        0.0,
        range_microns=1.0,
        num_points=11,
    )
    for raw_key, norm_key in (
        ("I_xy", "I_xy_norm"),
        ("I_xz", "I_xz_norm"),
        ("I_yz", "I_yz_norm"),
    ):
        raw = result[raw_key]
        normalized = result[norm_key]
        assert np.all(np.isfinite(raw)) and np.min(raw) >= 0.0
        assert np.all(np.isfinite(normalized)) and np.min(normalized) >= 0.0
        assert abs(float(np.max(normalized)) - 1.0) <= 1e-12


def test_deprecated_n_sd_is_inert():
    field = np.ones((24, 24), dtype=np.complex128)
    results = []
    for n_sd in (1.1, 1.8):
        results.append(
            CartesianPSFSimulator(n_sd=n_sd, torch_device="cpu").simulate_psf_planes_from_pupil_field(
                field, 8e-3, 8e-3, 5.6e-3, range_microns=1.0, num_points=9
            )
        )
    assert np.array_equal(results[0]["I_xy"], results[1]["I_xy"])
