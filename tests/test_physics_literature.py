"""Checks against Nasse & Woehl (2010) and Torok & Varga (1997) in Literature/."""

from math import pi

import numpy as np
import pytest
import torch

import psf_cli
import psf_mask_optimizer as pmo
from PSF_simulator import CartesianPSFSimulator
from psf_runtime import pupil_sampling_report


def _profile(sim, *, axis, n=128, size=8e-3, diameter=5e-3, extent=1.5, points=121):
    field = np.ones((n, n), dtype=np.complex128)
    key = "range_microns_z" if axis == "z" else "range_microns_x"
    return sim.simulate_psf_profile_from_pupil_field(
        field, size, size, diameter, axis=axis, num_points=points, **{key: extent}
    )


def _matched_low_index(mode, **overrides):
    # Aberration-free focus in a medium below the NA: only the aperture changes.
    params = dict(
        NA=1.4, n_i=1.38, n_g=1.38, n_s=1.38, n_id=1.518, n_gd=1.518,
        t_g=0.0, t_s=0.0, t_gd=0.0, integration_mode=mode,
        theta_samples=801, torch_device="cpu",
    )
    params.update(overrides)
    return CartesianPSFSimulator(**params)


def test_na_above_immersion_ri_truncates_at_total_internal_reflection():
    # Nasse & Woehl Sec. 2: NA_eff = min(NA, n1, n2, n3); no error for NA > n1.
    sim = _matched_low_index("polar_interp")
    assert sim.NA_eff == pytest.approx(1.38)
    assert sim.theta_max == pytest.approx(pi / 2)
    widths = {}
    for mode in ("polar_interp", "cartesian_jacobian"):
        result = _profile(_matched_low_index(mode), axis="x")
        assert np.all(np.isfinite(result["intensity_norm"]))
        widths[mode] = result["width_um"]
    assert widths["cartesian_jacobian"] == pytest.approx(widths["polar_interp"], rel=0.02)


@pytest.mark.parametrize("na", [1.474, 1.5])
def test_cartesian_rim_weight_is_finite_when_a_pixel_sits_on_the_rim(na):
    # 5 mm pupil on a 128 px / 8 mm grid: the radius is exactly 40 pixels, so
    # pixel centres lie on the theta = 90 deg rim where 1/cos(theta) diverges.
    kwargs = dict(NA=na, n_i=1.474, n_g=1.474, n_s=1.474, n_id=1.518, n_gd=1.518, t_s=0.0)
    cart = _matched_low_index("cartesian_jacobian", **kwargs)
    tables = cart._prepare_integration_tables_cartesian_jacobian(
        np.ones((128, 128), dtype=np.complex128), 8e-3, 8e-3, 5e-3
    )
    weights = torch.abs(tables["field_components_static"][0]).cpu().numpy()
    assert np.max(weights) / np.median(weights) < 50
    assert np.all(np.isfinite(_profile(cart, axis="z", extent=3.0)["intensity_norm"]))
    polar = _matched_low_index("polar_interp", **kwargs)
    assert _profile(cart, axis="x")["width_um"] == pytest.approx(
        _profile(polar, axis="x")["width_um"], rel=0.02
    )


def test_cartesian_full_aperture_converges_to_polar_and_is_flagged_when_coarse():
    # Near theta = 90 deg cos(theta) varies like a square root across the last
    # pupil pixels, so a coarse Cartesian grid converges slowly in z.
    kwargs = dict(NA=1.474, n_i=1.474, n_g=1.474, n_s=1.474, n_id=1.518, n_gd=1.518, t_s=0.0)
    polar = _profile(_matched_low_index("polar_interp", **kwargs), axis="z", extent=3.0)["width_um"]
    coarse = _profile(_matched_low_index("cartesian_jacobian", **kwargs), axis="z", extent=3.0)
    fine = _profile(_matched_low_index("cartesian_jacobian", **kwargs), axis="z", extent=3.0, n=512)
    assert abs(fine["width_um"] - polar) < 0.6 * abs(coarse["width_um"] - polar)
    report = _matched_low_index("cartesian_jacobian", **kwargs).pupil_phase_sampling(
        entrance_pupil_diameter=5e-3, shape=(128, 128), size_x=8e-3, size_y=8e-3, range_z_m=3e-6,
    )
    assert report["undersampled"]


def test_na_must_be_below_design_immersion_ri():
    with pytest.raises(ValueError, match="design immersion"):
        CartesianPSFSimulator(NA=1.4, n_i=1.515, n_id=1.33, torch_device="cpu")


def _dry_objective(**overrides):
    # Nasse & Woehl Fig. 2: 40x NA 0.95 dry objective, 170 um coverslip, water.
    params = dict(
        wavelength=488e-9, NA=0.95, n_i=1.0, n_id=1.0, n_g=1.515, n_gd=1.515,
        n_s=1.33, t_g=170e-6, t_gd=170e-6, t_s=10e-6, torch_device="cpu",
    )
    params.update(overrides)
    return CartesianPSFSimulator(**params)


def test_coverslip_transmission_matches_torok_varga_eq_32():
    sim = _dry_objective()
    theta = torch.linspace(0, sim.theta_max, 41, dtype=torch.float64)
    t_s, t_p, _theta_3 = sim._transmission_terms(theta)
    cos_1 = torch.cos(theta)
    cos_2 = torch.cos(sim._theta_2(theta))
    cos_3 = torch.cos(sim._theta_3(theta))
    t12_s, t12_p, r12_s, r12_p = sim._fresnel_coefficients(sim.n_i, cos_1, sim.n_g, cos_2)
    t23_s, t23_p, r23_s, r23_p = sim._fresnel_coefficients(sim.n_g, cos_2, sim.n_s, cos_3)
    beta, beta_design = sim._glass_phases(theta)
    expected_s = t12_s * t23_s * torch.exp(1j * (beta - beta_design)) / (
        1 + r12_s * r23_s * torch.exp(2j * beta)
    )
    expected_p = t12_p * t23_p * torch.exp(1j * (beta - beta_design)) / (
        1 + r12_p * r23_p * torch.exp(2j * beta)
    )
    np.testing.assert_allclose(t_s.numpy(), expected_s.numpy(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(t_p.numpy(), expected_p.numpy(), rtol=1e-12, atol=1e-14)


def test_design_coverslip_only_shifts_the_transmitted_phase():
    # The design coverslip is a compensating phase, not a layer light reflects
    # in, so |T| cannot depend on its thickness.
    theta = torch.linspace(0, 1.2, 41, dtype=torch.float64)
    reference = _dry_objective()._transmission_terms(theta)
    other = _dry_objective(t_gd=120e-6)._transmission_terms(theta)
    for ref, alt in zip(reference[:2], other[:2]):
        np.testing.assert_allclose(torch.abs(alt).numpy(), torch.abs(ref).numpy(), rtol=1e-12)


def test_zero_thickness_coverslip_reduces_to_a_single_interface():
    sim = _dry_objective(t_g=0.0)
    theta = torch.linspace(0, sim.theta_max, 41, dtype=torch.float64)
    t_s, t_p, _theta_3 = sim._transmission_terms(theta)
    direct = sim._fresnel_coefficients(
        sim.n_i, torch.cos(theta), sim.n_s, torch.cos(sim._theta_3(theta))
    )
    np.testing.assert_allclose(torch.abs(t_s).numpy(), torch.abs(direct[0]).numpy(), rtol=1e-12)
    np.testing.assert_allclose(torch.abs(t_p).numpy(), torch.abs(direct[1]).numpy(), rtol=1e-12)


def _dip_in(**overrides):
    params = dict(
        NA=1.4, n_i=1.474, n_g=1.474, n_s=1.474, n_id=1.518, n_gd=1.518,
        t_g=0.0, t_s=0.0, t_gd=170e-6, t_id=140e-6,
        aberration_model="gibson_lanni", torch_device="cpu",
    )
    params.update(overrides)
    return CartesianPSFSimulator(**params)


def test_constructor_kwargs_rebuild_the_same_model():
    sim = _dip_in(integration_mode="polar_interp", theta_samples=401)
    assert CartesianPSFSimulator(**sim.constructor_kwargs()).constructor_kwargs() == sim.constructor_kwargs()


def test_radial_optimizer_fast_path_uses_the_simulator_aberration_model():
    sim = _dip_in()
    fast = pmo._AxialSymmetricRadialFastPSF(
        sim,
        np.ones((48, 48), dtype=np.complex128),
        size_x=12e-3, size_y=12e-3, entrance_pupil_diameter=8.89e-3,
        range_microns=1.0, range_microns_x=1.0, range_microns_y=1.0, range_microns_z=5.0,
        num_points=21, compute_xy=True, compute_xz=True, compute_yz=False,
    )
    assert fast.axial_simulator.aberration_model == "gibson_lanni"
    theta = torch.linspace(0, sim.theta_max, 31, dtype=torch.float64)
    np.testing.assert_allclose(
        fast.axial_simulator._w_abb(theta).numpy(), sim._w_abb(theta).numpy(), rtol=0, atol=0
    )


def _dip_in_params(**overrides):
    params = psf_cli.validate_params({
        **psf_cli.DEFAULT_PARAMS,
        "NA": 1.4, "n_i": 1.474, "n_g": 1.474, "n_s": 1.474, "n_id": 1.518, "n_gd": 1.518,
        "t_g": 0.0, "t_s": 0.0, "t_gd": 170e-6, "t_id": 140e-6, "aberration_model": "gibson_lanni",
        "nx": 512, "ny": 512, "size_x": 12e-3, "size_y": 12e-3, "pupil_d": 8.89e-3,
        "range_x_um": 2.0, "range_y_um": 2.0, "range_z_um": 25.0,
    })
    params.update(overrides)
    return params


def test_sampling_check_flags_the_default_theta_samples_for_dip_in():
    # About 21 waves of immersion-mismatch aberration: 61 theta samples alias it
    # (axial FWHM 0.72 um instead of the converged 3.04 um).
    report = pupil_sampling_report(_dip_in_params(integration_mode="polar_interp", theta_samples=61))
    assert report["aberration_waves"] == pytest.approx(20.8, abs=0.5)
    assert report["undersampled"]
    assert report["suggested"]["theta_samples"] >= 401
    fine = pupil_sampling_report(_dip_in_params(integration_mode="polar_interp", theta_samples=801))
    assert not fine["undersampled"]


@pytest.mark.parametrize("mode", ["cartesian_jacobian", "polar_interp"])
def test_sampling_check_accepts_the_default_configuration(mode):
    params = psf_cli.validate_params({**psf_cli.DEFAULT_PARAMS, "integration_mode": mode})
    assert not pupil_sampling_report(params)["undersampled"]

