"""PARAM_SPECS is the single parameter table; these tests catch drift from it."""

import pytest

import psf_cli
from psf_models import OBJECTIVE_SPECS, PARAM_SPECS, objective_params, objective_ui_values


def test_cli_defaults_come_from_the_table():
    assert psf_cli.DEFAULT_PARAMS == {spec.key: spec.default for spec in PARAM_SPECS}


def test_every_parameter_has_a_cli_option():
    args = vars(psf_cli.build_parser().parse_args(["simulate"]))
    missing = [spec.dest for spec in PARAM_SPECS if spec.dest not in args]
    assert missing == []


def test_cli_unit_overrides_use_the_table_scales():
    args = psf_cli.build_parser().parse_args([
        "simulate", "--wavelength-nm", "800", "--na", "1.2", "--size-x-mm", "6",
        "--t-id-um", "120", "--aberration-model", "gibson_lanni",
    ])
    params = psf_cli.resolve_params(args)
    assert params["wavelength"] == pytest.approx(800e-9)
    assert params["NA"] == 1.2
    assert params["size_x"] == pytest.approx(6e-3)
    assert params["t_id"] == pytest.approx(120e-6)
    assert params["aberration_model"] == "gibson_lanni"


def test_ui_values_and_core_params_convert_consistently():
    ui = {spec.ui_key: spec.ui_default for spec in PARAM_SPECS}
    params = psf_cli.params_from_ui_values(ui)
    for spec in PARAM_SPECS:
        assert params[spec.key] == pytest.approx(spec.default), spec.key


def test_objective_presets_validate_and_default_optional_keys():
    values = {"NA": 1.4, "pupil_d_mm": 8.89, "n_id": 1.518, "n_gd": 1.518, "t_gd_um": 170}
    ui = objective_ui_values(values)
    assert ui["t_id_um"] == 90.0
    assert ui["n_sd"] == 1.518
    params = objective_params(values)
    assert set(params) == {spec.key for spec in OBJECTIVE_SPECS}
    assert params["pupil_d"] == pytest.approx(8.89e-3)
    with pytest.raises(ValueError, match="missing keys: t_gd_um"):
        objective_ui_values({key: value for key, value in values.items() if key != "t_gd_um"})
    with pytest.raises(ValueError, match="Design immersion gap must be finite and >= 0 um"):
        objective_ui_values({**values, "t_id_um": -1})


def test_gaussian_fwhm_input_is_the_half_maximum_width_of_the_envelope():
    import numpy as np
    from PSF_simulator import apply_gaussian_profile
    from psf_models import PARAM_SPECS_BY_KEY

    fwhm_mm = 3.0
    radius_m = PARAM_SPECS_BY_KEY["gauss_wx"].si_value(fwhm_mm)
    n, size = 2000, 8e-3
    profile = apply_gaussian_profile(np.ones((n, 1)), size, size, radius_m, 1.0)[:, 0]
    profile = profile / profile.max()  # the single y sample sits off-centre
    x = -size / 2 + np.arange(n) * size / n
    assert np.interp(fwhm_mm * 1e-3 / 2, x, profile) == pytest.approx(0.5, abs=1e-9)
    assert np.interp(-fwhm_mm * 1e-3 / 2, x, profile) == pytest.approx(0.5, abs=1e-9)


def test_cli_gaussian_flags_take_fwhm_and_keep_the_old_radius_flag():
    from psf_models import FWHM_PER_1E_RADIUS

    parser = psf_cli.build_parser()
    fwhm = psf_cli.resolve_params(parser.parse_args(["simulate", "--gauss-fwhm-x-mm", "3"]))
    assert fwhm["gauss_wx"] == pytest.approx(3e-3 / FWHM_PER_1E_RADIUS)
    legacy = psf_cli.resolve_params(parser.parse_args(["simulate", "--gauss-wx-mm", "1.6"]))
    assert legacy["gauss_wx"] == pytest.approx(1.6e-3)
    both = psf_cli.resolve_params(parser.parse_args(
        ["simulate", "--gauss-wx-mm", "1.6", "--gauss-fwhm-x-mm", "3"]
    ))
    assert both["gauss_wx"] == pytest.approx(3e-3 / FWHM_PER_1E_RADIUS)


def test_old_gui_configs_keep_the_same_physical_gaussian():
    params = psf_cli.params_from_ui_values({"gauss_wx_mm": 4.5, "gauss_wy_mm": 4.5})
    assert params["gauss_wx"] == pytest.approx(4.5e-3)
    assert params["gauss_wy"] == pytest.approx(4.5e-3)
