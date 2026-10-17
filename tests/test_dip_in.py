"""Physical limits of the full layer OPD, including homogeneous resin dipping."""

import json
from pathlib import Path

import numpy as np
import pytest

import psf_cli
from PSF_simulator import CartesianPSFSimulator
from psf_runtime import create_simulator


def _dip_sim(**overrides):
    params = dict(
        n_i=1.474, n_g=1.474, n_s=1.474,
        n_id=1.518, n_gd=1.518,
        t_g=0.0, t_s=0.0, t_gd=170e-6, t_id=140e-6,
        aberration_model="gibson_lanni", torch_device="cpu",
    )
    params.update(overrides)
    return CartesianPSFSimulator(**params)


def _total_phase(sim, q):
    theta = np.arcsin(np.asarray(q) / sim.n_i)
    ts, _tp, _theta_s = sim._transmission_terms(theta)
    # Homogeneous experimental resin has unit Fresnel amplitudes. This
    # includes the glass phase so a missing/doubled design glass is caught.
    return ts.detach().cpu().numpy() * np.exp(
        1j * sim.k * sim._w_abb(theta).detach().cpu().numpy()
    )


def test_index_matched_dip_in_has_no_relative_aberration():
    sim = _dip_sim(n_i=1.518, n_g=1.518, n_s=1.518)
    phase = _total_phase(sim, np.linspace(0, 1.4, 31))
    np.testing.assert_allclose(phase / phase[0], 1, rtol=0, atol=2e-12)


def test_homogeneous_dip_in_depends_on_total_design_path_not_split():
    # When design oil and glass have the same RI, exchanging their thicknesses
    # cannot change the homogeneous-resin PSF, even with index mismatch.
    q = np.linspace(0, 1.4, 31)
    p1 = _total_phase(_dip_sim(), q)
    p2 = _total_phase(_dip_sim(t_id=310e-6, t_gd=0), q)
    np.testing.assert_allclose(p1 / p1[0], p2 / p2[0], rtol=0, atol=2e-12)


def test_paraxial_focus_removes_quadratic_defocus():
    sim = _dip_sim()
    phase = _total_phase(sim, [0, 0.01, 0.02])
    residual = np.angle(phase / phase[0])
    # The leading remaining term is q**4: doubling q gives 16, not 4.
    assert residual[1] != 0
    assert residual[2] / residual[1] == pytest.approx(16, rel=0.002)


@pytest.mark.parametrize("mode", ["cartesian_jacobian", "polar_interp"])
def test_substrate_depth_does_not_add_an_interface_in_homogeneous_resin(mode):
    field = np.ones((40, 40), dtype=np.complex128)
    profiles = []
    for depth in (0, 100e-6):
        sim = _dip_sim(t_s=depth, integration_mode=mode)
        profiles.append(sim.simulate_psf_profile_from_pupil_field(
            field, 12e-3, 12e-3, 8.89e-3,
            axis="z", range_microns_z=25, num_points=101,
        )["intensity_norm"])
    np.testing.assert_allclose(profiles[0], profiles[1], rtol=0, atol=2e-12)


def test_design_immersion_gap_changes_mismatched_psf_and_cache():
    field = np.ones((40, 40), dtype=np.complex128)
    profiles = []
    for gap in (50e-6, 140e-6):
        sim = _dip_sim(t_id=gap)
        profiles.append(sim.simulate_psf_profile_from_pupil_field(
            field, 12e-3, 12e-3, 8.89e-3,
            axis="z", range_microns_z=25, num_points=101,
        )["intensity_norm"])
    assert np.max(np.abs(profiles[0] - profiles[1])) > 0.05


def test_impossible_negative_immersion_gap_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        _dip_sim(t_s=1e-3)


def test_example_cli_config_and_unit_overrides(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples" / "leica_63x_resin_dip_in.json"
    parser = psf_cli.build_parser()
    params = psf_cli.resolve_params(parser.parse_args([
        "profile", "--axis", "z", "--config", str(path), "--t-id-um", "150",
    ]))
    assert params["aberration_model"] == "gibson_lanni"
    assert params["t_id"] == pytest.approx(150e-6)
    assert params["n_i"] == params["n_s"] == 1.474
    assert params["t_g"] == 0
    saved = tmp_path / "config.json"
    saved.write_text(json.dumps({"params": params}), encoding="utf-8")
    restored = psf_cli.resolve_params(parser.parse_args([
        "profile", "--axis", "z", "--config", str(saved),
    ]))
    assert restored == params
    sim = create_simulator(restored)
    assert sim.t_id == params["t_id"]
    assert sim.aberration_model == params["aberration_model"]


def test_old_configs_retain_legacy_phase_model():
    params = psf_cli.validate_params({
        key: value for key, value in psf_cli.DEFAULT_PARAMS.items()
        if key not in {"aberration_model", "t_id"}
    })
    assert params["aberration_model"] == "legacy"
    assert params["t_id"] == 90e-6
    assert create_simulator(params).aberration_model == "legacy"


def test_profile_summary_reports_width_of_selected_two_photon_signal():
    coords = np.linspace(-3e-6, 3e-6, 601)
    signal = np.exp(-(coords / 1e-6) ** 2)
    result = {
        "axis": "z", "coords": coords, "coords_um": coords * 1e6,
        "intensity_norm": signal,
        "width_level": 0.5, "width_um": 2 * np.sqrt(np.log(2)),
        "peak_index": 300, "peak_coord_um": 0, "peak_intensity": 1,
        "z_focus_um": 0, "z_position_um": 0,
    }
    summary = psf_cli.profile_summary(psf_cli.DEFAULT_PARAMS, result, plot_mode="I2")
    assert summary["width_um"] == pytest.approx(np.sqrt(2 * np.log(2)), rel=1e-4)
    assert summary["intensity_width_um"] == result["width_um"]


def _gui_control_app():
    import ast
    from psf_gui import PSFSimulatorApp

    # Exercise actual control serialization without opening a window or
    # starting workers. Control doubles avoid requiring a display/Tcl runtime.
    class ControlValue:
        def __init__(self, value):
            self.value = value
            self.callbacks = []

        def get(self):
            return self.value

        def set(self, value):
            self.value = value
            for callback in self.callbacks:
                callback()

        def trace_add(self, _mode, callback):
            self.callbacks.append(callback)

    app = PSFSimulatorApp.__new__(PSFSimulatorApp)
    source = Path(__file__).resolve().parents[1] / "psf_gui.py"
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not isinstance(func, ast.Attribute) or func.attr not in {"DoubleVar", "IntVar", "StringVar", "BooleanVar"}:
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                value_kw = next((kw for kw in node.value.keywords if kw.arg == "value"), None)
                if value_kw is not None and isinstance(value_kw.value, ast.Constant):
                    setattr(app, target.attr, ControlValue(value_kw.value.value))
    for name in ("_on_integration_mode_change", "_update_sampling_info", "_refresh_mask_input_preview"):
        setattr(app, name, lambda: None)
    app._clear_loaded_mask = lambda **_kwargs: None
    app.exp_resin_ri_var.trace_add("write", app._sync_dip_in_media)
    return app


@pytest.mark.gui
def test_gui_design_gap_and_phase_model_survive_config_and_objective_round_trip(tmp_path):
    app = _gui_control_app()
    source = Path(__file__).resolve().parents[1] / "psf_gui.py"
    path = source.parent / "examples" / "leica_63x_resin_dip_in.json"
    app._apply_params_to_controls({**psf_cli.DEFAULT_PARAMS, **json.loads(path.read_text())["params"]})
    ui = app._collect_control_values()
    assert ui["t_id_um"] == pytest.approx(140)
    assert ui["aberration_model"] == "gibson_lanni"
    app._apply_control_values(ui)
    params = app._collect_params()
    assert params["t_id"] == pytest.approx(140e-6)
    assert params["aberration_model"] == "gibson_lanni"
    assert create_simulator(params).aberration_model == "gibson_lanni"

    preset = tmp_path / "objective.json"
    app._save_objective_lens_to_path(str(preset))
    app.design_immersion_gap_um_var.set(90)
    app._load_objective_lens_from_path(str(preset))
    assert app.design_immersion_gap_um_var.get() == pytest.approx(140)
    assert psf_cli.load_objective_lens(preset)["t_id"] == pytest.approx(140e-6)
    app._apply_control_values({})
    assert app.aberration_model_var.get() == "legacy"
    assert app.design_immersion_gap_um_var.get() == 90


@pytest.mark.gui
def test_dip_in_toggle_links_resin_restores_previous_settings_and_keeps_objective():
    app = _gui_control_app()
    app._apply_params_to_controls({
        **psf_cli.DEFAULT_PARAMS, "n_i": 1.518, "n_g": 1.52,
        "t_g": 175e-6, "t_s": 12e-6, "n_id": 1.518, "t_id": 140e-6,
    })
    objective = app._collect_objective_lens_values()
    previous = {key: var.get() for key, var in app._dip_in_value_vars().items()}
    app.dip_in_enabled_var.set(True)
    app._on_dip_in_toggle()
    assert app.aberration_model_var.get() == "gibson_lanni"
    app.exp_resin_ri_var.set(1.48)
    params = app._collect_params()
    assert params["dip_in_enabled"] is True
    assert params["n_i"] == params["n_g"] == params["n_s"] == 1.48
    assert params["t_g"] == params["t_s"] == 0
    assert create_simulator(params).aberration_model == "gibson_lanni"
    assert app._collect_objective_lens_values() == objective

    app.dip_in_enabled_var.set(False)
    app._on_dip_in_toggle()
    assert {key: var.get() for key, var in app._dip_in_value_vars().items()} == previous
    assert app.exp_resin_ri_var.get() == 1.48


@pytest.mark.gui
@pytest.mark.parametrize("dipping", [False, True])
@pytest.mark.parametrize("core_units", [False, True])
def test_gui_toggle_overrides_conflicting_model_in_loaded_and_submitted_settings(dipping, core_units):
    app = _gui_control_app()
    selected = "gibson_lanni" if dipping else "legacy"
    conflicting = "legacy" if dipping else "gibson_lanni"
    values = {"dip_in_enabled": dipping, "aberration_model": conflicting}
    if core_units:
        app._apply_params_to_controls({**psf_cli.DEFAULT_PARAMS, **values})
    else:
        app._apply_control_values(values)
    assert app.aberration_model_var.get() == selected

    # Serialization and worker submission must honor the toggle even if an
    # older code path assigns a conflicting model after loading.
    app.aberration_model_var.set(conflicting)
    assert app._collect_control_values()["aberration_model"] == selected
    app.aberration_model_var.set(conflicting)
    params = app._collect_params()
    assert params["aberration_model"] == selected
    assert create_simulator(params).aberration_model == selected


@pytest.mark.gui
@pytest.mark.parametrize("has_model_setting", [False, True])
def test_old_homogeneous_standard_config_keeps_original_model(has_model_setting):
    app = _gui_control_app()
    params = {
        **psf_cli.DEFAULT_PARAMS, "n_i": 1.474, "n_g": 1.474,
        "n_s": 1.474, "t_g": 0, "t_s": 0,
    }
    if not has_model_setting:
        params.pop("aberration_model")
    app._apply_params_to_controls(params)
    assert app.dip_in_enabled_var.get() is False
    assert create_simulator(app._collect_params()).aberration_model == "legacy"


@pytest.mark.gui
def test_dip_in_toggle_and_restore_snapshot_survive_gui_config_loading(tmp_path):
    app = _gui_control_app()
    app._apply_params_to_controls({**psf_cli.DEFAULT_PARAMS, "t_s": 12e-6})
    app.dip_in_enabled_var.set(True)
    app._on_dip_in_toggle()
    path = tmp_path / "dip_in.json"
    path.write_text(json.dumps({
        "format": "psf_sim_gui_config", "version": 2,
        "params": app._encode_precise_params(app._collect_params()),
        "ui_values": app._encode_precise_params(app._collect_control_values()),
    }), encoding="utf-8")
    restored = _gui_control_app()
    restored._load_simulation_config_from_path(str(path))
    assert restored.dip_in_enabled_var.get() is True
    # Older saved snapshots could contain a manually selected mismatch model.
    restored._dip_in_previous_values["aberration_model"] = "gibson_lanni"
    restored.exp_resin_ri_var.set(1.49)
    assert restored.exp_immersion_ri_var.get() == 1.49
    restored.dip_in_enabled_var.set(False)
    restored._on_dip_in_toggle()
    assert restored.exp_immersion_ri_var.get() == 1.515
    assert restored.exp_coverslip_thickness_um_var.get() == pytest.approx(170)
    assert restored.exp_resin_thickness_um_var.get() == pytest.approx(12)
    assert restored.aberration_model_var.get() == "legacy"

    # Loading an older layered config while dipping must not have its media
    # overwritten by the resin-RI trace or retain the previous mode.
    app._apply_params_to_controls({**psf_cli.DEFAULT_PARAMS, "n_s": 1.49})
    assert app.dip_in_enabled_var.get() is False
    assert app.exp_immersion_ri_var.get() == 1.515
    assert app.exp_resin_ri_var.get() == 1.49


@pytest.mark.gui
def test_dip_in_controls_are_linked_and_toggle_is_disabled_during_work():
    app = _gui_control_app()

    class Widget:
        def configure(self, *, state):
            self.state = state

    app.dip_in_check = Widget()
    app._dip_in_linked_entries = [Widget() for _ in range(4)]
    app.dip_in_enabled_var.set(True)
    app._on_dip_in_toggle()
    assert all(entry.state == "disabled" for entry in app._dip_in_linked_entries)
    app.is_running = True
    app._update_dip_in_controls()
    assert app.dip_in_check.state == "disabled"
    app.is_running = False
    app.opt_is_running = True
    app._update_dip_in_controls()
    assert app.dip_in_check.state == "disabled"
    app.opt_is_running = False
    app.dip_in_enabled_var.set(False)
    app._on_dip_in_toggle()
    assert app.dip_in_check.state == "normal"
    assert all(entry.state == "normal" for entry in app._dip_in_linked_entries)
