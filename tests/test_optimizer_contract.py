import json

import numpy as np
import pytest

from psf_mask_optimizer import OptimizerResult, save_result_npz
from psf_mask_optimizer import OptimizerConfig, PostFresnelMaskOptimizer
from psf_cli import DEFAULT_PARAMS, create_simulator_from_params


def _result():
    mask = np.ones((4, 4), dtype=np.float64)
    return OptimizerResult(
        best_objective=1.0,
        baseline_objective=1.0,
        best_metrics_um={},
        baseline_metrics_um={},
        amplitude_mask=mask,
        phase_mask_rad=np.zeros_like(mask),
        complex_mask=mask.astype(np.complex128),
        history=[],
        settings={},
        baseline_result_2d={},
        optimized_result_2d={},
    )


@pytest.mark.xfail(strict=True, reason="C6: optimizer metadata is stored as object dtype")
def test_optimizer_npz_is_pickle_free(tmp_path):
    path = tmp_path / "mask.npz"
    save_result_npz(path, _result())
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        assert metadata["format"] == "psf_optimizer_result"
        assert metadata["version"] == 2


def test_post_fresnel_complex_mask_is_applied_as_amplitude():
    from PSF_simulator import CartesianPSFSimulator

    sim = CartesianPSFSimulator(torch_device="cpu")
    source = np.ones((16, 16), dtype=np.float64)
    phase = np.linspace(-np.pi, np.pi, 16, dtype=np.float64)[:, None]
    mask = 0.5 * np.exp(1j * phase) * np.ones((1, 16), dtype=np.complex128)
    base = sim.prepare_pupil_field_from_input(source, 8e-3, 8e-3, 5.6e-3, 0.01)
    applied = sim.prepare_pupil_field_from_input(
        source,
        8e-3,
        8e-3,
        5.6e-3,
        0.01,
        post_fresnel_mask=mask,
    )
    assert np.allclose(
        applied["field_after_fresnel"], base["field_after_fresnel"] * mask, rtol=0, atol=1e-14
    )


def test_optimizer_returns_exact_candidate_not_worse_than_baseline():
    params = dict(DEFAULT_PARAMS)
    params.update(
        nx=12,
        ny=12,
        points_2d=9,
        fresnel_z=0.0,
        theta_samples=21,
        phi_samples=40,
        range_x_um=1.0,
        range_y_um=1.0,
        range_z_um=1.0,
    )
    simulator = create_simulator_from_params(params)
    result = PostFresnelMaskOptimizer(simulator, params).optimize(
        np.ones((12, 12), dtype=np.float64),
        OptimizerConfig(
            mode="amplitude",
            axial_symmetry=True,
            n_starts=1,
            maxiter=1,
            weights_xyz=(1.0, 0.0, 0.0),
            sidelobe_penalty_weight=0.0,
        ),
    )
    assert result.best_objective <= result.baseline_objective + 1e-10
    assert result.status in {"converged", "budget_exhausted_with_candidate"}
    assert result.solver_runs
    rerun = simulator.simulate_full_pipeline(
        np.ones((12, 12), dtype=np.float64),
        params["size_x"],
        params["size_y"],
        params["pupil_d"],
        params["fresnel_z"],
        range_microns=1.0,
        num_points=9,
        post_fresnel_mask=result.complex_mask,
    )
    assert np.max(
        np.abs(rerun["I_xz_norm"] - result.optimized_result_2d["I_xz_norm"])
    ) <= 1e-10
