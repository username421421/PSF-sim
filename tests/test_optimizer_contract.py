import json

import numpy as np
import pytest

from psf_mask_optimizer import OptimizerResult, save_result_npz


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
