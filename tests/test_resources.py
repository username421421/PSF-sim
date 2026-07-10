import importlib

import pytest


@pytest.mark.xfail(strict=True, reason="C13: resource estimation is not implemented")
def test_resource_estimator_has_ram_and_vram_bounds():
    runtime = importlib.import_module("psf_runtime")
    estimate = runtime.estimate_resources(
        {"nx": 64, "ny": 64, "points_2d": 21, "points_3d": 21, "z_chunk": 4},
        run_3d=True,
    )
    assert estimate.peak_ram_bytes > 0
    assert estimate.peak_vram_bytes >= 0
