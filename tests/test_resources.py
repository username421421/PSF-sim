import importlib

import pytest


def test_resource_estimator_has_ram_and_vram_bounds():
    runtime = importlib.import_module("psf_runtime")
    estimate = runtime.estimate_resources(
        {"nx": 64, "ny": 64, "points_2d": 21, "points_3d": 21, "z_chunk": 4},
        run_3d=True,
    )
    assert estimate.peak_ram_bytes > 0
    assert estimate.peak_vram_bytes >= 0


def test_resource_estimator_conservatively_bounds_audited_workloads():
    runtime = importlib.import_module("psf_runtime")
    params = {
        "nx": 256,
        "ny": 256,
        "points_2d": 101,
        "points_3d": 61,
        "z_chunk": 8,
        "theta_samples": 61,
        "phi_samples": 120,
        "torch_device": "cpu",
    }
    estimate_2d = runtime.estimate_resources(params, run_3d=False)
    estimate_3d = runtime.estimate_resources(params, run_3d=True)
    audited_2d = 300 * 1024**2
    audited_3d = 565 * 1024**2
    assert 1.10 * audited_2d <= estimate_2d.peak_ram_bytes <= 1.50 * audited_2d
    assert 1.10 * audited_3d <= estimate_3d.peak_ram_bytes <= 1.50 * audited_3d


def test_resource_estimator_bounds_measured_full_chunk_cuda_peak():
    runtime = importlib.import_module("psf_runtime")
    estimate = runtime.estimate_resources(
        {
            "nx": 256,
            "ny": 256,
            "points_2d": 101,
            "points_3d": 61,
            "z_chunk": 61,
            "theta_samples": 61,
            "phi_samples": 120,
            "torch_device": "cuda",
        },
        run_3d=True,
    )
    measured_peak = 3_174_636_032
    estimate_bytes = estimate.peak_vram_bytes or estimate.peak_ram_bytes
    assert 1.10 * measured_peak <= estimate_bytes <= 1.50 * measured_peak
