import subprocess
import sys
import time

import numpy as np
import pytest

from PSF_simulator import CartesianPSFSimulator


def test_prepared_tensor_path_does_not_host_hash(monkeypatch):
    simulator = CartesianPSFSimulator(torch_device="cpu")
    prep = simulator.prepare_pupil_field_from_input(
        np.ones((32, 32)), 8e-3, 8e-3, 5.6e-3, 0.0
    )

    def fail_hash(_value):
        raise AssertionError("prepared tensor path attempted content hashing")

    monkeypatch.setattr(simulator, "_array_hash", fail_hash)
    simulator.simulate_psf_planes_from_pupil_field(
        prep["_field_at_pupil_for_psf_torch"],
        prep["size_x_for_psf"],
        prep["size_y_for_psf"],
        5.6e-3,
        range_microns=1.0,
        num_points=9,
    )


def test_only_the_exact_prepared_tensor_uses_revision_token(monkeypatch):
    simulator = CartesianPSFSimulator(torch_device="cpu")
    prep = simulator.prepare_pupil_field_from_input(
        np.ones((24, 24)), 8e-3, 8e-3, 5.6e-3, 0.0
    )
    prepared = prep["_field_at_pupil_for_psf_torch"]
    hashed = []
    original_hash = simulator._array_hash

    def record_hash(value):
        hashed.append(value)
        return original_hash(value)

    monkeypatch.setattr(simulator, "_array_hash", record_hash)
    simulator._integration_table_cache_key(prepared, "cartesian_jacobian", 8e-3, 8e-3, 5.6e-3)
    assert not hashed
    cloned = prepared.clone()
    simulator._integration_table_cache_key(cloned, "cartesian_jacobian", 8e-3, 8e-3, 5.6e-3)
    assert len(hashed) == 1


def test_xz_only_does_not_construct_y_phase(monkeypatch):
    simulator = CartesianPSFSimulator(torch_device="cpu")

    def fail_y(*_args, **_kwargs):
        raise AssertionError("unrequested y phase was constructed")

    monkeypatch.setattr(simulator, "_precompute_exp_y", fail_y)
    result = simulator.simulate_psf_planes_from_pupil_field(
        np.ones((32, 32), dtype=np.complex128),
        8e-3,
        8e-3,
        5.6e-3,
        range_microns=1.0,
        num_points=9,
        compute_xy=False,
        compute_xz=True,
        compute_yz=False,
    )
    assert result["I_xz"] is not None


def test_cache_is_byte_bounded_and_clearable():
    simulator = CartesianPSFSimulator(torch_device="cpu")
    for scale in (1.0, 0.9, 0.8):
        simulator.simulate_psf_planes_from_pupil_field(
            np.full((32, 32), scale, dtype=np.complex128),
            8e-3,
            8e-3,
            5.6e-3,
            range_microns=1.0,
            num_points=9,
        )
    info = simulator.cache_info()
    assert info["geometry_bytes"] <= info["limit_bytes"]
    assert info["integration_bytes"] <= info["limit_bytes"]
    simulator.clear_caches()
    assert simulator.cache_info()["integration_entries"] == 0


def test_3d_chunk_invariance_cpu():
    simulator = CartesianPSFSimulator(torch_device="cpu")
    field = np.ones((24, 24), dtype=np.complex128)
    results = [
        simulator.simulate_psf_volume_from_pupil_field(
            field,
            8e-3,
            8e-3,
            5.6e-3,
            range_microns=1.0,
            num_points=9,
            z_chunk_size=chunk,
        )["I_xyz_norm"]
        for chunk in (1, 4)
    ]
    assert np.linalg.norm(results[0] - results[1]) / np.linalg.norm(results[0]) <= 1e-12


@pytest.mark.gui
def test_cli_and_headless_gui_help_start_under_one_second():
    for script in ("psf_cli.py", "psf_gui.py"):
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-B", script, "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        elapsed = time.perf_counter() - started
        assert completed.returncode == 0
        assert elapsed < 1.0
