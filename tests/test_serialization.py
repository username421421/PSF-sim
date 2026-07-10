import pickle

import numpy as np
import pytest

import psf_cli


def test_legacy_pickle_is_rejected_without_explicit_opt_in(tmp_path):
    path = tmp_path / "legacy.psfres"
    with path.open("wb") as handle:
        pickle.dump({"format": psf_cli.RESULT_FORMAT, "version": 1}, handle)
    with pytest.raises(ValueError, match="unsafe"):
        psf_cli.load_result(path)


def test_result_version_is_safe_v2():
    assert psf_cli.RESULT_VERSION == 2


def test_safe_v2_round_trip_uses_no_object_arrays(tmp_path):
    path = tmp_path / "run.psfres"
    params = dict(psf_cli.DEFAULT_PARAMS)
    result = {
        "x_coords": np.linspace(-1.0, 1.0, 3),
        "I_xy": np.arange(9, dtype=np.float64).reshape(3, 3),
        "complex": np.array([1.0 + 2.0j], dtype=np.complex128),
    }
    psf_cli.save_result(path, params, result)
    with np.load(path, allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    loaded = psf_cli.load_result(path)
    assert np.array_equal(loaded["results_2d"]["I_xy"], result["I_xy"])
    assert loaded["results_2d"]["complex"].dtype == np.complex128


def test_legacy_pickle_load_requires_and_honors_explicit_opt_in(tmp_path):
    path = tmp_path / "legacy.psfres"
    payload = {
        "format": psf_cli.RESULT_FORMAT,
        "version": 1,
        "params": dict(psf_cli.DEFAULT_PARAMS),
        "results_2d": {},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match="unsafe"):
        psf_cli.load_result(path)
    assert psf_cli.load_result(path, allow_unsafe_legacy_pickle=True)["version"] == 1


def test_default_loader_never_executes_pickle_payload(tmp_path):
    marker = tmp_path / "executed.txt"

    class Payload:
        def __reduce__(self):
            return (marker.write_text, ("executed",))

    path = tmp_path / "malicious.psfres"
    with path.open("wb") as handle:
        pickle.dump(Payload(), handle)
    with pytest.raises(ValueError, match="unsafe"):
        psf_cli.load_result(path)
    assert not marker.exists()
