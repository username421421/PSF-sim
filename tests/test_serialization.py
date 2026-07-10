import pickle

import pytest

import psf_cli


@pytest.mark.xfail(strict=True, reason="C11: result loader executes pickle by default")
def test_legacy_pickle_is_rejected_without_explicit_opt_in(tmp_path):
    path = tmp_path / "legacy.psfres"
    with path.open("wb") as handle:
        pickle.dump({"format": psf_cli.RESULT_FORMAT, "version": 1}, handle)
    with pytest.raises(ValueError, match="unsafe"):
        psf_cli.load_result(path)


@pytest.mark.xfail(strict=True, reason="C12: result format has no safe v2 writer")
def test_result_version_is_safe_v2():
    assert psf_cli.RESULT_VERSION == 2
