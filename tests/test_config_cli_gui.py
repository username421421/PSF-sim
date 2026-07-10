from types import SimpleNamespace

import pytest

import psf_cli


@pytest.mark.xfail(strict=True, reason="C9: fractional integers are silently truncated")
def test_fractional_integer_is_rejected():
    with pytest.raises(ValueError, match="integer"):
        psf_cli.finite_int(9.5, "points")


def test_cli_resolves_explicit_integration_mode():
    parser = psf_cli.build_parser()
    args = parser.parse_args(["simulate", "--integration-mode", "polar_interp"])
    params = psf_cli.resolve_params(args)
    assert params["integration_mode"] == "polar_interp"


def test_existing_cli_flags_remain_accepted():
    parser = psf_cli.build_parser()
    args = parser.parse_args(["simulate", "--theta-samples", "61", "--phi-samples", "120"])
    assert args.theta_samples == 61
    assert args.phi_samples == 120
