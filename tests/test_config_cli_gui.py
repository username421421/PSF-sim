from types import SimpleNamespace

import numpy as np
import pytest

import psf_cli


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


def test_string_boolean_in_config_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"params":{"use_gaussian":"false"}}', encoding="utf-8")
    parser = psf_cli.build_parser()
    args = parser.parse_args(["simulate", "--config", str(path)])
    with pytest.raises(ValueError, match="boolean"):
        psf_cli.resolve_params(args)


def test_config_rotation_and_sidecar_complex_mask_round_trip(tmp_path):
    sidecar = tmp_path / "masks.npz"
    pre = np.linspace(0.1, 0.9, 64, dtype=np.float64).reshape(8, 8)
    post = np.exp(1j * np.linspace(0.0, 0.3, 64).reshape(8, 8))
    np.savez_compressed(
        sidecar,
        pre_fresnel_intensity=pre,
        complex_mask_real=post.real,
        complex_mask_imag=post.imag,
    )
    config = tmp_path / "config.json"
    config.write_text(
        '{"format":"psf_sim_gui_config","version":2,"mask":'
        '{"source":"optimized_complex","path":"masks.npz",'
        '"rotation_quadrants_ccw":1,"normalization":"physical","loaded":true}}',
        encoding="utf-8",
    )
    parser = psf_cli.build_parser()
    args = parser.parse_args(
        ["simulate", "--config", str(config), "--nx", "8", "--ny", "8"]
    )
    params = psf_cli.resolve_params(args)
    actual_pre, actual_post, _path, source = psf_cli.resolve_input_and_post_masks(params, args)
    assert source == "optimized_complex"
    assert np.array_equal(actual_pre, np.rot90(pre))
    assert np.allclose(actual_post, np.rot90(post), rtol=0, atol=1e-15)
