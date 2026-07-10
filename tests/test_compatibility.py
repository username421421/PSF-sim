import pytest

import psf_cli


@pytest.mark.xfail(strict=True, reason="C14: equal-radius sweep pairs create duplicate no-op masks")
def test_ring_sweep_has_one_baseline_and_only_nonzero_width_pairs():
    pairs = psf_cli.ring_radius_pairs([0.0, 1.0, 2.0])
    assert pairs == [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 2.0)]


def test_csv_schema_version_remains_one():
    assert psf_cli.CONFIG_VERSION == 1
