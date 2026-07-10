import pytest
import json
import subprocess
import sys

import psf_cli


def test_ring_sweep_has_one_baseline_and_only_nonzero_width_pairs():
    pairs = psf_cli.ring_radius_pairs([0.0, 1.0, 2.0])
    assert pairs == [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 2.0)]


def test_config_schema_is_v2():
    assert psf_cli.CONFIG_VERSION == 2


def test_core_import_has_no_gui_plotting_or_scipy_optimizer_side_effects():
    code = (
        "import json,sys; import PSF_simulator; "
        "print(json.dumps({k:(k in sys.modules) for k in "
        "['tkinter','matplotlib','scipy.optimize']}))"
    )
    output = subprocess.check_output([sys.executable, "-B", "-c", code], text=True)
    loaded = json.loads(output)
    assert loaded == {"tkinter": False, "matplotlib": False, "scipy.optimize": False}
