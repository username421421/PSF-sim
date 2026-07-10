import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_collection_modifyitems(config, items):
    del config
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    if cuda_available:
        return
    skip = pytest.mark.skip(reason="CUDA is not available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)
