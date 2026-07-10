"""Deterministic PSF timing/memory harness used for before/after comparisons."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PSF_simulator import CartesianPSFSimulator


def _rss_bytes():
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _run_2d(device: str, points: int = 101):
    simulator = CartesianPSFSimulator(torch_device=device)
    intensity = np.ones((256, 256), dtype=np.float64)
    return simulator.simulate_full_pipeline(
        intensity,
        size_x=8e-3,
        size_y=8e-3,
        entrance_pupil_diameter=5.6e-3,
        fresnel_distance=0.0,
        range_microns=2.0,
        num_points=points,
    )


def benchmark_case(name, function, repeats):
    function()
    values = []
    rss_before = _rss_bytes()
    for _ in range(int(repeats)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    rss_after = _rss_bytes()
    return {
        "name": name,
        "seconds": values,
        "median_seconds": statistics.median(values),
        "rss_delta_bytes": None if rss_before is None else max(0, rss_after - rss_before),
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all", choices=("all", "smoke"))
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compare")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    cases = []
    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        cases.append(benchmark_case(f"2d_default_{device}", lambda d=device: _run_2d(d), args.repeats))

    payload = {
        "schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "cases": cases,
    }
    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as handle:
            payload["comparison_baseline"] = json.load(handle)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
