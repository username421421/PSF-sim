"""Deterministic PSF timing/memory harness used for before/after comparisons."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
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


def _run_3d(device: str, points: int = 61):
    simulator = CartesianPSFSimulator(torch_device=device)
    intensity = np.ones((256, 256), dtype=np.float64)
    prep = simulator.prepare_pupil_field_from_input(
        intensity, 8e-3, 8e-3, 5.6e-3, 0.0
    )
    return simulator.simulate_psf_volume_from_pupil_field(
        prep["_field_at_pupil_for_psf_torch"],
        prep["size_x_for_psf"],
        prep["size_y_for_psf"],
        5.6e-3,
        range_microns=2.0,
        num_points=points,
        # Match the runtime planner: full-z chunks are substantially faster on
        # CUDA when the preflight estimate says they fit; CPU retains the
        # smaller audited chunk to avoid a larger contraction workspace.
        z_chunk_size=points if device == "cuda" else 8,
    )


def _run_profile_sequence(device: str, count: int = 10):
    simulator = CartesianPSFSimulator(torch_device=device)
    size = 256
    x = np.linspace(-1.0, 1.0, size, endpoint=False)[:, None]
    y = np.linspace(-1.0, 1.0, size, endpoint=False)[None, :]
    radius = np.sqrt(x * x + y * y)
    result = None
    for index in range(count):
        inner = 0.05 + 0.7 * index / max(1, count - 1)
        mask = np.asarray((radius < inner) | (radius > min(0.95, inner + 0.1)), dtype=np.float64)
        result = simulator.simulate_psf_profile_full_pipeline(
            mask,
            8e-3,
            8e-3,
            5.6e-3,
            0.0,
            axis="z",
            range_microns=2.0,
            num_points=101,
        )
    return result


def benchmark_case(name, function, repeats, *, items_per_run=1, parameters=None):
    function()
    values = []
    for _ in range(int(repeats)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    baseline_rss = _rss_bytes() or 0
    run_peak = {"rss": baseline_rss}
    stop_monitor = threading.Event()

    def monitor():
        while not stop_monitor.is_set():
            run_peak["rss"] = max(run_peak["rss"], _rss_bytes() or 0)
            stop_monitor.wait(0.002)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stop_monitor.set()
    watcher.join(timeout=1.0)
    peak_rss = max(run_peak["rss"], _rss_bytes() or 0)
    peak_cuda = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return {
        "name": name,
        "seconds": values,
        "median_seconds": statistics.median(values),
        "items_per_run": int(items_per_run),
        "median_seconds_per_item": statistics.median(values) / int(items_per_run),
        "parameters": dict(parameters or {}),
        "peak_rss_bytes": int(peak_rss),
        "baseline_rss_bytes": int(baseline_rss),
        "peak_rss_delta_bytes": int(max(0, peak_rss - baseline_rss)),
        "cuda_peak_bytes": int(peak_cuda),
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
        if args.suite == "all":
            cases.append(
                benchmark_case(
                    f"3d_61_{device}",
                    lambda d=device: _run_3d(d),
                    args.repeats,
                    parameters={"z_chunk": 61 if device == "cuda" else 8},
                )
            )
            cases.append(
                benchmark_case(
                    f"profile_sequence_10_{device}",
                    lambda d=device: _run_profile_sequence(d),
                    args.repeats,
                    items_per_run=10,
                )
            )

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
