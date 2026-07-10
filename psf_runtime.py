"""Shared runtime estimates and lightweight orchestration helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from psf_models import ResourceEstimate


def create_simulator(params: Mapping[str, Any]):
    from PSF_simulator import CartesianPSFSimulator

    kwargs = {
        "wavelength": float(params["wavelength"]),
        "NA": float(params["NA"]),
        "theta_samples": int(params["theta_samples"]),
        "phi_samples": int(params["phi_samples"]),
        "integration_mode": str(params.get("integration_mode", "cartesian_jacobian")),
        "torch_device": str(params.get("torch_device", "cpu")).strip().lower(),
        "n_i": float(params["n_i"]),
        "n_g": float(params["n_g"]),
        "t_g": float(params["t_g"]),
        "n_s": float(params["n_s"]),
        "t_s": float(params["t_s"]),
        "n_id": float(params["n_id"]),
        "n_gd": float(params["n_gd"]),
        "t_gd": float(params["t_gd"]),
        "n_sd": float(params.get("n_sd", params["n_id"])),
    }
    if "t_id" in params:
        kwargs["t_id"] = float(params["t_id"])
    return CartesianPSFSimulator(**kwargs)


def _available_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 8 * 1024**3


def _available_vram_bytes(device: str) -> int:
    if str(device).lower() not in {"cuda", "auto"}:
        return 0
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return 0


def estimate_resources(
    params: Mapping[str, Any],
    *,
    run_3d: bool = False,
    optimizer_variables: int = 0,
    predicted_objective_evals_per_iteration: int = 0,
) -> ResourceEstimate:
    nx = max(1, int(params.get("nx", 256)))
    ny = max(1, int(params.get("ny", 256)))
    points_2d = max(3, int(params.get("points_2d", 101)))
    points_3d = max(3, int(params.get("points_3d", 61)))
    z_chunk = max(1, int(params.get("z_chunk", 8)))
    mode = str(params.get("integration_mode", "cartesian_jacobian"))
    device = str(params.get("torch_device", "cpu")).lower()
    if mode == "polar_interp":
        integration_points = max(
            1, int(params.get("theta_samples", 61)) * int(params.get("phi_samples", 120))
        )
    else:
        integration_points = max(1, int(math.ceil(0.40 * nx * ny)))

    pupil_arrays = nx * ny * (8 * 5 + 16 * 4)
    integration_tables = integration_points * (16 * 4 + 8 * 8)
    phase_2d = integration_points * points_2d * 16 * 3
    output_2d = 3 * points_2d * points_2d * 8 * 2
    optimizer_state = max(0, int(optimizer_variables)) * 8 * 8
    volume = points_3d**3 * 8 * (1 if run_3d else 0)
    volume_work = (
        z_chunk * points_3d * integration_points * 16 + z_chunk * points_3d**2 * 8
        if run_3d
        else 0
    )
    working = pupil_arrays + integration_tables + phase_2d + output_2d + optimizer_state + volume + volume_work
    # Include allocator/workspace headroom; audited peaks should remain below this.
    # Calibrated against measured CPU and CUDA peaks.  The 2x allocator and
    # contraction-workspace allowance keeps representative estimates 10--50%
    # above observed peaks, including full-z CUDA chunks.
    peak = int(math.ceil(working * 2.00 + 96 * 1024**2))
    available_ram = _available_ram_bytes()
    available_vram = _available_vram_bytes(device)
    peak_vram = peak if device in {"cuda", "auto"} and available_vram > 0 else 0

    per_chunk = max(1, points_3d * integration_points * 16 + points_3d**2 * 8)
    safe_budget = int(0.60 * (available_vram if peak_vram else available_ram))
    fixed = max(0, peak - volume_work)
    suggested = max(1, min(points_3d, int(max(1, safe_budget - fixed) // per_chunk)))
    return ResourceEstimate(
        peak_ram_bytes=peak,
        peak_vram_bytes=peak_vram,
        available_ram_bytes=available_ram,
        available_vram_bytes=available_vram,
        predicted_objective_evals_per_iteration=max(
            0, int(predicted_objective_evals_per_iteration)
        ),
        suggested_z_chunk=suggested if run_3d else None,
        details={
            "integration_points": int(integration_points),
            "output_bytes": int(output_2d + volume),
            "working_bytes": int(working),
        },
    )


def enforce_resource_limits(
    estimate: ResourceEstimate,
    *,
    allow_expensive: bool = False,
    hard_fraction: float = 0.70,
    confirm_fraction: float = 0.25,
) -> None:
    ratios = []
    if estimate.available_ram_bytes > 0:
        ratios.append(("RAM", estimate.peak_ram_bytes / estimate.available_ram_bytes))
    if estimate.peak_vram_bytes > 0 and estimate.available_vram_bytes > 0:
        ratios.append(("VRAM", estimate.peak_vram_bytes / estimate.available_vram_bytes))
    worst_name, worst_ratio = max(ratios, key=lambda item: item[1], default=("memory", 0.0))
    if worst_ratio > hard_fraction:
        raise ValueError(
            f"Estimated peak {worst_name} use is {worst_ratio:.1%}, above the {hard_fraction:.0%} safety limit."
        )
    expensive = worst_ratio > confirm_fraction or estimate.predicted_objective_evals_per_iteration > 1000
    if expensive and not allow_expensive:
        raise ValueError(
            "Request is expensive; rerun with --allow-expensive after reviewing the resource estimate."
        )
