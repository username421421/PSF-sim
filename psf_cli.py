import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PSF_simulator import CartesianPSFSimulator, SimulationStoppedError

try:
    import tifffile
except Exception:  # pragma: no cover - optional dependency
    tifffile = None

try:
    from psf_mask_optimizer import (
        OptimizationStoppedError,
        OptimizerConfig,
        PostFresnelMaskOptimizer,
        save_result_npz,
    )

    HAS_OPTIMIZER = True
    OPTIMIZER_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - optional dependency
    OptimizationStoppedError = RuntimeError
    OptimizerConfig = None
    PostFresnelMaskOptimizer = None
    save_result_npz = None
    HAS_OPTIMIZER = False
    OPTIMIZER_IMPORT_ERROR = str(exc)


RESULT_FORMAT = "psf_sim_gui_result"
RESULT_VERSION = 1
CONFIG_FORMAT = "psf_sim_cli_config"
CONFIG_VERSION = 1


DEFAULT_PARAMS = {
    "wavelength": 780.0e-9,
    "NA": 1.4,
    "theta_samples": 61,
    "phi_samples": 120,
    "torch_device": "cpu",
    "nx": 256,
    "ny": 256,
    "size_x": 8.0e-3,
    "size_y": 8.0e-3,
    "pupil_d": 5.6e-3,
    "fresnel_z": 0.30,
    "range_x_um": 2.0,
    "range_y_um": 2.0,
    "range_z_um": 2.0,
    "points_2d": 101,
    "points_3d": 61,
    "z_chunk": 8,
    "n_i": 1.515,
    "n_g": 1.515,
    "t_g": 170.0e-6,
    "n_s": 1.474,
    "t_s": 1.0e-6,
    "n_id": 1.515,
    "n_gd": 1.515,
    "t_gd": 170.0e-6,
    "n_sd": 1.515,
    "use_gaussian": False,
    "gauss_wx": 1.6e-3,
    "gauss_wy": 1.6e-3,
    "mask_offset_x": 0.0,
    "mask_offset_y": 0.0,
}


DEFAULT_OPTIMIZER_SETTINGS = {
    "optimizer_name": "L-BFGS-B",
    "mode": "amplitude_phase",
    "axial_symmetry": False,
    "binary_amplitude": False,
    "binary_structure": "pixel",
    "width_level": 0.50,
    "weight_x": 1.0,
    "weight_y": 1.0,
    "weight_z": 1.0,
    "weight_abs_xy": 0.0,
    "n_starts": 2,
    "maxiter": 30,
    "ftol": 1e-6,
    "gtol": 1e-5,
    "seed": None,
}


def clone_payload(data):
    return pickle.loads(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))


def encode_precise_params(params):
    def encode_value(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return {"type": "float64_hex", "value": float(value).hex()}
        if isinstance(value, dict):
            return {str(k): encode_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode_value(v) for v in value]
        return value

    if not isinstance(params, dict):
        return {}
    return {str(k): encode_value(v) for k, v in params.items()}


def decode_precise_params(params):
    def decode_value(value):
        if isinstance(value, dict) and value.get("type") == "float64_hex" and "value" in value:
            try:
                return float.fromhex(str(value["value"]))
            except Exception:
                return value
        if isinstance(value, dict):
            return {str(k): decode_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode_value(v) for v in value]
        return value

    if not isinstance(params, dict):
        return {}
    return {str(k): decode_value(v) for k, v in params.items()}


def finite_float(value, name):
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def finite_int(value, name):
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def normalise_plot_mode(value):
    text = str(value).strip().upper()
    if text in {"I", "INTENSITY"}:
        return "I"
    if text in {"I2", "I^2", "2P", "TWO-PHOTON", "TWO_PHOTON"}:
        return "I^2"
    raise ValueError("plot mode must be I or I2.")


def intensity_power(plot_mode):
    return 2 if normalise_plot_mode(plot_mode) == "I^2" else 1


def display_intensity(arr, plot_mode="I", normalized=True, apply_power=True):
    data = np.asarray(arr, dtype=float)
    if bool(apply_power):
        data = data ** intensity_power(plot_mode)
    if bool(normalized):
        max_val = float(np.max(data)) if data.size else 0.0
        return data / max_val if max_val > 0.0 else data
    return data


def choose_result_array(result_dict, raw_key, norm_key):
    if raw_key in result_dict and result_dict[raw_key] is not None:
        return np.asarray(result_dict[raw_key], dtype=float)
    return np.asarray(result_dict[norm_key], dtype=float)


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def params_from_ui_values(values):
    values = decode_precise_params(values)
    params = {}
    mapping = {
        "wavelength_nm": ("wavelength", 1e-9, float),
        "NA": ("NA", 1.0, float),
        "theta_samples": ("theta_samples", 1.0, int),
        "phi_samples": ("phi_samples", 1.0, int),
        "torch_device": ("torch_device", 1.0, str),
        "nx": ("nx", 1.0, int),
        "ny": ("ny", 1.0, int),
        "size_x_mm": ("size_x", 1e-3, float),
        "size_y_mm": ("size_y", 1e-3, float),
        "pupil_d_mm": ("pupil_d", 1e-3, float),
        "fresnel_z_m": ("fresnel_z", 1.0, float),
        "range_x_um": ("range_x_um", 1.0, float),
        "range_y_um": ("range_y_um", 1.0, float),
        "range_z_um": ("range_z_um", 1.0, float),
        "points_2d": ("points_2d", 1.0, int),
        "points_3d": ("points_3d", 1.0, int),
        "z_chunk": ("z_chunk", 1.0, int),
        "n_i": ("n_i", 1.0, float),
        "n_g": ("n_g", 1.0, float),
        "t_g_um": ("t_g", 1e-6, float),
        "n_s": ("n_s", 1.0, float),
        "t_s_um": ("t_s", 1e-6, float),
        "n_id": ("n_id", 1.0, float),
        "n_gd": ("n_gd", 1.0, float),
        "t_gd_um": ("t_gd", 1e-6, float),
        "n_sd": ("n_sd", 1.0, float),
        "gauss_wx_mm": ("gauss_wx", 1e-3, float),
        "gauss_wy_mm": ("gauss_wy", 1e-3, float),
        "mask_offset_x_mm": ("mask_offset_x", 1e-3, float),
        "mask_offset_y_mm": ("mask_offset_y", 1e-3, float),
    }
    for src, (dst, scale, cast) in mapping.items():
        if src in values:
            params[dst] = cast(values[src]) * scale if cast is not str else str(values[src])
    if "use_gaussian" in values:
        params["use_gaussian"] = bool(values["use_gaussian"])
    return params


def params_from_core_values(values):
    values = decode_precise_params(values)
    params = {}
    for key in DEFAULT_PARAMS:
        if key in values:
            params[key] = values[key]
    if "range_um" in values:
        params["range_x_um"] = values["range_um"]
        params["range_y_um"] = values["range_um"]
        params["range_z_um"] = values["range_um"]
    return params


def optimizer_settings_from_values(values):
    values = decode_precise_params(values)
    settings = {}
    key_map = {
        "optimizer_name": "optimizer_name",
        "mode": "mode",
        "axial_symmetry": "axial_symmetry",
        "binary_amplitude": "binary_amplitude",
        "binary_structure": "binary_structure",
        "width_level": "width_level",
        "weight_x": "weight_x",
        "weight_y": "weight_y",
        "weight_z": "weight_z",
        "weight_abs_xy": "weight_abs_xy",
        "n_starts": "n_starts",
        "maxiter": "maxiter",
        "ftol": "ftol",
        "gtol": "gtol",
        "seed": "seed",
    }
    for src, dst in key_map.items():
        if src in values:
            settings[dst] = values[src]
    if "seed_text" in values:
        text = str(values["seed_text"]).strip()
        settings["seed"] = None if text == "" else int(text)
    return settings


def load_config(path):
    payload = load_json_file(path)
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON object.")
    payload = decode_precise_params(payload)

    params = {}
    optimizer_settings = {}

    if isinstance(payload.get("ui_values"), dict):
        params.update(params_from_ui_values(payload["ui_values"]))
    if isinstance(payload.get("params"), dict):
        params.update(params_from_core_values(payload["params"]))
    if not params:
        params.update(params_from_ui_values(payload))
        params.update(params_from_core_values(payload))

    if isinstance(payload.get("optimizer_ui_values"), dict):
        optimizer_settings.update(optimizer_settings_from_values(payload["optimizer_ui_values"]))
    if isinstance(payload.get("optimizer_settings"), dict):
        optimizer_settings.update(optimizer_settings_from_values(payload["optimizer_settings"]))
    if not optimizer_settings:
        optimizer_settings.update(optimizer_settings_from_values(payload))

    return params, optimizer_settings


def mask_path_from_config(path):
    if not path:
        return None
    try:
        payload = decode_precise_params(load_json_file(path))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    mask_info = payload.get("mask")
    if not isinstance(mask_info, dict):
        return None
    if not bool(mask_info.get("loaded", False)):
        return None
    mask_path = mask_info.get("path")
    if mask_path is None:
        return None
    mask_path = str(mask_path).strip()
    return mask_path if mask_path else None


def resolve_mask_path(args):
    explicit = getattr(args, "mask", None)
    if explicit:
        return explicit
    return mask_path_from_config(getattr(args, "config", None))


def load_objective_lens(path):
    payload = decode_precise_params(load_json_file(path))
    values = payload.get("values", payload)
    if not isinstance(values, dict):
        raise ValueError("objective lens preset must contain a values object.")
    required = ("NA", "pupil_d_mm", "n_id", "n_gd", "t_gd_um", "n_sd")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"objective lens preset missing keys: {', '.join(missing)}.")
    return {
        "NA": float(values["NA"]),
        "pupil_d": float(values["pupil_d_mm"]) * 1e-3,
        "n_id": float(values["n_id"]),
        "n_gd": float(values["n_gd"]),
        "t_gd": float(values["t_gd_um"]) * 1e-6,
        "n_sd": float(values["n_sd"]),
    }


def validate_params(params):
    p = dict(params)
    for key in ("wavelength", "NA", "size_x", "size_y", "pupil_d", "fresnel_z"):
        p[key] = finite_float(p[key], key)
    for key in ("range_x_um", "range_y_um", "range_z_um"):
        p[key] = finite_float(p[key], key)
        if p[key] <= 0.0:
            raise ValueError("PSF ranges X/Y/Z must be > 0.")
    for key in ("nx", "ny", "theta_samples", "phi_samples", "points_2d", "points_3d", "z_chunk"):
        p[key] = finite_int(p[key], key)

    if p["wavelength"] <= 0.0:
        raise ValueError("wavelength must be > 0.")
    if p["nx"] < 8 or p["ny"] < 8:
        raise ValueError("Nx and Ny must be >= 8.")
    if p["size_x"] <= 0.0 or p["size_y"] <= 0.0:
        raise ValueError("Mask size must be positive.")
    if p["pupil_d"] <= 0.0:
        raise ValueError("Entrance pupil diameter must be positive.")
    if p["points_2d"] < 9 or p["points_3d"] < 9:
        raise ValueError("2D/3D points must be >= 9.")
    if p["z_chunk"] < 1:
        raise ValueError("3D z chunk must be >= 1.")
    if str(p["torch_device"]).strip().lower() not in {"auto", "cpu", "cuda"}:
        raise ValueError("Torch device must be one of: auto, cpu, cuda.")
    p["torch_device"] = str(p["torch_device"]).strip().lower()

    for key in ("n_i", "n_g", "n_s", "n_id", "n_gd", "n_sd"):
        p[key] = finite_float(p[key], key)
        if p[key] <= 0.0:
            raise ValueError(f"{key} must be > 0.")
    for key in ("t_g", "t_s", "t_gd"):
        p[key] = finite_float(p[key], key)
        if p[key] < 0.0:
            raise ValueError(f"{key} must be >= 0.")
    p["use_gaussian"] = bool(p["use_gaussian"])
    p["gauss_wx"] = finite_float(p["gauss_wx"], "gauss_wx")
    p["gauss_wy"] = finite_float(p["gauss_wy"], "gauss_wy")
    if p["use_gaussian"] and (p["gauss_wx"] <= 0.0 or p["gauss_wy"] <= 0.0):
        raise ValueError("Gaussian wx/wy must be positive when Gaussian is enabled.")
    p["mask_offset_x"] = finite_float(p["mask_offset_x"], "mask_offset_x")
    p["mask_offset_y"] = finite_float(p["mask_offset_y"], "mask_offset_y")
    return p


def apply_param_overrides(params, args):
    p = dict(params)
    if getattr(args, "objective", None):
        p.update(load_objective_lens(args.objective))

    unit_overrides = {
        "wavelength_nm": ("wavelength", 1e-9, float),
        "na": ("NA", 1.0, float),
        "theta_samples": ("theta_samples", 1.0, int),
        "phi_samples": ("phi_samples", 1.0, int),
        "torch_device": ("torch_device", 1.0, str),
        "nx": ("nx", 1.0, int),
        "ny": ("ny", 1.0, int),
        "size_x_mm": ("size_x", 1e-3, float),
        "size_y_mm": ("size_y", 1e-3, float),
        "pupil_d_mm": ("pupil_d", 1e-3, float),
        "fresnel_z_m": ("fresnel_z", 1.0, float),
        "range_x_um": ("range_x_um", 1.0, float),
        "range_y_um": ("range_y_um", 1.0, float),
        "range_z_um": ("range_z_um", 1.0, float),
        "points_2d": ("points_2d", 1.0, int),
        "points_3d": ("points_3d", 1.0, int),
        "z_chunk": ("z_chunk", 1.0, int),
        "n_i": ("n_i", 1.0, float),
        "n_g": ("n_g", 1.0, float),
        "t_g_um": ("t_g", 1e-6, float),
        "n_s": ("n_s", 1.0, float),
        "t_s_um": ("t_s", 1e-6, float),
        "n_id": ("n_id", 1.0, float),
        "n_gd": ("n_gd", 1.0, float),
        "t_gd_um": ("t_gd", 1e-6, float),
        "n_sd": ("n_sd", 1.0, float),
        "gauss_wx_mm": ("gauss_wx", 1e-3, float),
        "gauss_wy_mm": ("gauss_wy", 1e-3, float),
        "mask_offset_x_mm": ("mask_offset_x", 1e-3, float),
        "mask_offset_y_mm": ("mask_offset_y", 1e-3, float),
    }
    if getattr(args, "range_um", None) is not None:
        p["range_x_um"] = float(args.range_um)
        p["range_y_um"] = float(args.range_um)
        p["range_z_um"] = float(args.range_um)
    if getattr(args, "points", None) is not None:
        p["points_2d"] = int(args.points)
        p["points_3d"] = int(args.points)
    for attr, (dst, scale, cast) in unit_overrides.items():
        value = getattr(args, attr, None)
        if value is not None:
            p[dst] = cast(value) * scale if cast is not str else str(value)
    use_gaussian = getattr(args, "use_gaussian", None)
    if use_gaussian is not None:
        p["use_gaussian"] = bool(use_gaussian)
    return validate_params(p)


def resolve_params(args):
    params = dict(DEFAULT_PARAMS)
    if getattr(args, "config", None):
        loaded_params, _loaded_opt = load_config(args.config)
        params.update(loaded_params)
    return apply_param_overrides(params, args)


def resolve_optimizer_settings(args):
    settings = dict(DEFAULT_OPTIMIZER_SETTINGS)
    if getattr(args, "config", None):
        _loaded_params, loaded_opt = load_config(args.config)
        settings.update(loaded_opt)

    overrides = {
        "optimizer_name": "optimizer_name",
        "optimizer_mode": "mode",
        "binary_structure": "binary_structure",
        "opt_width_level": "width_level",
        "weight_x": "weight_x",
        "weight_y": "weight_y",
        "weight_z": "weight_z",
        "weight_abs_xy": "weight_abs_xy",
        "n_starts": "n_starts",
        "maxiter": "maxiter",
        "ftol": "ftol",
        "gtol": "gtol",
        "seed": "seed",
    }
    for attr, key in overrides.items():
        value = getattr(args, attr, None)
        if value is not None:
            settings[key] = value

    for attr, key in (
        ("axial_symmetry", "axial_symmetry"),
        ("binary_amplitude", "binary_amplitude"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            settings[key] = bool(value)

    if settings["seed"] == "":
        settings["seed"] = None
    elif settings["seed"] is not None:
        settings["seed"] = int(settings["seed"])

    settings["mode"] = str(settings["mode"]).strip().lower()
    settings["binary_structure"] = str(settings["binary_structure"]).strip().lower()
    settings["optimizer_name"] = str(settings["optimizer_name"]).strip()
    settings["width_level"] = float(settings["width_level"])
    settings["weight_x"] = float(settings["weight_x"])
    settings["weight_y"] = float(settings["weight_y"])
    settings["weight_z"] = float(settings["weight_z"])
    settings["weight_abs_xy"] = float(settings["weight_abs_xy"])
    settings["n_starts"] = int(settings["n_starts"])
    settings["maxiter"] = int(settings["maxiter"])
    settings["ftol"] = float(settings["ftol"])
    settings["gtol"] = float(settings["gtol"])
    return settings


def create_simulator_from_params(params):
    return CartesianPSFSimulator(
        wavelength=float(params["wavelength"]),
        NA=float(params["NA"]),
        theta_samples=int(params["theta_samples"]),
        phi_samples=int(params["phi_samples"]),
        torch_device=str(params.get("torch_device", "cpu")).strip().lower(),
        n_i=float(params["n_i"]),
        n_g=float(params["n_g"]),
        t_g=float(params["t_g"]),
        n_s=float(params["n_s"]),
        t_s=float(params["t_s"]),
        n_id=float(params["n_id"]),
        n_gd=float(params["n_gd"]),
        t_gd=float(params["t_gd"]),
        n_sd=float(params.get("n_sd", 1.515)),
    )


def read_mask(path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in {".csv", ".txt"}:
        try:
            arr = np.loadtxt(path, delimiter=",")
        except Exception:
            arr = np.loadtxt(path)
    elif ext in {".tif", ".tiff"}:
        if tifffile is not None:
            arr = tifffile.imread(path)
        else:
            arr = plt.imread(path)
    else:
        arr = plt.imread(path)

    arr = np.asarray(arr, dtype=float)
    if arr.ndim >= 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = np.mean(arr[..., :3], axis=2)
        elif arr.shape[0] in (3, 4):
            arr = np.mean(arr[:3, ...], axis=0)
        else:
            arr = arr[0]
    if arr.ndim != 2:
        raise ValueError("Mask must be a 2D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Mask contains NaN or Inf.")

    amin = float(np.min(arr))
    amax = float(np.max(arr))
    rng = amax - amin
    if rng <= 0.0:
        arr = np.ones_like(arr) if amax > 0.0 else np.zeros_like(arr)
    else:
        arr = (arr - amin) / rng
    return np.clip(arr, 0.0, 1.0)


def resize_mask(mask, shape):
    mask = np.asarray(mask, dtype=float)
    if tuple(mask.shape) == tuple(shape):
        return np.clip(mask, 0.0, 1.0)
    mask_t = torch.as_tensor(mask, dtype=torch.float64)
    resized_t = F.interpolate(
        mask_t[None, None, :, :],
        size=(int(shape[0]), int(shape[1])),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return torch.clamp(resized_t, 0.0, 1.0).detach().cpu().numpy()


def mask_rotation_quadrants(args):
    q = getattr(args, "mask_rotate_quadrants", None)
    if q is not None:
        return int(q) % 4
    deg = getattr(args, "mask_rotate", None)
    if deg is None:
        return 0
    deg = int(deg)
    if deg % 90 != 0:
        raise ValueError("--mask-rotate must be a multiple of 90 degrees.")
    return (deg // 90) % 4


def build_input_mask(params, mask_path=None, rotation_quadrants=0):
    shape = (int(params["nx"]), int(params["ny"]))
    if mask_path:
        base = read_mask(mask_path)
        k = int(rotation_quadrants) % 4
        if k:
            base = np.rot90(base, k=k)
    else:
        base = np.ones(shape, dtype=float)
    resized = resize_mask(base, shape)
    return CartesianPSFSimulator.apply_mask_offset(
        resized,
        size_x=float(params["size_x"]),
        size_y=float(params["size_y"]),
        offset_x=float(params["mask_offset_x"]),
        offset_y=float(params["mask_offset_y"]),
    )


class ProgressPrinter:
    def __init__(self, quiet=False, label=""):
        self.quiet = bool(quiet)
        self.label = str(label)
        self._last_pct = -100.0
        self._last_time = 0.0

    def emit(self, percent, text, force=False):
        if self.quiet:
            return
        now = time.time()
        pct = float(min(100.0, max(0.0, percent)))
        if (
            bool(force)
            or pct >= self._last_pct + 5.0
            or pct >= 100.0
            or (now - self._last_time) >= 5.0
        ):
            prefix = f"{self.label}: " if self.label else ""
            print(f"{prefix}{pct:6.2f}%  {text}", file=sys.stderr, flush=True)
            self._last_pct = pct
            self._last_time = now


def run_simulation(params, mask, run_3d=False, progress=None):
    simulator = create_simulator_from_params(params)

    def emit(percent, text):
        if progress is not None:
            progress.emit(percent, text)

    if run_3d:
        nx_in = int(params["nx"])
        ny_in = int(params["ny"])
        max_n = max(nx_in, ny_in, 2)
        prep_est = float(nx_in * ny_in) * (
            1.0 if np.isclose(float(params["fresnel_z"]), 0.0) else np.log2(max_n)
        )
        if simulator.integration_mode == "cartesian_jacobian":
            pupil_radius = 0.5 * float(params["pupil_d"])
            area_ratio = np.pi * pupil_radius * pupil_radius / max(
                float(params["size_x"] * params["size_y"]), 1e-30
            )
            area_ratio = min(1.0, max(0.0, area_ratio))
            pint_est = max(1.0, area_ratio * nx_in * ny_in)
        else:
            pint_est = float(max(1, simulator.theta_samples * simulator.phi_samples))
        n3 = max(1, int(params["points_3d"]))
        vol_est = float(n3 * n3 * n3) * pint_est
        total_est = max(1.0, prep_est + vol_est)
        prep_share = prep_est / total_est

        prep = simulator.prepare_pupil_field_from_input(
            input_intensity=mask,
            size_x=float(params["size_x"]),
            size_y=float(params["size_y"]),
            entrance_pupil_diameter=float(params["pupil_d"]),
            fresnel_distance=float(params["fresnel_z"]),
            use_gaussian_profile=bool(params["use_gaussian"]),
            gaussian_waist_x=float(params["gauss_wx"]) if bool(params["use_gaussian"]) else None,
            gaussian_waist_y=float(params["gauss_wy"]) if bool(params["use_gaussian"]) else None,
            custom_mask=None,
            ring_inner_diameter=None,
            ring_outer_diameter=None,
            progress_callback=lambda stage, frac: emit(
                100.0 * prep_share * float(frac), f"Pupil prep: {stage}"
            ),
        )
        res3d = simulator.simulate_psf_volume_from_pupil_field(
            pupil_field=prep.get("field_at_pupil_for_psf", prep["field_at_pupil"]),
            size_x=float(prep.get("size_x_for_psf", params["size_x"])),
            size_y=float(prep.get("size_y_for_psf", params["size_y"])),
            entrance_pupil_diameter=float(params["pupil_d"]),
            range_microns_x=float(params["range_x_um"]),
            range_microns_y=float(params["range_y_um"]),
            range_microns_z=float(params["range_z_um"]),
            num_points=int(params["points_3d"]),
            z_chunk_size=int(params["z_chunk"]),
            verbose=False,
            progress_callback=lambda stage, frac: emit(
                100.0 * (prep_share + (1.0 - prep_share) * float(frac)), f"3D: {stage}"
            ),
        )
        res2d = {
            "x_coords": res3d["x_coords"],
            "y_coords": res3d["y_coords"],
            "z_coords": res3d["z_coords"],
            "z_focus": res3d["z_focus"],
            "I_xz": res3d["I_xz"],
            "I_xy": res3d["I_xy"],
            "I_yz": res3d["I_yz"],
            "I_xz_norm": res3d["I_xz_norm"],
            "I_xy_norm": res3d["I_xy_norm"],
            "I_yz_norm": res3d["I_yz_norm"],
            "fwhm_x_m": res3d["fwhm_x_m"],
            "fwhm_y_m": res3d["fwhm_y_m"],
            "fwhm_z_m": res3d["fwhm_z_m"],
            "fwhm_x_um": res3d["fwhm_x_um"],
            "fwhm_y_um": res3d["fwhm_y_um"],
            "fwhm_z_um": res3d["fwhm_z_um"],
            "x_profile_at_y0": res3d["x_profile_at_y0"],
            "y_profile_at_x0": res3d["y_profile_at_x0"],
            "z_profile_at_x0": res3d["z_profile_at_x0"],
        }
        res2d.update(prep)
    else:
        res2d = simulator.simulate_full_pipeline(
            input_intensity=mask,
            size_x=float(params["size_x"]),
            size_y=float(params["size_y"]),
            entrance_pupil_diameter=float(params["pupil_d"]),
            fresnel_distance=float(params["fresnel_z"]),
            use_gaussian_profile=bool(params["use_gaussian"]),
            gaussian_waist_x=float(params["gauss_wx"]) if bool(params["use_gaussian"]) else None,
            gaussian_waist_y=float(params["gauss_wy"]) if bool(params["use_gaussian"]) else None,
            custom_mask=None,
            ring_inner_diameter=None,
            ring_outer_diameter=None,
            range_microns_x=float(params["range_x_um"]),
            range_microns_y=float(params["range_y_um"]),
            range_microns_z=float(params["range_z_um"]),
            num_points=int(params["points_2d"]),
            verbose=False,
            progress_callback=lambda stage, frac: emit(100.0 * float(frac), f"2D: {stage}"),
        )
        res3d = None

    if progress is not None:
        progress.emit(100.0, "Done", force=True)
    return simulator, res2d, res3d


def run_profile(params, mask, axis, z_um=None, width_level=0.5, progress=None, num_points=None):
    simulator = create_simulator_from_params(params)

    def emit(percent, text):
        if progress is not None:
            progress.emit(percent, text)

    axis = str(axis).strip().lower()
    if num_points is None:
        num_points = int(params["points_2d"])
    z_position = None if z_um is None else float(z_um) * 1e-6
    result = simulator.simulate_psf_profile_full_pipeline(
        input_intensity=mask,
        size_x=float(params["size_x"]),
        size_y=float(params["size_y"]),
        entrance_pupil_diameter=float(params["pupil_d"]),
        fresnel_distance=float(params["fresnel_z"]),
        axis=axis,
        use_gaussian_profile=bool(params["use_gaussian"]),
        gaussian_waist_x=float(params["gauss_wx"]) if bool(params["use_gaussian"]) else None,
        gaussian_waist_y=float(params["gauss_wy"]) if bool(params["use_gaussian"]) else None,
        custom_mask=None,
        ring_inner_diameter=None,
        ring_outer_diameter=None,
        range_microns_x=float(params["range_x_um"]),
        range_microns_y=float(params["range_y_um"]),
        range_microns_z=float(params["range_z_um"]),
        num_points=int(num_points),
        z_position=z_position,
        width_level=float(width_level),
        verbose=False,
        progress_callback=lambda stage, frac: emit(100.0 * float(frac), str(stage)),
    )
    if progress is not None:
        progress.emit(100.0, "Done", force=True)
    return simulator, result


def ensure_parent(path):
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def save_result(path, params, results_2d, results_3d=None):
    ensure_parent(path)
    payload = {
        "format": RESULT_FORMAT,
        "version": RESULT_VERSION,
        "params": clone_payload(params),
        "results_2d": clone_payload(results_2d),
    }
    if results_3d is not None:
        payload["results_3d"] = clone_payload(results_3d)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_result(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError("result file does not contain a result payload.")
    if "params" not in payload or "results_2d" not in payload:
        raise ValueError("result file is missing params or results_2d.")
    return payload


def write_csv_metadata_row(writer, dataset, key, value):
    writer.writerow(["metadata", str(dataset), str(key), value, "", "", "", "", ""])


def write_csv_heatmap_rows(writer, dataset, axis_1_label, axis_1_coords, axis_2_label, axis_2_coords, values):
    x = np.asarray(axis_1_coords, dtype=float).ravel()
    y = np.asarray(axis_2_coords, dtype=float).ravel()
    arr = np.asarray(values, dtype=float)
    expected_shape = (int(y.size), int(x.size))
    if arr.shape != expected_shape:
        if arr.shape == (int(x.size), int(y.size)):
            arr = arr.T
        else:
            raise ValueError(f"{dataset} heatmap shape mismatch: got {arr.shape}, expected {expected_shape}.")
    for iy, yv in enumerate(y):
        row = arr[iy]
        for ix, xv in enumerate(x):
            writer.writerow(
                ["heatmap", str(dataset), "", "", str(axis_1_label), float(xv), str(axis_2_label), float(yv), float(row[ix])]
            )


def write_csv_profile_rows(writer, dataset, axis_label, axis_coords, values):
    x = np.asarray(axis_coords, dtype=float).ravel()
    y = np.asarray(values, dtype=float).ravel()
    if int(x.size) != int(y.size):
        raise ValueError(f"{dataset} profile length mismatch: got {int(y.size)} values for {int(x.size)} coordinates.")
    for xv, yv in zip(x, y):
        writer.writerow(["profile", str(dataset), "", "", str(axis_label), float(xv), "", "", float(yv)])


def collect_export_data(simulator, results_2d, plot_mode="I", normalized=True, width_level=0.5):
    mode = normalise_plot_mode(plot_mode)
    level = min(0.99, max(0.01, float(width_level)))
    x_um = np.asarray(results_2d["x_coords"], dtype=float) * 1e6
    y_um = np.asarray(results_2d["y_coords"], dtype=float) * 1e6
    z_um = np.asarray(results_2d["z_coords"], dtype=float) * 1e6
    i_xy = display_intensity(
        choose_result_array(results_2d, "I_xy", "I_xy_norm"),
        plot_mode=mode,
        normalized=normalized,
    )
    i_xz = display_intensity(
        choose_result_array(results_2d, "I_xz", "I_xz_norm"),
        plot_mode=mode,
        normalized=normalized,
    )
    i_yz = display_intensity(
        choose_result_array(results_2d, "I_yz", "I_yz_norm"),
        plot_mode=mode,
        normalized=normalized,
    )
    width_metrics = simulator.compute_peak_width_metrics(
        x_coords=results_2d["x_coords"],
        y_coords=results_2d["y_coords"],
        z_coords=results_2d["z_coords"],
        i_xy_norm=i_xy,
        i_xz_norm=i_xz,
        i_yz_norm=i_yz,
        level=level,
    )
    focus_z_um = np.nan
    if "z_focus" in results_2d:
        try:
            focus_z_um = float(results_2d["z_focus"]) * 1e6
        except Exception:
            focus_z_um = np.nan
    return {
        "x_um": x_um,
        "y_um": y_um,
        "z_um": z_um,
        "i_xy": np.asarray(i_xy, dtype=float),
        "i_xz": np.asarray(i_xz, dtype=float),
        "i_yz": np.asarray(i_yz, dtype=float),
        "x_profile": np.asarray(width_metrics["x_profile_peak"], dtype=float),
        "y_profile": np.asarray(width_metrics["y_profile_peak"], dtype=float),
        "z_profile": np.asarray(width_metrics["z_profile_peak"], dtype=float),
        "width_metrics": dict(width_metrics),
        "plot_mode": mode,
        "normalized": bool(normalized),
        "focus_z_um": float(focus_z_um),
    }


def write_export_csv(path, simulator, results_2d, plot_mode="I", normalized=True, width_level=0.5):
    ensure_parent(path)
    export = collect_export_data(
        simulator,
        results_2d,
        plot_mode=plot_mode,
        normalized=normalized,
        width_level=width_level,
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "record_type",
                "dataset",
                "key",
                "meta_value",
                "axis_1_label",
                "axis_1_value",
                "axis_2_label",
                "axis_2_value",
                "value",
            ]
        )
        write_csv_metadata_row(writer, "export", "format", "psf_sim_gui_csv_export_v1")
        write_csv_metadata_row(writer, "display", "plot_mode", export["plot_mode"])
        write_csv_metadata_row(writer, "display", "normalized", int(export["normalized"]))
        write_csv_metadata_row(writer, "display", "width_level", float(export["width_metrics"].get("level", width_level)))
        write_csv_metadata_row(writer, "result", "focus_z_um", export["focus_z_um"])
        write_csv_metadata_row(writer, "result", "width_x_um", float(export["width_metrics"].get("width_x_um", np.nan)))
        write_csv_metadata_row(writer, "result", "width_y_um", float(export["width_metrics"].get("width_y_um", np.nan)))
        write_csv_metadata_row(writer, "result", "width_z_um", float(export["width_metrics"].get("width_z_um", np.nan)))
        write_csv_metadata_row(writer, "result", "width_ratio_zx", float(export["width_metrics"].get("width_ratio_zx", np.nan)))
        write_csv_metadata_row(writer, "result", "width_ratio_zy", float(export["width_metrics"].get("width_ratio_zy", np.nan)))

        write_csv_heatmap_rows(writer, "XY", "x_um", export["x_um"], "y_um", export["y_um"], export["i_xy"])
        write_csv_heatmap_rows(writer, "XZ", "x_um", export["x_um"], "z_um", export["z_um"], export["i_xz"])
        write_csv_heatmap_rows(writer, "YZ", "y_um", export["y_um"], "z_um", export["z_um"], export["i_yz"])
        write_csv_profile_rows(writer, "X", "x_um", export["x_um"], export["x_profile"])
        write_csv_profile_rows(writer, "Y", "y_um", export["y_um"], export["y_profile"])
        write_csv_profile_rows(writer, "Z", "z_um", export["z_um"], export["z_profile"])
    return export


def write_profile_csv(path, profile_result, plot_mode="I", normalized=True):
    ensure_parent(path)
    axis = str(profile_result["axis"]).lower()
    coords_um = np.asarray(profile_result["coords_um"], dtype=float)
    intensity = np.asarray(profile_result["intensity"], dtype=float)
    intensity_norm = np.asarray(profile_result["intensity_norm"], dtype=float)
    display = display_intensity(intensity, plot_mode=plot_mode, normalized=normalized)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "axis",
                "coord_um",
                "intensity",
                "intensity_norm",
                "display_intensity",
            ]
        )
        for coord, raw, norm, shown in zip(coords_um, intensity, intensity_norm, display):
            writer.writerow([axis, float(coord), float(raw), float(norm), float(shown)])


def profile_summary(params, profile_result, plot_mode="I", normalized=True, mask_path=None):
    return json_safe(
        {
            "axis": str(profile_result["axis"]).lower(),
            "num_points": int(np.asarray(profile_result["coords"]).size),
            "coord_min_um": float(np.min(profile_result["coords_um"])),
            "coord_max_um": float(np.max(profile_result["coords_um"])),
            "width_level": float(profile_result["width_level"]),
            "width_um": float(profile_result["width_um"]),
            "peak_index": int(profile_result["peak_index"]),
            "peak_coord_um": float(profile_result["peak_coord_um"]),
            "peak_intensity": float(profile_result["peak_intensity"]),
            "z_focus_um": float(profile_result["z_focus_um"]),
            "z_position_um": float(profile_result["z_position_um"]),
            "plot_mode": normalise_plot_mode(plot_mode),
            "normalized": bool(normalized),
            "mask_path": mask_path,
            "grid": {
                "nx": int(params["nx"]),
                "ny": int(params["ny"]),
                "size_x_mm": float(params["size_x"]) * 1e3,
                "size_y_mm": float(params["size_y"]) * 1e3,
            },
            "optics": {
                "wavelength_nm": float(params["wavelength"]) * 1e9,
                "NA": float(params["NA"]),
                "pupil_d_mm": float(params["pupil_d"]) * 1e3,
                "fresnel_z_m": float(params["fresnel_z"]),
            },
            "pupil_trimmed_for_psf": bool(profile_result.get("pupil_trimmed_for_psf", False)),
        }
    )


def print_profile_summary(summary, json_output=False):
    if json_output:
        print(json.dumps(summary, indent=2))
        return
    print(
        f"Profile: axis={summary['axis'].upper()}, points={summary['num_points']}, "
        f"coord=[{summary['coord_min_um']:.6g}, {summary['coord_max_um']:.6g}] um"
    )
    print(
        f"Peak: coord={summary['peak_coord_um']:.6g} um, "
        f"z_focus={summary['z_focus_um']:.6g} um, z_used={summary['z_position_um']:.6g} um"
    )
    print(f"Width @{summary['width_level']:.3g}: {summary['width_um']:.6g} um")


def write_profile_plot_png(path, profile_result, plot_mode="I", normalized=True):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ensure_parent(path)
    axis = str(profile_result["axis"]).lower()
    coords_um = np.asarray(profile_result["coords_um"], dtype=float)
    intensity = display_intensity(
        np.asarray(profile_result["intensity"], dtype=float),
        plot_mode=plot_mode,
        normalized=normalized,
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(coords_um, intensity, color="#1B3B6F", linewidth=1.4)
    ax.set_xlabel(f"{axis} (um)")
    ax.set_ylabel("Normalized intensity" if normalized else "Intensity (a.u.)")
    ax.set_title(f"{axis.upper()} intensity profile")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def ring_radius_grid(max_radius_mm, radius_points):
    max_radius_mm = finite_float(max_radius_mm, "max_radius_mm")
    radius_points = finite_int(radius_points, "radius_points")
    if max_radius_mm < 0.0:
        raise ValueError("--max-radius-mm must be >= 0.")
    if radius_points < 1:
        raise ValueError("--radius-points must be >= 1.")
    return np.linspace(0.0, max_radius_mm, radius_points, dtype=float)


def ring_radius_pairs(radius_grid_mm):
    grid = np.asarray(radius_grid_mm, dtype=float).ravel()
    if grid.size < 1:
        raise ValueError("radius grid is empty.")
    pairs = []
    for outer_idx, outer_mm in enumerate(grid):
        for inner_idx in range(outer_idx + 1):
            pairs.append((float(grid[inner_idx]), float(outer_mm)))
    return pairs


def build_radius_map(shape, size_x, size_y):
    nx, ny = int(shape[0]), int(shape[1])
    x = np.linspace(-0.5 * float(size_x), 0.5 * float(size_x), nx, endpoint=False, dtype=float)
    y = np.linspace(-0.5 * float(size_y), 0.5 * float(size_y), ny, endpoint=False, dtype=float)
    return np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)


def create_ring_block_mask_from_radius_map(radius_map, inner_radius_m, outer_radius_m):
    inner_radius_m = finite_float(inner_radius_m, "inner_radius_m")
    outer_radius_m = finite_float(outer_radius_m, "outer_radius_m")
    if inner_radius_m < 0.0 or outer_radius_m < 0.0:
        raise ValueError("ring radii must be >= 0.")
    if inner_radius_m > outer_radius_m:
        raise ValueError("inner radius must be <= outer radius.")

    mask = np.ones_like(np.asarray(radius_map, dtype=float), dtype=float)
    if outer_radius_m > inner_radius_m:
        blocked = (radius_map >= inner_radius_m) & (radius_map <= outer_radius_m)
        mask[blocked] = 0.0
    return mask


def default_output_path(output_dir, override, filename):
    if override:
        return Path(override).expanduser()
    return Path(output_dir).expanduser() / filename


def write_ring_sweep_metadata(path, metadata):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(metadata), f, indent=2)


def write_ring_profiles_npz(path, arrays):
    ensure_parent(path)
    np.savez_compressed(path, **arrays)


def print_ring_sweep_summary(summary, json_output=False):
    if json_output:
        print(json.dumps(json_safe(summary), indent=2))
        return
    print(
        f"Ring sweep: axis={summary['profile_axis'].upper()}, masks={summary['mask_count']}, "
        f"profile_points={summary['profile_points']}, range=+/-{summary['profile_range_um']:.6g} um"
    )
    print(f"Profiles: {summary['outputs']['profiles_csv']}")
    print(f"Profile NPZ: {summary['outputs']['profiles_npz']}")
    print(f"Summary:  {summary['outputs']['summary_csv']}")
    print(f"Metadata: {summary['outputs']['metadata_json']}")


def ndarray_metadata(value):
    arr = np.asarray(value)
    return {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(np.real(arr))) if arr.size else np.nan,
        "max": float(np.nanmax(np.real(arr))) if arr.size else np.nan,
    }


def write_arrays_npz(path, params, results_2d, results_3d=None):
    ensure_parent(path)
    arrays = {}
    metadata = {
        "format": "psf_sim_cli_arrays_npz",
        "params": json_safe(params),
        "results_2d": {},
        "results_3d": {},
    }
    for prefix, results, meta_key in (
        ("results_2d", results_2d, "results_2d"),
        ("results_3d", results_3d, "results_3d"),
    ):
        if not isinstance(results, dict):
            continue
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                arr_key = f"{prefix}__{key}"
                arrays[arr_key] = value
                metadata[meta_key][key] = ndarray_metadata(value)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata), dtype=object)
    np.savez_compressed(path, **arrays)
    return metadata


def write_plot_png(path, results_2d, plot_mode="I", normalized=True):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ensure_parent(path)
    mode = normalise_plot_mode(plot_mode)
    x_um = np.asarray(results_2d["x_coords"], dtype=float) * 1e6
    y_um = np.asarray(results_2d["y_coords"], dtype=float) * 1e6
    z_um = np.asarray(results_2d["z_coords"], dtype=float) * 1e6
    i_xy = display_intensity(choose_result_array(results_2d, "I_xy", "I_xy_norm"), mode, normalized)
    i_xz = display_intensity(choose_result_array(results_2d, "I_xz", "I_xz_norm"), mode, normalized)
    i_yz = display_intensity(choose_result_array(results_2d, "I_yz", "I_yz_norm"), mode, normalized)
    input_i = np.asarray(results_2d.get("input_intensity", np.zeros((2, 2))), dtype=float)
    pupil_before = display_intensity(
        choose_result_array(results_2d, "intensity_at_entrance_pupil", "intensity_at_entrance_pupil_norm"),
        mode,
        normalized,
        apply_power=False,
    )
    pupil_after = display_intensity(
        choose_result_array(results_2d, "intensity_after_pupil_mask", "intensity_after_pupil_mask_norm"),
        mode,
        normalized,
        apply_power=False,
    )

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    panels = [
        (axes[0, 0], input_i.T, "Input mask", None),
        (axes[0, 1], pupil_before.T, "After Fresnel", None),
        (axes[0, 2], pupil_after.T, "Pupil", None),
        (axes[1, 0], i_xy, "XY", [x_um[0], x_um[-1], y_um[0], y_um[-1]]),
        (axes[1, 1], i_xz, "XZ", [x_um[0], x_um[-1], z_um[0], z_um[-1]]),
        (axes[1, 2], i_yz, "YZ", [y_um[0], y_um[-1], z_um[0], z_um[-1]]),
    ]
    for ax, arr, title, extent in panels:
        kwargs = {
            "origin": "lower",
            "cmap": "magma",
            "aspect": "equal",
            "interpolation": "nearest",
        }
        if extent is not None:
            kwargs["extent"] = extent
        im = ax.imshow(arr, **kwargs)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[1, 0].set_xlabel("x (um)")
    axes[1, 0].set_ylabel("y (um)")
    axes[1, 1].set_xlabel("x (um)")
    axes[1, 1].set_ylabel("z (um)")
    axes[1, 2].set_xlabel("y (um)")
    axes[1, 2].set_ylabel("z (um)")
    fig.suptitle(f"PSF simulation ({mode}, {'normalized' if normalized else 'raw'})")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return ndarray_metadata(value)
    return value


def build_summary(params, simulator, results_2d, results_3d=None, plot_mode="I", normalized=True, width_level=0.5):
    export = collect_export_data(
        simulator,
        results_2d,
        plot_mode=plot_mode,
        normalized=normalized,
        width_level=width_level,
    )
    widths = export["width_metrics"]
    summary = {
        "run_type": "3d" if results_3d is not None else "2d",
        "grid": {
            "nx": int(params["nx"]),
            "ny": int(params["ny"]),
            "size_x_mm": float(params["size_x"]) * 1e3,
            "size_y_mm": float(params["size_y"]) * 1e3,
        },
        "optics": {
            "wavelength_nm": float(params["wavelength"]) * 1e9,
            "NA": float(params["NA"]),
            "pupil_d_mm": float(params["pupil_d"]) * 1e3,
            "theta_samples": int(params["theta_samples"]),
            "phi_samples": int(params["phi_samples"]),
            "torch_device": str(params["torch_device"]),
        },
        "psf": {
            "focus_z_um": export["focus_z_um"],
            "fwhm_x_um": float(results_2d.get("fwhm_x_um", np.nan)),
            "fwhm_y_um": float(results_2d.get("fwhm_y_um", np.nan)),
            "fwhm_z_um": float(results_2d.get("fwhm_z_um", np.nan)),
            "width_level": float(widths.get("level", width_level)),
            "width_x_um": float(widths.get("width_x_um", np.nan)),
            "width_y_um": float(widths.get("width_y_um", np.nan)),
            "width_z_um": float(widths.get("width_z_um", np.nan)),
            "width_ratio_zx": float(widths.get("width_ratio_zx", np.nan)),
            "width_ratio_zy": float(widths.get("width_ratio_zy", np.nan)),
        },
        "display": {
            "plot_mode": normalise_plot_mode(plot_mode),
            "normalized": bool(normalized),
        },
        "shapes": {
            "I_xy": list(np.asarray(results_2d["I_xy"]).shape),
            "I_xz": list(np.asarray(results_2d["I_xz"]).shape),
            "I_yz": list(np.asarray(results_2d["I_yz"]).shape),
        },
    }
    if results_3d is not None and "I_xyz" in results_3d:
        summary["shapes"]["I_xyz"] = list(np.asarray(results_3d["I_xyz"]).shape)
    return json_safe(summary)


def print_summary(summary, json_output=False):
    if json_output:
        print(json.dumps(summary, indent=2))
        return

    def fmt(value):
        if value is None:
            return "N/A"
        try:
            value = float(value)
        except Exception:
            return "N/A"
        return f"{value:.6g}" if np.isfinite(value) else "N/A"

    psf = summary["psf"]
    optics = summary["optics"]
    grid = summary["grid"]
    print(f"Run: {summary['run_type'].upper()} PSF")
    print(
        "Optics: "
        f"{optics['wavelength_nm']:.3f} nm, NA={optics['NA']:.4g}, "
        f"pupil={optics['pupil_d_mm']:.4g} mm"
    )
    print(
        "Grid: "
        f"{grid['nx']} x {grid['ny']}, "
        f"{grid['size_x_mm']:.4g} x {grid['size_y_mm']:.4g} mm"
    )
    print(
        "Focus/FWHM: "
        f"z={fmt(psf['focus_z_um'])} um, "
        f"x={fmt(psf['fwhm_x_um'])} um, "
        f"y={fmt(psf['fwhm_y_um'])} um, "
        f"z={fmt(psf['fwhm_z_um'])} um"
    )
    print(
        f"Widths @{psf['width_level']:.3g}: "
        f"x={fmt(psf['width_x_um'])} um, "
        f"y={fmt(psf['width_y_um'])} um, "
        f"z={fmt(psf['width_z_um'])} um"
    )


def write_summary_json(path, summary):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_config(path, params, optimizer_settings=None):
    ensure_parent(path)
    payload = {
        "format": CONFIG_FORMAT,
        "version": CONFIG_VERSION,
        "params": encode_precise_params(params),
    }
    if optimizer_settings is not None:
        payload["optimizer_settings"] = encode_precise_params(optimizer_settings)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def command_gui(args, gui_launcher=None):
    if gui_launcher is not None:
        gui_launcher()
        return 0
    from psf_gui import launch_gui

    launch_gui()
    return 0


def command_defaults(args):
    params = resolve_params(args)
    optimizer_settings = resolve_optimizer_settings(args) if getattr(args, "include_optimizer", False) else None
    payload = {
        "format": CONFIG_FORMAT,
        "version": CONFIG_VERSION,
        "params": json_safe(params),
    }
    if optimizer_settings is not None:
        payload["optimizer_settings"] = json_safe(optimizer_settings)
    print(json.dumps(payload, indent=2))
    return 0


def command_write_config(args):
    params = resolve_params(args)
    optimizer_settings = resolve_optimizer_settings(args) if args.include_optimizer else None
    write_config(args.path, params, optimizer_settings)
    print(f"Wrote config: {args.path}")
    return 0


def command_mask_info(args):
    params = resolve_params(args)
    mask = build_input_mask(params, args.mask, mask_rotation_quadrants(args))
    info = {
        "path": args.mask,
        "shape": [int(v) for v in mask.shape],
        "min": float(np.min(mask)),
        "max": float(np.max(mask)),
        "mean": float(np.mean(mask)),
        "rotation_quadrants_ccw": int(mask_rotation_quadrants(args)),
        "offset_x_mm": float(params["mask_offset_x"]) * 1e3,
        "offset_y_mm": float(params["mask_offset_y"]) * 1e3,
    }
    print(json.dumps(info, indent=2) if args.json else f"mask shape={tuple(mask.shape)}, min={info['min']:.6g}, max={info['max']:.6g}, mean={info['mean']:.6g}")
    return 0


def command_simulate(args):
    params = resolve_params(args)
    progress = ProgressPrinter(args.quiet, "simulate")
    mask = build_input_mask(params, resolve_mask_path(args), mask_rotation_quadrants(args))
    run_3d = str(args.mode).strip().lower() == "3d"
    simulator, results_2d, results_3d = run_simulation(params, mask, run_3d=run_3d, progress=progress)

    paths = {}
    if args.output:
        save_result(args.output, params, results_2d, results_3d)
        paths["result"] = args.output
    if args.csv:
        write_export_csv(
            args.csv,
            simulator,
            results_2d,
            plot_mode=args.plot_mode,
            normalized=args.normalized,
            width_level=args.width_level,
        )
        paths["csv"] = args.csv
    if args.npz:
        write_arrays_npz(args.npz, params, results_2d, results_3d)
        paths["npz"] = args.npz
    if args.plot:
        write_plot_png(args.plot, results_2d, plot_mode=args.plot_mode, normalized=args.normalized)
        paths["plot"] = args.plot
    if args.config_out:
        write_config(args.config_out, params)
        paths["config"] = args.config_out

    summary = build_summary(
        params,
        simulator,
        results_2d,
        results_3d,
        plot_mode=args.plot_mode,
        normalized=args.normalized,
        width_level=args.width_level,
    )
    if paths:
        summary["outputs"] = paths
    if args.summary_json:
        write_summary_json(args.summary_json, summary)
    print_summary(summary, json_output=args.json)
    return 0


def command_profile(args):
    params = resolve_params(args)
    mask_path = resolve_mask_path(args)
    progress = ProgressPrinter(args.quiet, "profile")
    mask = build_input_mask(params, mask_path, mask_rotation_quadrants(args))
    _simulator, profile_result = run_profile(
        params,
        mask,
        axis=args.axis,
        z_um=args.z_um,
        width_level=args.width_level,
        progress=progress,
    )
    paths = {}
    if args.csv:
        write_profile_csv(
            args.csv,
            profile_result,
            plot_mode=args.plot_mode,
            normalized=args.normalized,
        )
        paths["csv"] = args.csv
    if args.plot:
        write_profile_plot_png(
            args.plot,
            profile_result,
            plot_mode=args.plot_mode,
            normalized=args.normalized,
        )
        paths["plot"] = args.plot

    summary = profile_summary(
        params,
        profile_result,
        plot_mode=args.plot_mode,
        normalized=args.normalized,
        mask_path=mask_path,
    )
    if paths:
        summary["outputs"] = paths
    if args.summary_json:
        write_summary_json(args.summary_json, summary)
    print_profile_summary(summary, json_output=args.json)
    return 0


def command_ring_sweep_z(args):
    params = resolve_params(args)
    axis = str(args.axis).strip().lower()
    range_key = f"range_{axis}_um"
    if getattr(args, "range_um", None) is None and getattr(args, range_key, None) is None:
        params[range_key] = 20.0
    if axis in {"x", "y"} and getattr(args, "z_um", None) is None and getattr(args, "range_z_um", None) is None:
        params["range_z_um"] = 20.0
    if getattr(args, "points", None) is None and getattr(args, "points_2d", None) is None:
        params["points_2d"] = 300
    params = validate_params(params)

    radius_grid_mm = ring_radius_grid(args.max_radius_mm, args.radius_points)
    radius_pairs_mm = ring_radius_pairs(radius_grid_mm)
    mask_count = len(radius_pairs_mm)
    output_dir = Path(args.output_dir).expanduser()
    output_stem = f"ring_mask_{axis}"
    profiles_csv = default_output_path(output_dir, args.profiles_csv, f"{output_stem}_profiles.csv")
    profiles_npz = default_output_path(output_dir, args.profiles_npz, f"{output_stem}_profiles.npz")
    summary_csv = default_output_path(output_dir, args.summary_csv, f"{output_stem}_summary.csv")
    metadata_json = default_output_path(output_dir, args.metadata_json, f"{output_stem}_metadata.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(profiles_csv)
    ensure_parent(profiles_npz)
    ensure_parent(summary_csv)
    ensure_parent(metadata_json)

    half_extent_m = 0.5 * min(float(params["size_x"]), float(params["size_y"]))
    max_radius_m = float(radius_grid_mm[-1]) * 1e-3 if radius_grid_mm.size else 0.0
    if max_radius_m > half_extent_m:
        raise ValueError("--max-radius-mm exceeds half of the smaller mask physical dimension.")

    radius_map = build_radius_map(
        (int(params["nx"]), int(params["ny"])),
        float(params["size_x"]),
        float(params["size_y"]),
    )
    progress = ProgressPrinter(args.quiet, "ring-sweep-z")
    coord_column = f"{axis}_um"
    profile_columns = [
        "mask_id",
        "inner_radius_mm",
        "outer_radius_mm",
        "profile_axis",
        coord_column,
        "intensity",
        "intensity_norm",
        "intensity_norm_squared",
    ]
    summary_columns = [
        "mask_id",
        "inner_radius_mm",
        "outer_radius_mm",
        "inner_diameter_mm",
        "outer_diameter_mm",
        "nx",
        "ny",
        "size_x_mm",
        "size_y_mm",
        "pixel_pitch_x_um",
        "pixel_pitch_y_um",
        "mask_offset_x_mm",
        "mask_offset_y_mm",
        "profile_axis",
        f"{axis}_min_um",
        f"{axis}_max_um",
        f"{axis}_points",
        f"peak_{axis}_um",
        "z_position_um",
        "peak_intensity",
        "fwhm_I_um",
        "fwhm_I2_um",
    ]

    profile_row_count = 0
    npz_mask_ids = []
    npz_inner_radius_mm = []
    npz_outer_radius_mm = []
    npz_intensity = []
    npz_intensity_norm = []
    npz_intensity_norm_squared = []
    npz_peak_coord_um = []
    npz_z_position_um = []
    npz_peak_intensity = []
    npz_fwhm_i_um = []
    npz_fwhm_i2_um = []
    npz_coords_um = None
    with open(profiles_csv, "w", newline="", encoding="utf-8") as profile_f, open(
        summary_csv, "w", newline="", encoding="utf-8"
    ) as summary_f:
        profile_writer = csv.writer(profile_f)
        summary_writer = csv.writer(summary_f)
        profile_writer.writerow(profile_columns)
        summary_writer.writerow(summary_columns)

        for idx, (inner_mm, outer_mm) in enumerate(radius_pairs_mm, start=1):
            mask_id = f"mask_{idx:04d}"
            progress.emit(
                100.0 * float(idx - 1) / max(1, mask_count),
                f"{mask_id}/{mask_count}: inner={inner_mm:.6g} mm, outer={outer_mm:.6g} mm",
                force=(idx == 1),
            )

            mask = create_ring_block_mask_from_radius_map(
                radius_map,
                inner_radius_m=inner_mm * 1e-3,
                outer_radius_m=outer_mm * 1e-3,
            )
            mask = CartesianPSFSimulator.apply_mask_offset(
                mask,
                size_x=float(params["size_x"]),
                size_y=float(params["size_y"]),
                offset_x=float(params["mask_offset_x"]),
                offset_y=float(params["mask_offset_y"]),
            )

            _simulator, profile_result = run_profile(
                params,
                mask,
                axis=axis,
                z_um=args.z_um,
                width_level=0.5,
                progress=None,
                num_points=int(params["points_2d"]),
            )
            coords_um = np.asarray(profile_result["coords_um"], dtype=float)
            intensity = np.asarray(profile_result["intensity"], dtype=float)
            intensity_norm = np.asarray(profile_result["intensity_norm"], dtype=float)
            intensity_norm_squared = intensity_norm**2
            fwhm_i_um = CartesianPSFSimulator._width_at_level_1d(coords_um, intensity_norm, level=0.5)
            fwhm_i2_um = CartesianPSFSimulator._width_at_level_1d(coords_um, intensity_norm_squared, level=0.5)
            if npz_coords_um is None:
                npz_coords_um = coords_um.copy()

            for coord_val, raw_val, norm_val, norm_sq_val in zip(
                coords_um, intensity, intensity_norm, intensity_norm_squared
            ):
                profile_writer.writerow(
                    [
                        mask_id,
                        float(inner_mm),
                        float(outer_mm),
                        axis,
                        float(coord_val),
                        float(raw_val),
                        float(norm_val),
                        float(norm_sq_val),
                    ]
                )
                profile_row_count += 1

            summary_writer.writerow(
                [
                    mask_id,
                    float(inner_mm),
                    float(outer_mm),
                    float(2.0 * inner_mm),
                    float(2.0 * outer_mm),
                    int(params["nx"]),
                    int(params["ny"]),
                    float(params["size_x"]) * 1e3,
                    float(params["size_y"]) * 1e3,
                    float(params["size_x"]) * 1e6 / max(1, int(params["nx"])),
                    float(params["size_y"]) * 1e6 / max(1, int(params["ny"])),
                    float(params["mask_offset_x"]) * 1e3,
                    float(params["mask_offset_y"]) * 1e3,
                    axis,
                    float(np.min(coords_um)) if coords_um.size else np.nan,
                    float(np.max(coords_um)) if coords_um.size else np.nan,
                    int(coords_um.size),
                    float(profile_result["peak_coord_um"]),
                    float(profile_result["z_position_um"]),
                    float(profile_result["peak_intensity"]),
                    float(fwhm_i_um),
                    float(fwhm_i2_um),
                ]
            )
            npz_mask_ids.append(mask_id)
            npz_inner_radius_mm.append(float(inner_mm))
            npz_outer_radius_mm.append(float(outer_mm))
            npz_intensity.append(intensity.astype(float, copy=True))
            npz_intensity_norm.append(intensity_norm.astype(float, copy=True))
            npz_intensity_norm_squared.append(intensity_norm_squared.astype(float, copy=True))
            npz_peak_coord_um.append(float(profile_result["peak_coord_um"]))
            npz_z_position_um.append(float(profile_result["z_position_um"]))
            npz_peak_intensity.append(float(profile_result["peak_intensity"]))
            npz_fwhm_i_um.append(float(fwhm_i_um))
            npz_fwhm_i2_um.append(float(fwhm_i2_um))
            if idx % 10 == 0:
                profile_f.flush()
                summary_f.flush()

    progress.emit(100.0, "Done", force=True)
    npz_arrays = {
        "profile_axis": np.asarray(axis),
        "coord_label": np.asarray(coord_column),
        "coord_um": np.asarray(npz_coords_um if npz_coords_um is not None else [], dtype=float),
        "mask_id": np.asarray(npz_mask_ids, dtype=str),
        "inner_radius_mm": np.asarray(npz_inner_radius_mm, dtype=float),
        "outer_radius_mm": np.asarray(npz_outer_radius_mm, dtype=float),
        "inner_diameter_mm": 2.0 * np.asarray(npz_inner_radius_mm, dtype=float),
        "outer_diameter_mm": 2.0 * np.asarray(npz_outer_radius_mm, dtype=float),
        "mask_offset_x_mm": np.full(len(npz_mask_ids), float(params["mask_offset_x"]) * 1e3, dtype=float),
        "mask_offset_y_mm": np.full(len(npz_mask_ids), float(params["mask_offset_y"]) * 1e3, dtype=float),
        "intensity": np.asarray(npz_intensity, dtype=float),
        "intensity_norm": np.asarray(npz_intensity_norm, dtype=float),
        "intensity_norm_squared": np.asarray(npz_intensity_norm_squared, dtype=float),
        "peak_coord_um": np.asarray(npz_peak_coord_um, dtype=float),
        "z_position_um": np.asarray(npz_z_position_um, dtype=float),
        "peak_intensity": np.asarray(npz_peak_intensity, dtype=float),
        "fwhm_I_um": np.asarray(npz_fwhm_i_um, dtype=float),
        "fwhm_I2_um": np.asarray(npz_fwhm_i2_um, dtype=float),
        "radius_grid_mm": np.asarray(radius_grid_mm, dtype=float),
    }
    write_ring_profiles_npz(profiles_npz, npz_arrays)
    summary = {
        "run_type": "ring-sweep-z",
        "profile_axis": axis,
        "mask_count": int(mask_count),
        "profile_row_count": int(profile_row_count),
        "radius_points": int(radius_grid_mm.size),
        "radius_grid_mm": [float(v) for v in radius_grid_mm],
        "max_radius_mm": float(radius_grid_mm[-1]) if radius_grid_mm.size else 0.0,
        "profile_range_um": float(params[range_key]),
        "profile_points": int(params["points_2d"]),
        "z_position_um": None if args.z_um is None else float(args.z_um),
        "mask_offset_x_mm": float(params["mask_offset_x"]) * 1e3,
        "mask_offset_y_mm": float(params["mask_offset_y"]) * 1e3,
        "outputs": {
            "profiles_csv": str(Path(profiles_csv).resolve()),
            "profiles_npz": str(Path(profiles_npz).resolve()),
            "summary_csv": str(Path(summary_csv).resolve()),
            "metadata_json": str(Path(metadata_json).resolve()),
        },
    }
    metadata = {
        "format": "psf_ring_sweep_profile",
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_config": getattr(args, "config", None),
        "params": json_safe(params),
        "profile_axis": axis,
        "profile_coord_column": coord_column,
        "z_position_um": None if args.z_um is None else float(args.z_um),
        "mask_offset_x_mm": float(params["mask_offset_x"]) * 1e3,
        "mask_offset_y_mm": float(params["mask_offset_y"]) * 1e3,
        "profile_npz_arrays": sorted(npz_arrays.keys()),
        "mask_definition": {
            "units": "mm",
            "transparent": "r < inner_radius or r > outer_radius",
            "blocked": "inner_radius <= r <= outer_radius when outer_radius > inner_radius; equal radii are saved as all-pass zero-width controls",
            "max_radius_mm": float(radius_grid_mm[-1]) if radius_grid_mm.size else 0.0,
            "radius_grid_mm": [float(v) for v in radius_grid_mm],
            "pair_count": int(mask_count),
        },
        "profile_columns": profile_columns,
        "summary_columns": summary_columns,
        "outputs": summary["outputs"],
    }
    write_ring_sweep_metadata(metadata_json, metadata)
    print_ring_sweep_summary(summary, json_output=args.json)
    return 0


def command_export(args):
    payload = load_result(args.input)
    params = validate_params({**DEFAULT_PARAMS, **params_from_core_values(payload["params"])})
    simulator = create_simulator_from_params(params)
    results_2d = payload["results_2d"]
    results_3d = payload.get("results_3d")

    paths = {}
    if args.csv:
        write_export_csv(
            args.csv,
            simulator,
            results_2d,
            plot_mode=args.plot_mode,
            normalized=args.normalized,
            width_level=args.width_level,
        )
        paths["csv"] = args.csv
    if args.npz:
        write_arrays_npz(args.npz, params, results_2d, results_3d)
        paths["npz"] = args.npz
    if args.plot:
        write_plot_png(args.plot, results_2d, plot_mode=args.plot_mode, normalized=args.normalized)
        paths["plot"] = args.plot
    summary = build_summary(
        params,
        simulator,
        results_2d,
        results_3d,
        plot_mode=args.plot_mode,
        normalized=args.normalized,
        width_level=args.width_level,
    )
    if paths:
        summary["outputs"] = paths
    if args.summary_json:
        write_summary_json(args.summary_json, summary)
    print_summary(summary, json_output=args.json)
    return 0


def read_npz_metadata(path):
    with np.load(path, allow_pickle=True) as data:
        if "metadata_json" not in data.files:
            return {"format": "npz", "arrays": list(data.files)}
        raw = data["metadata_json"]
        try:
            text = str(raw.item())
        except Exception:
            text = str(raw)
        return json.loads(text)


def command_inspect(args):
    suffix = Path(args.input).suffix.lower()
    if suffix in {".psfres", ".pkl", ".pickle"}:
        payload = load_result(args.input)
        params = validate_params({**DEFAULT_PARAMS, **params_from_core_values(payload["params"])})
        simulator = create_simulator_from_params(params)
        summary = build_summary(
            params,
            simulator,
            payload["results_2d"],
            payload.get("results_3d"),
            plot_mode=args.plot_mode,
            normalized=args.normalized,
            width_level=args.width_level,
        )
        if args.summary_json:
            write_summary_json(args.summary_json, summary)
        print_summary(summary, json_output=args.json)
        return 0
    if suffix == ".npz":
        metadata = read_npz_metadata(args.input)
        if args.summary_json:
            write_summary_json(args.summary_json, json_safe(metadata))
        print(json.dumps(json_safe(metadata), indent=2))
        return 0
    raise ValueError("inspect supports .psfres/.pkl result files and .npz array/optimizer files.")


def command_optimizers(args):
    if not HAS_OPTIMIZER:
        raise RuntimeError(f"optimizer backend unavailable: {OPTIMIZER_IMPORT_ERROR}")
    print(json.dumps(PostFresnelMaskOptimizer.get_optimizer_category_map(), indent=2))
    return 0


def command_optimize(args):
    if not HAS_OPTIMIZER:
        raise RuntimeError(f"optimizer backend unavailable: {OPTIMIZER_IMPORT_ERROR}")
    params = resolve_params(args)
    settings = resolve_optimizer_settings(args)
    mask = build_input_mask(params, args.mask, mask_rotation_quadrants(args))
    simulator = create_simulator_from_params(params)
    optimizer = PostFresnelMaskOptimizer(simulator=simulator, params=params)
    cfg = OptimizerConfig(
        optimizer_name=settings["optimizer_name"],
        mode=settings["mode"],
        axial_symmetry=bool(settings["axial_symmetry"]),
        binary_amplitude=bool(settings["binary_amplitude"]),
        binary_structure=str(settings["binary_structure"]),
        width_level=float(settings["width_level"]),
        weights_xyz=(float(settings["weight_x"]), float(settings["weight_y"]), float(settings["weight_z"])),
        weight_abs_xy=float(settings["weight_abs_xy"]),
        n_starts=int(settings["n_starts"]),
        maxiter=int(settings["maxiter"]),
        ftol=float(settings["ftol"]),
        gtol=float(settings["gtol"]),
        seed=settings["seed"],
    )
    progress = ProgressPrinter(args.quiet, "optimize")

    def progress_callback(stage, frac, info):
        text = str(stage)
        if isinstance(info, dict):
            parts = [text]
            obj = info.get("objective")
            if obj is not None and np.isfinite(float(obj)):
                parts.append(f"objective={float(obj):.6g}")
            iter_index = info.get("iter_index")
            maxiter = info.get("maxiter")
            if iter_index is not None:
                if maxiter is not None:
                    parts.append(f"iter={int(float(iter_index))}/{int(float(maxiter))}")
                else:
                    parts.append(f"iter={int(float(iter_index))}")
            text = " | ".join(parts)
        progress.emit(100.0 * float(frac), text)

    result = optimizer.optimize(input_intensity=mask, config=cfg, progress_callback=progress_callback)
    progress.emit(100.0, "Optimization complete", force=True)

    paths = {}
    if args.output:
        ensure_parent(args.output)
        save_result_npz(args.output, result, extra_metadata={"simulation_params": clone_payload(params)})
        paths["optimized_npz"] = args.output
    if args.amplitude_csv:
        ensure_parent(args.amplitude_csv)
        np.savetxt(args.amplitude_csv, np.asarray(result.amplitude_mask, dtype=float), delimiter=",")
        paths["amplitude_csv"] = args.amplitude_csv
    if args.phase_csv:
        ensure_parent(args.phase_csv)
        np.savetxt(args.phase_csv, np.asarray(result.phase_mask_rad, dtype=float), delimiter=",")
        paths["phase_csv"] = args.phase_csv

    summary = {
        "best_objective": float(result.best_objective),
        "baseline_objective": float(result.baseline_objective),
        "best_metrics_um": json_safe(result.best_metrics_um),
        "baseline_metrics_um": json_safe(result.baseline_metrics_um),
        "settings": json_safe(result.settings),
        "outputs": paths,
    }
    if args.summary_json:
        write_summary_json(args.summary_json, summary)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        before = result.baseline_metrics_um
        after = result.best_metrics_um
        print(f"Objective: {result.baseline_objective:.6g} -> {result.best_objective:.6g}")
        print(
            "Widths (um): "
            f"X {before['width_x_um']:.6g} -> {after['width_x_um']:.6g}, "
            f"Y {before['width_y_um']:.6g} -> {after['width_y_um']:.6g}, "
            f"Z {before['width_z_um']:.6g} -> {after['width_z_um']:.6g}"
        )
        if paths:
            for key, path in paths.items():
                print(f"{key}: {path}")
    return 0


def add_common_param_options(parser):
    parser.add_argument("--config", help="Load parameters from a CLI or GUI last-used JSON config.")
    parser.add_argument("--objective", help="Load objective lens preset JSON before applying explicit overrides.")
    parser.add_argument("--wavelength-nm", type=float, help="Excitation wavelength in nm.")
    parser.add_argument("--na", "--NA", dest="na", type=float, help="Objective numerical aperture.")
    parser.add_argument("--theta-samples", type=int, help="Theta integration samples.")
    parser.add_argument("--phi-samples", type=int, help="Phi integration samples.")
    parser.add_argument("--torch-device", choices=("auto", "cpu", "cuda"), help="Torch compute device.")
    parser.add_argument("--nx", type=int, help="Input mask grid rows.")
    parser.add_argument("--ny", type=int, help="Input mask grid columns.")
    parser.add_argument("--size-x-mm", type=float, help="Mask physical size in x, mm.")
    parser.add_argument("--size-y-mm", type=float, help="Mask physical size in y, mm.")
    parser.add_argument("--pupil-d-mm", type=float, help="Entrance pupil diameter, mm.")
    parser.add_argument("--fresnel-z-m", type=float, help="Fresnel propagation distance, m.")
    parser.add_argument("--range-um", type=float, help="Set all PSF +/- axis ranges, um.")
    parser.add_argument("--range-x-um", type=float, help="PSF +/- x range, um.")
    parser.add_argument("--range-y-um", type=float, help="PSF +/- y range, um.")
    parser.add_argument("--range-z-um", type=float, help="PSF +/- z range, um.")
    parser.add_argument("--points", type=int, help="Set both 2D and 3D point counts.")
    parser.add_argument("--points-2d", type=int, help="2D plane point count.")
    parser.add_argument("--points-3d", type=int, help="3D volume point count.")
    parser.add_argument("--z-chunk", type=int, help="3D z chunk size.")
    parser.add_argument("--n-i", type=float, help="Experimental immersion refractive index.")
    parser.add_argument("--n-g", type=float, help="Experimental coverslip refractive index.")
    parser.add_argument("--t-g-um", type=float, help="Experimental coverslip thickness, um.")
    parser.add_argument("--n-s", type=float, help="Experimental resin/sample refractive index.")
    parser.add_argument("--t-s-um", type=float, help="Resin thickness / PSF position, um.")
    parser.add_argument("--n-id", type=float, help="Design immersion refractive index.")
    parser.add_argument("--n-gd", type=float, help="Design coverslip refractive index.")
    parser.add_argument("--t-gd-um", type=float, help="Design coverslip thickness, um.")
    parser.add_argument("--n-sd", type=float, help="Design resin/sample refractive index.")
    gaussian = parser.add_mutually_exclusive_group()
    gaussian.add_argument("--gaussian", dest="use_gaussian", action="store_true", default=None, help="Apply Gaussian envelope.")
    gaussian.add_argument("--no-gaussian", dest="use_gaussian", action="store_false", help="Disable Gaussian envelope.")
    parser.add_argument("--gauss-wx-mm", type=float, help="Gaussian wx, mm.")
    parser.add_argument("--gauss-wy-mm", type=float, help="Gaussian wy, mm.")
    parser.add_argument("--mask-offset-x-mm", type=float, help="Mask x offset, mm.")
    parser.add_argument("--mask-offset-y-mm", type=float, help="Mask y offset, mm.")


def add_mask_options(parser, required=False):
    parser.add_argument("--mask", required=required, help="Mask path (.npy/.csv/.txt/.tif/.tiff/image). Omit for all-pass.")
    parser.add_argument("--mask-rotate", type=int, choices=(-270, -180, -90, 0, 90, 180, 270), help="Rotate mask counter-clockwise by degrees.")
    parser.add_argument("--mask-rotate-quadrants", type=int, help="Rotate mask by this many 90-degree counter-clockwise turns.")


def add_display_options(parser):
    parser.add_argument("--plot-mode", default="I", help="Display/export mode: I or I2.")
    norm = parser.add_mutually_exclusive_group()
    norm.add_argument("--normalized", dest="normalized", action="store_true", default=True, help="Normalize displayed/exported intensities.")
    norm.add_argument("--raw", dest="normalized", action="store_false", help="Export raw displayed intensities.")
    parser.add_argument("--width-level", type=float, default=0.5, help="Peak width level in [0.01, 0.99].")
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout.")
    parser.add_argument("--summary-json", help="Write JSON summary to this path.")


def add_optimizer_options(parser):
    parser.add_argument("--optimizer-name", help="Optimizer name, e.g. L-BFGS-B, SLSQP, Powell, Nelder-Mead.")
    parser.add_argument("--optimizer-mode", choices=("amplitude", "phase", "amplitude_phase"), help="Mask variables to optimize.")
    axial = parser.add_mutually_exclusive_group()
    axial.add_argument("--axial-symmetry", dest="axial_symmetry", action="store_true", default=None, help="Use axial/radial parameterization.")
    axial.add_argument("--no-axial-symmetry", dest="axial_symmetry", action="store_false", help="Use full 2D parameterization.")
    binary = parser.add_mutually_exclusive_group()
    binary.add_argument("--binary-amplitude", dest="binary_amplitude", action="store_true", default=None, help="Use binary amplitude latent parameterization.")
    binary.add_argument("--no-binary-amplitude", dest="binary_amplitude", action="store_false", help="Disable binary amplitude mode.")
    parser.add_argument("--binary-structure", choices=("pixel", "simple_ring_block", "simple_ellipse_block"), help="Binary structure mode.")
    parser.add_argument("--opt-width-level", type=float, help="Optimization width level.")
    parser.add_argument("--weight-x", type=float, help="Objective weight for x width.")
    parser.add_argument("--weight-y", type=float, help="Objective weight for y width.")
    parser.add_argument("--weight-z", type=float, help="Objective weight for z width.")
    parser.add_argument("--weight-abs-xy", type=float, help="Objective weight for |x-y|.")
    parser.add_argument("--n-starts", type=int, help="Multi-start count.")
    parser.add_argument("--maxiter", type=int, help="Maximum optimizer iterations.")
    parser.add_argument("--ftol", type=float, help="Optimizer ftol.")
    parser.add_argument("--gtol", type=float, help="Optimizer gtol.")
    parser.add_argument("--seed", help="Integer seed, or empty string for random.")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="psf_cli",
        description="Headless CLI for the PSF Simulator GUI app.",
    )
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="Launch the Tk GUI.")
    gui.set_defaults(func=command_gui)

    defaults = sub.add_parser("defaults", help="Print default CLI config as JSON.")
    add_common_param_options(defaults)
    defaults.add_argument("--include-optimizer", action="store_true", help="Include optimizer defaults.")
    defaults.set_defaults(func=command_defaults)

    write_config_cmd = sub.add_parser("write-config", help="Write a reusable JSON config.")
    write_config_cmd.add_argument("path", help="Output JSON path.")
    add_common_param_options(write_config_cmd)
    write_config_cmd.add_argument("--include-optimizer", action="store_true", help="Include optimizer settings.")
    add_optimizer_options(write_config_cmd)
    write_config_cmd.set_defaults(func=command_write_config)

    mask_info = sub.add_parser("mask-info", help="Load, normalize, rotate, resize, offset, and summarize a mask.")
    add_common_param_options(mask_info)
    add_mask_options(mask_info, required=True)
    mask_info.add_argument("--json", action="store_true", help="Print JSON.")
    mask_info.set_defaults(func=command_mask_info)

    simulate = sub.add_parser("simulate", help="Run a 2D or 3D PSF simulation.")
    simulate.add_argument("--mode", choices=("2d", "3d"), default="2d", help="Simulation mode.")
    add_common_param_options(simulate)
    add_mask_options(simulate)
    add_display_options(simulate)
    simulate.add_argument("--output", "-o", help="Write app-compatible .psfres result.")
    simulate.add_argument("--csv", help="Write GUI-compatible CSV export.")
    simulate.add_argument("--npz", help="Write compressed array bundle.")
    simulate.add_argument("--plot", help="Write PNG summary plot.")
    simulate.add_argument("--config-out", help="Write resolved parameters to config JSON.")
    simulate.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    simulate.set_defaults(func=command_simulate)

    profile = sub.add_parser("profile", help="Run a quick 1D PSF intensity profile.")
    profile.add_argument("--axis", choices=("x", "y", "z"), required=True, help="Profile axis.")
    add_common_param_options(profile)
    add_mask_options(profile)
    add_display_options(profile)
    profile.add_argument(
        "--z-um",
        type=float,
        help="Fixed z position for x/y profiles in um. Defaults to on-axis z peak.",
    )
    profile.add_argument("--csv", help="Write compact profile CSV.")
    profile.add_argument("--plot", help="Write profile PNG plot.")
    profile.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    profile.set_defaults(func=command_profile)

    ring_sweep = sub.add_parser("ring-sweep-z", help="Sweep annular blocking masks and write 1D x/y/z profiles.")
    add_common_param_options(ring_sweep)
    ring_sweep.add_argument("--axis", choices=("x", "y", "z"), default="z", help="1D profile axis to sweep.")
    ring_sweep.add_argument(
        "--z-um",
        type=float,
        help="Fixed z position for x/y profiles in um. Defaults to each mask's on-axis z peak.",
    )
    ring_sweep.add_argument("--output-dir", required=True, help="Output folder for CSV, NPZ, and metadata files.")
    ring_sweep.add_argument("--max-radius-mm", type=float, default=2.3, help="Maximum inner/outer radius in mm.")
    ring_sweep.add_argument("--radius-points", type=int, default=20, help="Evenly spaced radius samples from 0 to max.")
    ring_sweep.add_argument("--profiles-csv", help="Override long-form profile CSV path.")
    ring_sweep.add_argument("--profiles-npz", help="Override compact profile NPZ path.")
    ring_sweep.add_argument("--summary-csv", help="Override per-mask summary CSV path.")
    ring_sweep.add_argument("--metadata-json", help="Override metadata JSON path.")
    ring_sweep.add_argument("--json", action="store_true", help="Print JSON summary to stdout.")
    ring_sweep.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    ring_sweep.set_defaults(func=command_ring_sweep_z)

    export = sub.add_parser("export", help="Export or plot a saved .psfres/.pkl result.")
    export.add_argument("input", help="Input .psfres/.pkl file.")
    add_display_options(export)
    export.add_argument("--csv", help="Write GUI-compatible CSV export.")
    export.add_argument("--npz", help="Write compressed array bundle.")
    export.add_argument("--plot", help="Write PNG summary plot.")
    export.set_defaults(func=command_export)

    inspect = sub.add_parser("inspect", help="Inspect .psfres/.pkl or .npz outputs.")
    inspect.add_argument("input", help="Input result or npz file.")
    add_display_options(inspect)
    inspect.set_defaults(func=command_inspect)

    optimizers = sub.add_parser("optimizers", help="List available optimizer categories and names.")
    optimizers.set_defaults(func=command_optimizers)

    optimize = sub.add_parser("optimize", help="Run post-Fresnel mask optimization.")
    add_common_param_options(optimize)
    add_mask_options(optimize)
    add_optimizer_options(optimize)
    optimize.add_argument("--output", "-o", default="optimized_mask_cli.npz", help="Output optimized mask .npz.")
    optimize.add_argument("--amplitude-csv", help="Optional amplitude mask CSV output.")
    optimize.add_argument("--phase-csv", help="Optional phase mask CSV output.")
    optimize.add_argument("--summary-json", help="Write JSON summary to this path.")
    optimize.add_argument("--json", action="store_true", help="Print JSON summary to stdout.")
    optimize.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    optimize.set_defaults(func=command_optimize)

    parser.set_defaults(func=None)
    return parser


def main(argv=None, gui_launcher=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.func is None:
        parser.print_help()
        return 0
    try:
        if args.func is command_gui:
            return args.func(args, gui_launcher=gui_launcher)
        return args.func(args)
    except (ValueError, RuntimeError, SimulationStoppedError, OptimizationStoppedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
