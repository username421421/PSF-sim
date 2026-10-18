"""Typed boundary models shared by CLI, GUI, and runtime orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite, log, sqrt
from typing import Any, Dict, Optional, Tuple

INTEGRATION_MODES = ("cartesian_jacobian", "polar_interp")
ABERRATION_MODELS = ("legacy", "gibson_lanni")
TORCH_DEVICES = ("auto", "cpu", "cuda")

# The Gaussian envelope is exp(-(x/w)^2) in intensity, so its intensity FWHM is
# 2*sqrt(ln 2)*w. Users enter the FWHM; params and the simulator keep w.
FWHM_PER_1E_RADIUS = 2.0 * sqrt(log(2.0))

# Display-unit keys written by older GUI configs: old key -> (current key, factor).
LEGACY_UI_KEYS = {
    "gauss_wx_mm": ("gauss_fwhm_x_mm", FWHM_PER_1E_RADIUS),
    "gauss_wy_mm": ("gauss_fwhm_y_mm", FWHM_PER_1E_RADIUS),
}


@dataclass(frozen=True)
class ParamSpec:
    """One simulation parameter: SI ``key`` in params/configs, ``ui_key`` in display units."""

    key: str
    ui_key: str
    cast: type
    default: Any
    to_si: float = 1.0
    from_si: float = 1.0
    choices: Tuple[str, ...] = ()
    bound: Optional[str] = None
    label: str = ""
    unit: str = ""
    cli_dest: Optional[str] = None
    # Newer fields that older configs omit; loading them resets to the default.
    reset_on_load: bool = False
    # Significant digits shown after a non-decimal unit conversion (None = exact).
    display_digits: Optional[int] = None

    @property
    def dest(self) -> str:
        return self.cli_dest or self.ui_key

    @property
    def ui_default(self) -> Any:
        if self.cast is float:
            return float(f"{self.default * self.from_si:.12g}")
        return self.default

    def ui_value(self, raw: Any) -> Any:
        if self.cast is str:
            return str(raw).strip().lower()
        return self.cast(raw)

    def si_value(self, raw: Any) -> Any:
        value = self.ui_value(raw)
        return value * self.to_si if self.cast is float else value


# Single source for parameter keys, unit conversions, defaults, and choices.
PARAM_SPECS = (
    ParamSpec("wavelength", "wavelength_nm", float, 780.0e-9, 1e-9, 1e9, label="Wavelength", unit="nm"),
    ParamSpec("NA", "NA", float, 1.4, bound="positive", label="NA", cli_dest="na"),
    ParamSpec("theta_samples", "theta_samples", int, 61),
    ParamSpec("phi_samples", "phi_samples", int, 120),
    ParamSpec("integration_mode", "integration_mode", str, "cartesian_jacobian", choices=INTEGRATION_MODES),
    ParamSpec("aberration_model", "aberration_model", str, "legacy", choices=ABERRATION_MODELS, reset_on_load=True),
    ParamSpec("torch_device", "torch_device", str, "cpu", choices=TORCH_DEVICES),
    ParamSpec("nx", "nx", int, 256),
    ParamSpec("ny", "ny", int, 256),
    ParamSpec("size_x", "size_x_mm", float, 8.0e-3, 1e-3, 1e3),
    ParamSpec("size_y", "size_y_mm", float, 8.0e-3, 1e-3, 1e3),
    ParamSpec("pupil_d", "pupil_d_mm", float, 5.6e-3, 1e-3, 1e3, bound="positive",
              label="Entrance pupil diameter", unit="mm"),
    ParamSpec("fresnel_z", "fresnel_z_m", float, 0.30),
    ParamSpec("range_x_um", "range_x_um", float, 2.0),
    ParamSpec("range_y_um", "range_y_um", float, 2.0),
    ParamSpec("range_z_um", "range_z_um", float, 2.0),
    ParamSpec("points_2d", "points_2d", int, 101),
    ParamSpec("points_3d", "points_3d", int, 61),
    ParamSpec("z_chunk", "z_chunk", int, 8),
    ParamSpec("n_i", "n_i", float, 1.515, bound="positive"),
    ParamSpec("n_g", "n_g", float, 1.515, bound="positive"),
    ParamSpec("t_g", "t_g_um", float, 170.0e-6, 1e-6, 1e6, bound="nonnegative"),
    ParamSpec("n_s", "n_s", float, 1.474, bound="positive"),
    ParamSpec("t_s", "t_s_um", float, 1.0e-6, 1e-6, 1e6, bound="nonnegative"),
    ParamSpec("n_id", "n_id", float, 1.515, bound="positive", label="Design immersion RI"),
    ParamSpec("t_id", "t_id_um", float, 90.0e-6, 1e-6, 1e6, bound="nonnegative",
              label="Design immersion gap", unit="um", reset_on_load=True),
    ParamSpec("n_gd", "n_gd", float, 1.515, bound="positive", label="Design coverslip RI"),
    ParamSpec("t_gd", "t_gd_um", float, 170.0e-6, 1e-6, 1e6, bound="nonnegative",
              label="Design coverslip thickness", unit="um"),
    ParamSpec("n_sd", "n_sd", float, 1.515, bound="positive", label="Design resin RI"),
    ParamSpec("use_gaussian", "use_gaussian", bool, False),
    # Stored as the 1/e intensity radius; entered and shown as intensity FWHM.
    ParamSpec("gauss_wx", "gauss_fwhm_x_mm", float, 2.7e-3 / FWHM_PER_1E_RADIUS,
              1e-3 / FWHM_PER_1E_RADIUS, 1e3 * FWHM_PER_1E_RADIUS, label="Gaussian FWHM x", unit="mm",
              display_digits=6),
    ParamSpec("gauss_wy", "gauss_fwhm_y_mm", float, 2.7e-3 / FWHM_PER_1E_RADIUS,
              1e-3 / FWHM_PER_1E_RADIUS, 1e3 * FWHM_PER_1E_RADIUS, label="Gaussian FWHM y", unit="mm",
              display_digits=6),
    ParamSpec("mask_offset_x", "mask_offset_x_mm", float, 0.0, 1e-3, 1e3),
    ParamSpec("mask_offset_y", "mask_offset_y_mm", float, 0.0, 1e-3, 1e3),
)
PARAM_SPECS_BY_KEY = {spec.key: spec for spec in PARAM_SPECS}

# Objective-lens presets carry the design half of the optical model.
OBJECTIVE_SPECS = tuple(
    PARAM_SPECS_BY_KEY[key] for key in ("NA", "pupil_d", "n_id", "t_id", "n_gd", "t_gd", "n_sd")
)


def upgrade_ui_values(values: Dict[str, Any]) -> Dict[str, Any]:
    """Convert display-unit keys from older GUI configs to the current ones."""
    upgraded = dict(values)
    for old_key, (new_key, factor) in LEGACY_UI_KEYS.items():
        if old_key in upgraded:
            raw = upgraded.pop(old_key)
            if new_key not in upgraded:
                upgraded[new_key] = float(raw) * factor
    return upgraded


def objective_ui_values(values: Dict[str, Any]) -> Dict[str, float]:
    """Validate an objective-lens preset in display units (NA, pupil_d_mm, n_id, t_id_um, ...)."""
    if not isinstance(values, dict):
        raise ValueError("Objective lens values must be a dictionary.")
    # Older presets lack t_id (default gap); n_sd defaults to the design immersion RI.
    missing = [
        spec.ui_key for spec in OBJECTIVE_SPECS
        if spec.ui_key not in values and spec.key not in ("t_id", "n_sd")
    ]
    if missing:
        raise ValueError(f"Objective lens preset missing keys: {', '.join(missing)}.")
    result = {}
    for spec in OBJECTIVE_SPECS:
        if spec.ui_key in values:
            raw = values[spec.ui_key]
        elif spec.key == "n_sd":
            raw = values["n_id"]
        else:
            raw = spec.ui_default
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Objective lens preset contains non-numeric values.") from exc
        positive = spec.bound == "positive"
        if not isfinite(value) or value < 0.0 or (positive and value == 0.0):
            unit = f" {spec.unit}" if spec.unit else ""
            raise ValueError(f"{spec.label} must be finite and {'>' if positive else '>='} 0{unit}.")
        result[spec.ui_key] = value
    return result


def objective_params(values: Dict[str, Any]) -> Dict[str, float]:
    """Objective-lens preset converted to SI simulation parameters."""
    ui = objective_ui_values(values)
    return {spec.key: ui[spec.ui_key] * spec.to_si for spec in OBJECTIVE_SPECS}


@dataclass(frozen=True)
class ResourceEstimate:
    peak_ram_bytes: int
    peak_vram_bytes: int
    available_ram_bytes: int
    available_vram_bytes: int
    predicted_objective_evals_per_iteration: int = 0
    suggested_z_chunk: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaskSpec:
    source: str = "all_pass"
    path: Optional[str] = None
    normalization: str = "physical"
    rotation_quadrants_ccw: int = 0
    plane: str = "pre_fresnel_intensity"

    @classmethod
    def from_mapping(cls, value: Optional[Dict[str, Any]]) -> "MaskSpec":
        data = value if isinstance(value, dict) else {}
        return cls(
            source=str(data.get("source", "file" if data.get("path") else "all_pass")),
            path=None if data.get("path") in (None, "") else str(data.get("path")),
            normalization=str(data.get("normalization", "physical")),
            rotation_quadrants_ccw=int(data.get("rotation_quadrants_ccw", 0)) % 4,
            plane=str(data.get("plane", "pre_fresnel_intensity")),
        )

    def to_mapping(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreparedPupilField:
    field: Any
    size_x: float
    size_y: float
    entrance_pupil_diameter: float
    cache_token: Optional[str] = None


@dataclass
class SimulationParams:
    values: Dict[str, Any]

    @classmethod
    def from_mapping(cls, values: Dict[str, Any]) -> "SimulationParams":
        return cls(dict(values))

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.values)


@dataclass
class OptimizerSettings:
    values: Dict[str, Any]

    @classmethod
    def from_mapping(cls, values: Dict[str, Any]) -> "OptimizerSettings":
        return cls(dict(values))

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.values)
