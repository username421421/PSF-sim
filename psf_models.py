"""Typed boundary models shared by CLI, GUI, and runtime orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


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
