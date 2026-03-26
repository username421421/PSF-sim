import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
try:
    from scipy.optimize import differential_evolution
except Exception:
    differential_evolution = None
try:
    from scipy.optimize import shgo
except Exception:
    shgo = None

from PSF_simulator import AxialSymmetricPSFSimulator, CartesianPSFSimulator


class OptimizationStoppedError(RuntimeError):
    """Raised when optimization is interrupted by user request."""


@dataclass
class OptimizerConfig:
    optimizer_name: str = "L-BFGS-B"
    mode: str = "amplitude_phase"
    width_level: float = 0.50
    weights_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    weight_abs_xy: float = 0.0
    n_starts: int = 2
    maxiter: int = 30
    ftol: float = 1e-6
    gtol: float = 1e-5
    seed: Optional[int] = None
    axial_symmetry: bool = False
    binary_amplitude: bool = False
    binary_structure: str = "pixel"
    binary_anneal_alpha_start: float = 2.0
    binary_anneal_alpha_end: float = 24.0
    sidelobe_penalty_weight: float = 1000.0


@dataclass
class OptimizerResult:
    best_objective: float
    baseline_objective: float
    best_metrics_um: Dict[str, float]
    baseline_metrics_um: Dict[str, float]
    amplitude_mask: np.ndarray
    phase_mask_rad: np.ndarray
    complex_mask: np.ndarray
    history: List[Dict[str, Any]]
    settings: Dict[str, Any]
    baseline_result_2d: Dict[str, Any]
    optimized_result_2d: Dict[str, Any]


def _resize_map(arr: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if tuple(arr.shape) == tuple(out_shape):
        return arr.copy()
    tensor = torch.as_tensor(arr, dtype=torch.float64)[None, None, :, :]
    resized = F.interpolate(
        tensor,
        size=(int(out_shape[0]), int(out_shape[1])),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.detach().cpu().numpy()


class _LinearizedFull2DFastPSF:
    """
    Precompute reusable full-2D pupil-to-field factors for optimization.

    The post-Fresnel pupil field is fixed across optimization. Only the complex
    mask changes, so the focal field remains linear in the sampled complex mask
    coefficients. This helper caches the field basis once and reuses it for each
    objective evaluation.
    """

    _CPU_PHASED_OPERATOR_LIMIT_BYTES = 1536 * 1024 * 1024
    _CUDA_PHASED_OPERATOR_LIMIT_BYTES = 768 * 1024 * 1024

    def __init__(
        self,
        simulator: CartesianPSFSimulator,
        base_field_for_psf: np.ndarray,
        *,
        size_x: float,
        size_y: float,
        entrance_pupil_diameter: float,
        range_microns: float,
        range_microns_x: float,
        range_microns_y: float,
        range_microns_z: float,
        num_points: int,
        compute_xy: bool,
        compute_xz: bool,
        compute_yz: bool,
    ):
        self.simulator = simulator
        self.device = simulator.device
        self.compute_xy = bool(compute_xy)
        self.compute_xz = bool(compute_xz)
        self.compute_yz = bool(compute_yz)
        if not (self.compute_xy or self.compute_xz or self.compute_yz):
            raise ValueError("At least one PSF plane must be enabled for fast path.")

        field_t = simulator._to_torch_complex(base_field_for_psf)
        if field_t.ndim != 2:
            raise ValueError("base_field_for_psf must be 2D.")
        self.mask_shape = (int(field_t.shape[0]), int(field_t.shape[1]))

        tables = simulator._prepare_integration_tables(
            field_t,
            size_x=float(size_x),
            size_y=float(size_y),
            entrance_pupil_diameter=float(entrance_pupil_diameter),
        )
        if str(tables.get("mode", "")).strip().lower() != "cartesian_jacobian":
            raise ValueError("Linear fast path currently supports cartesian_jacobian only.")

        flat_indices = tables.get("flat_indices", None)
        if flat_indices is None:
            raise ValueError("Integration tables missing flat_indices for fast path.")
        self.flat_indices = flat_indices.to(device=self.device, dtype=torch.int64).contiguous()

        weighted_base = simulator._weighted_components(tables).to(dtype=torch.complex128).contiguous()
        self.weighted_base = weighted_base
        self._plane_tables = {"mode": "cartesian_jacobian"}

        range_x_m, range_y_m, range_z_m = simulator._resolve_axis_ranges_m(
            range_microns=range_microns,
            range_microns_x=range_microns_x,
            range_microns_y=range_microns_y,
            range_microns_z=range_microns_z,
        )
        npts = int(num_points)
        self.x_coords = torch.linspace(-range_x_m, range_x_m, npts, dtype=torch.float64, device=self.device)
        self.y_coords = torch.linspace(-range_y_m, range_y_m, npts, dtype=torch.float64, device=self.device)
        self.z_coords = torch.linspace(-range_z_m, range_z_m, npts, dtype=torch.float64, device=self.device)

        phase_terms = simulator._get_phase_terms(
            tables,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
            z_coords=self.z_coords,
        )
        self.exp_x = phase_terms["exp_x"].to(dtype=torch.complex128).contiguous()
        self.exp_y = phase_terms["exp_y"].to(dtype=torch.complex128).contiguous()
        self.axial_by_z = phase_terms["axial_by_z"].to(dtype=torch.complex128).contiguous()

        self.base_wx_t = None
        self.base_wy_t = None
        self.base_wx_center_t = None
        self.precompute_mode = "factorized"
        self.precompute_reason = "enabled: cached base field factors and phase terms"
        self.precomputed_operator_bytes = 0

        if self.compute_xy and (not self.compute_xz) and (not self.compute_yz):
            x_center = torch.zeros((1,), dtype=torch.float64, device=self.device)
            exp_x_center = simulator._precompute_exp_x(x_center, tables).to(dtype=torch.complex128)
            self.base_wx_center_t = (
                self.weighted_base[:, :, None] * exp_x_center[None, :, :]
            ).permute(0, 2, 1).contiguous()

        self._maybe_precompute_phased_operators()

        self._z_chunk_size = 8
        self._y_chunk_size = 8

    def _phase_operator_limit_bytes(self) -> int:
        if str(getattr(self.device, "type", "")).strip().lower() == "cuda":
            return int(self._CUDA_PHASED_OPERATOR_LIMIT_BYTES)
        return int(self._CPU_PHASED_OPERATOR_LIMIT_BYTES)

    def _maybe_precompute_phased_operators(self) -> None:
        n_samples = int(self.flat_indices.numel())
        nx = int(self.x_coords.numel())
        ny = int(self.y_coords.numel())
        bytes_per_complex = int(np.dtype(np.complex128).itemsize)
        need_x = bool(self.compute_xz or self.compute_xy)
        need_y = bool(self.compute_yz)

        estimated = 0
        if need_x:
            estimated += 3 * n_samples * nx * bytes_per_complex
        if need_y:
            estimated += 3 * n_samples * ny * bytes_per_complex
        self.precomputed_operator_bytes = int(estimated)

        limit = self._phase_operator_limit_bytes()
        if estimated <= 0:
            self.precompute_reason = "enabled: cached base field factors and phase terms"
            return
        if estimated > limit:
            self.precompute_reason = (
                "enabled: factorized cache only "
                f"(estimated phased operator {estimated / (1024.0 ** 2):.1f} MiB "
                f"> limit {limit / (1024.0 ** 2):.1f} MiB)"
            )
            return

        try:
            if need_x:
                self.base_wx_t = (
                    self.weighted_base[:, :, None] * self.exp_x[None, :, :]
                ).permute(0, 2, 1).contiguous()
            if need_y:
                self.base_wy_t = (
                    self.weighted_base[:, :, None] * self.exp_y[None, :, :]
                ).permute(0, 2, 1).contiguous()
            self.precompute_mode = "phased_operator"
            self.precompute_reason = (
                "enabled: precomputed phased operators "
                f"({estimated / (1024.0 ** 2):.1f} MiB)"
            )
        except Exception as exc:
            self.base_wx_t = None
            self.base_wy_t = None
            if str(getattr(self.device, "type", "")).strip().lower() == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            self.precompute_mode = "factorized"
            self.precompute_reason = (
                "enabled: factorized cache only "
                f"(phased-operator allocation fallback: {exc})"
            )

    @staticmethod
    def _check_stop(stop_callback):
        if stop_callback is not None and bool(stop_callback()):
            raise OptimizationStoppedError("Optimization stopped by user.")

    def _complex_mask_samples(self, complex_mask_map: np.ndarray) -> torch.Tensor:
        mask_t = self.simulator._to_torch_complex(complex_mask_map)
        if tuple(int(v) for v in mask_t.shape) != tuple(self.mask_shape):
            raise ValueError(
                f"Fast-path complex-mask shape mismatch: got {tuple(mask_t.shape)}, expected {self.mask_shape}."
            )
        mask_flat = mask_t.reshape(-1)
        return mask_flat[self.flat_indices].to(dtype=torch.complex128)

    def _weighted_components_for_mask(self, mask_p: torch.Tensor) -> torch.Tensor:
        return self.weighted_base * mask_p[None, :]

    def _compute_xz_from_mask(
        self,
        mask_p: torch.Tensor,
        weighted_components: Optional[torch.Tensor] = None,
        stop_callback=None,
    ) -> torch.Tensor:
        if self.base_wx_t is None:
            if weighted_components is None:
                weighted_components = self._weighted_components_for_mask(mask_p)
            return self.simulator._compute_xz_plane(
                self.x_coords,
                self.z_coords,
                self._plane_tables,
                verbose=False,
                progress_callback=None,
                stop_callback=stop_callback,
                weighted_components=weighted_components,
                exp_x=self.exp_x,
                axial_by_z=self.axial_by_z,
            )

        nz = int(self.z_coords.numel())
        nx = int(self.x_coords.numel())
        out = torch.empty((nz, nx), dtype=torch.float64, device=self.device)
        for start in range(0, nz, int(self._z_chunk_size)):
            self._check_stop(stop_callback)
            end = min(nz, start + int(self._z_chunk_size))
            rhs = self.axial_by_z[:, start:end] * mask_p[:, None]
            field_chunk = torch.matmul(self.base_wx_t, rhs).permute(0, 2, 1)
            ex = field_chunk[0]
            ey = field_chunk[1]
            ez = field_chunk[2]
            out[start:end, :] = torch.abs(ex) ** 2 + torch.abs(ey) ** 2 + torch.abs(ez) ** 2
        return out

    def _compute_yz_from_mask(
        self,
        mask_p: torch.Tensor,
        weighted_components: Optional[torch.Tensor] = None,
        stop_callback=None,
    ) -> torch.Tensor:
        if self.base_wy_t is None:
            if weighted_components is None:
                weighted_components = self._weighted_components_for_mask(mask_p)
            return self.simulator._compute_yz_plane(
                self.y_coords,
                self.z_coords,
                self._plane_tables,
                verbose=False,
                progress_callback=None,
                stop_callback=stop_callback,
                weighted_components=weighted_components,
                exp_y=self.exp_y,
                axial_by_z=self.axial_by_z,
            )

        nz = int(self.z_coords.numel())
        ny = int(self.y_coords.numel())
        out = torch.empty((nz, ny), dtype=torch.float64, device=self.device)
        for start in range(0, nz, int(self._z_chunk_size)):
            self._check_stop(stop_callback)
            end = min(nz, start + int(self._z_chunk_size))
            rhs = self.axial_by_z[:, start:end] * mask_p[:, None]
            field_chunk = torch.matmul(self.base_wy_t, rhs).permute(0, 2, 1)
            ex = field_chunk[0]
            ey = field_chunk[1]
            ez = field_chunk[2]
            out[start:end, :] = torch.abs(ex) ** 2 + torch.abs(ey) ** 2 + torch.abs(ez) ** 2
        return out

    def _compute_xy_from_mask(
        self,
        mask_p: torch.Tensor,
        z_index: int,
        weighted_components: Optional[torch.Tensor] = None,
        stop_callback=None,
    ) -> torch.Tensor:
        z_idx = int(max(0, min(int(self.z_coords.numel()) - 1, int(z_index))))
        if self.base_wx_t is None:
            if weighted_components is None:
                weighted_components = self._weighted_components_for_mask(mask_p)
            return self.simulator._compute_xy_plane(
                self.x_coords,
                self.y_coords,
                self.z_coords[z_idx],
                self._plane_tables,
                verbose=False,
                progress_callback=None,
                stop_callback=stop_callback,
                weighted_components=weighted_components,
                exp_x=self.exp_x,
                exp_y=self.exp_y,
                axial=self.axial_by_z[:, z_idx],
            )

        mask_ax = mask_p * self.axial_by_z[:, z_idx]
        ny = int(self.y_coords.numel())
        nx = int(self.x_coords.numel())
        out = torch.empty((ny, nx), dtype=torch.float64, device=self.device)
        for start in range(0, ny, int(self._y_chunk_size)):
            self._check_stop(stop_callback)
            end = min(ny, start + int(self._y_chunk_size))
            rhs = self.exp_y[:, start:end] * mask_ax[:, None]
            field_chunk = torch.matmul(self.base_wx_t, rhs).permute(0, 2, 1)
            ex = field_chunk[0]
            ey = field_chunk[1]
            ez = field_chunk[2]
            out[start:end, :] = torch.abs(ex) ** 2 + torch.abs(ey) ** 2 + torch.abs(ez) ** 2
        return out

    def _compute_xz_center_from_mask(
        self,
        mask_p: torch.Tensor,
        weighted_components: Optional[torch.Tensor] = None,
        stop_callback=None,
    ) -> torch.Tensor:
        if self.base_wx_center_t is None:
            if weighted_components is None:
                weighted_components = self._weighted_components_for_mask(mask_p)
            x_center = torch.zeros((1,), dtype=torch.float64, device=self.device)
            exp_x_center = torch.ones(
                (int(mask_p.numel()), 1), dtype=torch.complex128, device=self.device
            )
            return self.simulator._compute_xz_plane(
                x_center,
                self.z_coords,
                self._plane_tables,
                verbose=False,
                progress_callback=None,
                stop_callback=stop_callback,
                weighted_components=weighted_components,
                exp_x=exp_x_center,
                axial_by_z=self.axial_by_z,
            )[:, 0]
        self._check_stop(stop_callback)
        rhs = self.axial_by_z * mask_p[:, None]
        field = torch.matmul(self.base_wx_center_t, rhs).permute(0, 2, 1)
        ex = field[0, :, 0]
        ey = field[1, :, 0]
        ez = field[2, :, 0]
        return torch.abs(ex) ** 2 + torch.abs(ey) ** 2 + torch.abs(ez) ** 2

    def simulate_from_complex_mask_map(
        self,
        complex_mask_map: np.ndarray,
        stop_callback=None,
    ) -> Dict[str, Any]:
        res_torch = self.simulate_from_complex_mask_tensor(
            complex_mask_map,
            stop_callback=stop_callback,
        )

        i_xz = res_torch.get("I_xz", None)
        i_xy = res_torch.get("I_xy", None)
        i_yz = res_torch.get("I_yz", None)
        i_xz_norm = None if i_xz is None else self.simulator._normalize_intensity(i_xz.clone())
        i_xy_norm = None if i_xy is None else self.simulator._normalize_intensity(i_xy.clone())
        i_yz_norm = None if i_yz is None else self.simulator._normalize_intensity(i_yz.clone())

        x_coords_np = self.simulator._to_numpy(res_torch["x_coords"])
        y_coords_np = self.simulator._to_numpy(res_torch["y_coords"])
        z_coords_np = self.simulator._to_numpy(res_torch["z_coords"])
        z_focus = res_torch["z_focus"]
        z_focus_np = float(z_focus.item()) if torch.is_tensor(z_focus) else float(z_focus)

        i_xz_np = None if i_xz is None else self.simulator._to_numpy(i_xz)
        i_xy_np = None if i_xy is None else self.simulator._to_numpy(i_xy)
        i_yz_np = None if i_yz is None else self.simulator._to_numpy(i_yz)
        i_xz_norm_np = None if i_xz_norm is None else self.simulator._to_numpy(i_xz_norm)
        i_xy_norm_np = None if i_xy_norm is None else self.simulator._to_numpy(i_xy_norm)
        i_yz_norm_np = None if i_yz_norm is None else self.simulator._to_numpy(i_yz_norm)

        if (i_xy_norm_np is not None) and (i_xz_norm_np is not None):
            fwhm_metrics = self.simulator._compute_fwhm_metrics(
                x_coords=x_coords_np,
                y_coords=y_coords_np,
                z_coords=z_coords_np,
                i_xy_norm=i_xy_norm_np,
                i_xz_norm=i_xz_norm_np,
            )
        else:
            fwhm_metrics = {
                "fwhm_x_m": np.nan,
                "fwhm_y_m": np.nan,
                "fwhm_z_m": np.nan,
                "fwhm_x_um": np.nan,
                "fwhm_y_um": np.nan,
                "fwhm_z_um": np.nan,
                "x_profile_at_y0": None,
                "y_profile_at_x0": None,
                "z_profile_at_x0": None,
            }

        return {
            "x_coords": x_coords_np,
            "y_coords": y_coords_np,
            "z_coords": z_coords_np,
            "z_focus": z_focus_np,
            "I_xz": i_xz_np,
            "I_xy": i_xy_np,
            "I_yz": i_yz_np,
            "I_xz_norm": i_xz_norm_np,
            "I_xy_norm": i_xy_norm_np,
            "I_yz_norm": i_yz_norm_np,
            "fwhm_x_m": fwhm_metrics["fwhm_x_m"],
            "fwhm_y_m": fwhm_metrics["fwhm_y_m"],
            "fwhm_z_m": fwhm_metrics["fwhm_z_m"],
            "fwhm_x_um": fwhm_metrics["fwhm_x_um"],
            "fwhm_y_um": fwhm_metrics["fwhm_y_um"],
            "fwhm_z_um": fwhm_metrics["fwhm_z_um"],
            "x_profile_at_y0": fwhm_metrics["x_profile_at_y0"],
            "y_profile_at_x0": fwhm_metrics["y_profile_at_x0"],
            "z_profile_at_x0": fwhm_metrics["z_profile_at_x0"],
        }

    def simulate_from_complex_mask_tensor(
        self,
        complex_mask_map,
        stop_callback=None,
    ) -> Dict[str, Any]:
        self._check_stop(stop_callback)
        mask_p = self._complex_mask_samples(complex_mask_map)
        weighted_components = None
        if (
            self.base_wx_t is None
            or self.base_wy_t is None
            or (
                self.compute_xy
                and (not self.compute_xz)
                and (not self.compute_yz)
                and self.base_wx_center_t is None
            )
        ):
            weighted_components = self._weighted_components_for_mask(mask_p)

        i_xz = None
        i_xy = None
        i_yz = None
        z_focus = None
        max_i_z = None

        if self.compute_xz:
            i_xz = self._compute_xz_from_mask(
                mask_p,
                weighted_components=weighted_components,
                stop_callback=stop_callback,
            )
            max_idx = int(torch.argmax(i_xz).item())
            nx = int(self.x_coords.numel())
            max_i_z = max_idx // max(1, nx)
            z_focus = self.z_coords[max_i_z]

        if self.compute_yz and ((not self.compute_xy) or (z_focus is None)):
            i_yz = self._compute_yz_from_mask(
                mask_p,
                weighted_components=weighted_components,
                stop_callback=stop_callback,
            )
            if z_focus is None:
                max_idx_yz = int(torch.argmax(i_yz).item())
                ny = int(self.y_coords.numel())
                max_i_z = max_idx_yz // max(1, ny)
                z_focus = self.z_coords[max_i_z]

        if self.compute_xy:
            if z_focus is None:
                i_z_center = self._compute_xz_center_from_mask(
                    mask_p,
                    weighted_components=weighted_components,
                    stop_callback=stop_callback,
                )
                max_i_z = int(torch.argmax(i_z_center).item())
                z_focus = self.z_coords[max_i_z]
            if max_i_z is None:
                max_i_z = int(torch.argmin(torch.abs(self.z_coords - z_focus)).item())
            i_xy = self._compute_xy_from_mask(
                mask_p,
                z_index=max_i_z,
                weighted_components=weighted_components,
                stop_callback=stop_callback,
            )

        if self.compute_yz and (i_yz is None):
            i_yz = self._compute_yz_from_mask(
                mask_p,
                weighted_components=weighted_components,
                stop_callback=stop_callback,
            )
            if z_focus is None:
                max_idx_yz = int(torch.argmax(i_yz).item())
                ny = int(self.y_coords.numel())
                max_i_z = max_idx_yz // max(1, ny)
                z_focus = self.z_coords[max_i_z]
        if z_focus is None:
            z_focus = self.z_coords[int(len(self.z_coords) // 2)]

        return {
            "x_coords": self.x_coords,
            "y_coords": self.y_coords,
            "z_coords": self.z_coords,
            "z_focus": z_focus,
            "I_xz": i_xz,
            "I_xy": i_xy,
            "I_yz": i_yz,
        }

    def simulate_surrogate_planes_from_complex_mask_tensor(
        self,
        complex_mask_map,
        *,
        need_xz: bool,
        need_yz: bool,
        stop_callback=None,
    ) -> Dict[str, Any]:
        self._check_stop(stop_callback)
        mask_p = self._complex_mask_samples(complex_mask_map)
        weighted_components = None
        if self.base_wx_t is None or self.base_wy_t is None:
            weighted_components = self._weighted_components_for_mask(mask_p)

        i_xz = None
        i_yz = None
        if bool(need_xz):
            i_xz = self._compute_xz_from_mask(
                mask_p,
                weighted_components=weighted_components,
                stop_callback=stop_callback,
            )
        if bool(need_yz):
            i_yz = self._compute_yz_from_mask(
                mask_p,
                weighted_components=weighted_components,
                stop_callback=stop_callback,
            )
        return {
            "x_coords": self.x_coords,
            "y_coords": self.y_coords,
            "z_coords": self.z_coords,
            "I_xz": i_xz,
            "I_yz": i_yz,
        }


class _AxialSymmetricRadialFastPSF:
    """
    Precompute theta-only axial-symmetry operators for radial-mask optimization.
    """

    def __init__(
        self,
        simulator: CartesianPSFSimulator,
        base_field_for_psf: np.ndarray,
        *,
        size_x: float,
        size_y: float,
        entrance_pupil_diameter: float,
        range_microns: float,
        range_microns_x: float,
        range_microns_y: float,
        range_microns_z: float,
        num_points: int,
        compute_xy: bool,
        compute_xz: bool,
        compute_yz: bool,
    ):
        self.compute_xy = bool(compute_xy)
        self.compute_xz = bool(compute_xz)
        self.compute_yz = bool(compute_yz)
        if not (self.compute_xy or self.compute_xz or self.compute_yz):
            raise ValueError("At least one PSF plane must be enabled for axial fast path.")

        device_str = str(getattr(simulator, "device", "cpu"))
        self.axial_simulator = AxialSymmetricPSFSimulator(
            wavelength=float(simulator.wavelength),
            NA=float(simulator.NA),
            theta_min=float(simulator.theta_min),
            theta_samples=int(simulator.theta_samples),
            phi_samples=int(simulator.phi_samples),
            t_gd=float(simulator.t_gd),
            n_gd=float(simulator.n_gd),
            t_id=float(simulator.t_id),
            n_id=float(simulator.n_id),
            n_sd=float(simulator.n_sd),
            t_g=float(simulator.t_g),
            n_g=float(simulator.n_g),
            t_s=float(simulator.t_s),
            n_s=float(simulator.n_s),
            n_i=float(simulator.n_i),
            integration_mode=str(getattr(simulator, "integration_mode", "cartesian_jacobian")),
            torch_device=device_str,
            trim_pupil_for_psf=bool(getattr(simulator, "trim_pupil_for_psf", True)),
            pupil_trim_margin_pixels=int(getattr(simulator, "pupil_trim_margin_pixels", 2)),
        )

        field_t = self.axial_simulator._to_torch_complex(base_field_for_psf)
        if field_t.ndim != 2:
            raise ValueError("base_field_for_psf must be 2D.")

        self.precomputed = self.axial_simulator.prepare_axial_symmetric_precomputed(
            pupil_field=field_t,
            size_x=float(size_x),
            size_y=float(size_y),
            entrance_pupil_diameter=float(entrance_pupil_diameter),
            range_microns=float(range_microns),
            range_microns_x=float(range_microns_x),
            range_microns_y=float(range_microns_y),
            range_microns_z=float(range_microns_z),
            num_points=int(num_points),
        )
        self.radial_norm_axis = np.asarray(
            self.axial_simulator._to_numpy(self.precomputed["radial_norm_axis"]),
            dtype=np.float64,
        )

    @staticmethod
    def _check_stop(stop_callback):
        if stop_callback is not None and bool(stop_callback()):
            raise OptimizationStoppedError("Optimization stopped by user.")

    @staticmethod
    def _resample_profile_np(profile: np.ndarray, out_len: int) -> np.ndarray:
        arr = np.asarray(profile, dtype=np.float64).ravel()
        out_len = int(out_len)
        if out_len < 1:
            raise ValueError("out_len must be >= 1.")
        if int(arr.size) == out_len:
            return arr.copy()
        if int(arr.size) < 2:
            return np.full((out_len,), float(arr[0]) if arr.size > 0 else 0.0, dtype=np.float64)
        src = np.linspace(0.0, 1.0, int(arr.size), dtype=np.float64)
        dst = np.linspace(0.0, 1.0, out_len, dtype=np.float64)
        return np.interp(dst, src, arr).astype(np.float64)

    def simulate_from_profiles(
        self,
        amplitude_profile: np.ndarray,
        phase_profile: Optional[np.ndarray] = None,
        stop_callback=None,
    ) -> Dict[str, Any]:
        self._check_stop(stop_callback)
        n_theta = int(self.radial_norm_axis.size)
        amp = np.clip(
            self._resample_profile_np(np.asarray(amplitude_profile, dtype=np.float64), n_theta),
            0.0,
            1.0,
        )
        if phase_profile is None:
            ph = np.zeros((n_theta,), dtype=np.float64)
        else:
            ph = np.clip(
                self._resample_profile_np(np.asarray(phase_profile, dtype=np.float64), n_theta),
                -np.pi,
                np.pi,
            )
        return self.axial_simulator.simulate_psf_planes_from_radial_profiles(
            radial_amplitude_profile=amp,
            radial_phase_profile=ph,
            precomputed=self.precomputed,
            stop_callback=stop_callback,
            compute_xy=bool(self.compute_xy),
            compute_xz=bool(self.compute_xz),
            compute_yz=bool(self.compute_yz),
        )


class PostFresnelMaskOptimizer:
    """
    Optimize a complex mask applied after Fresnel propagation and before
    entrance pupil masking.
    """

    _VALID_MODES = {"amplitude", "phase", "amplitude_phase"}
    _OPTIMIZER_CATEGORIES = {
        "Local (gradient-based)": ("L-BFGS-B", "SLSQP"),
        "Local (derivative-free)": ("Powell", "Nelder-Mead"),
        "Global": tuple(
            name
            for name, fn in (
                ("Differential Evolution", differential_evolution),
                ("SHGO", shgo),
            )
            if fn is not None
        ),
    }
    _VALID_OPTIMIZERS = {
        str(name)
        for names in _OPTIMIZER_CATEGORIES.values()
        for name in names
    }
    _VALID_BINARY_STRUCTURES = {"pixel", "simple_ring_block", "simple_ellipse_block"}
    def __init__(self, simulator: CartesianPSFSimulator, params: Dict[str, Any]):
        self.simulator = simulator
        self.params = dict(params)
        self.history: List[Dict[str, Any]] = []
        self._pupil_trim_window = None
        self._size_x_for_psf = float(self.params.get("size_x", 1.0))
        self._size_y_for_psf = float(self.params.get("size_y", 1.0))
        self._need_x = True
        self._need_y = True
        self._need_z = True
        self._compute_xy = True
        self._compute_xz = True
        self._compute_yz = True
        self._radial_bins = max(8, int(getattr(self.simulator, "theta_samples", 81)))
        self._linear_fastpath = None
        self._linear_fastpath_enabled = False
        self._linear_fastpath_reason = "disabled"
        self._surrogate_loss_used = False
        self._surrogate_loss_reason = "disabled"
        self._surrogate_optimizer_name = None
        self._axial_radial_fastpath = None
        self._axial_radial_fastpath_enabled = False
        self._axial_radial_fastpath_reason = "disabled"

    @staticmethod
    def _check_stop(stop_callback):
        if stop_callback is not None and bool(stop_callback()):
            raise OptimizationStoppedError("Optimization stopped by user.")

    @classmethod
    def get_optimizer_category_map(cls) -> Dict[str, Tuple[str, ...]]:
        return {
            str(category): tuple(str(name) for name in names)
            for category, names in cls._OPTIMIZER_CATEGORIES.items()
            if len(tuple(names)) > 0
        }

    @classmethod
    def get_optimizer_category_for_name(cls, optimizer_name: str) -> Optional[str]:
        name = str(optimizer_name).strip()
        for category, names in cls.get_optimizer_category_map().items():
            if name in names:
                return str(category)
        return None

    @staticmethod
    def _sanitize_config(config: OptimizerConfig, full_shape: Tuple[int, int]) -> OptimizerConfig:
        mode = str(config.mode).strip().lower()
        if mode not in PostFresnelMaskOptimizer._VALID_MODES:
            raise ValueError("mode must be one of: amplitude, phase, amplitude_phase.")

        optimizer_name = str(config.optimizer_name).strip()
        if optimizer_name not in PostFresnelMaskOptimizer._VALID_OPTIMIZERS:
            valid_txt = ", ".join(
                sorted(str(v) for v in PostFresnelMaskOptimizer._VALID_OPTIMIZERS)
            )
            raise ValueError(f"optimizer_name must be one of: {valid_txt}.")

        level = float(config.width_level)
        if not np.isfinite(level) or level < 0.01 or level > 0.99:
            raise ValueError("width_level must be in [0.01, 0.99].")

        if int(config.n_starts) < 1:
            raise ValueError("n_starts must be >= 1.")
        if int(config.maxiter) < 1:
            raise ValueError("maxiter must be >= 1.")

        ftol = float(config.ftol)
        gtol = float(config.gtol)
        if (not np.isfinite(ftol)) or ftol < 0:
            raise ValueError("ftol must be finite and >= 0.")
        if (not np.isfinite(gtol)) or gtol < 0:
            raise ValueError("gtol must be finite and >= 0.")

        wx, wy, wz = tuple(float(v) for v in config.weights_xyz)
        if any((not np.isfinite(v)) or v < 0 for v in (wx, wy, wz)):
            raise ValueError("weights_xyz must be finite non-negative values.")
        weight_abs_xy = float(getattr(config, "weight_abs_xy", 0.0))
        if (not np.isfinite(weight_abs_xy)) or weight_abs_xy < 0:
            raise ValueError("weight_abs_xy must be finite and >= 0.")
        if (wx + wy + wz + weight_abs_xy) <= 0:
            raise ValueError("At least one objective weight must be > 0.")

        axial_symmetry = bool(config.axial_symmetry)
        binary_amplitude = bool(config.binary_amplitude)
        binary_structure = str(getattr(config, "binary_structure", "pixel")).strip().lower()
        if binary_structure not in PostFresnelMaskOptimizer._VALID_BINARY_STRUCTURES:
            raise ValueError(
                "binary_structure must be one of: pixel, simple_ring_block, simple_ellipse_block."
            )
        if mode != "amplitude":
            # Binary pass/block is meaningful for amplitude-only mode.
            binary_amplitude = False
            binary_structure = "pixel"
        if not binary_amplitude:
            binary_structure = "pixel"
        if binary_structure == "simple_ring_block":
            axial_symmetry = True
        elif binary_structure == "simple_ellipse_block":
            axial_symmetry = False
        alpha_start = float(config.binary_anneal_alpha_start)
        alpha_end = float(config.binary_anneal_alpha_end)
        if (not np.isfinite(alpha_start)) or alpha_start <= 0:
            raise ValueError("binary_anneal_alpha_start must be finite and > 0.")
        if (not np.isfinite(alpha_end)) or alpha_end <= 0:
            raise ValueError("binary_anneal_alpha_end must be finite and > 0.")
        if alpha_end < alpha_start:
            alpha_end = alpha_start
        sidelobe_penalty_weight = float(config.sidelobe_penalty_weight)
        if (not np.isfinite(sidelobe_penalty_weight)) or sidelobe_penalty_weight < 0:
            raise ValueError("sidelobe_penalty_weight must be finite and >= 0.")

        return OptimizerConfig(
            optimizer_name=optimizer_name,
            mode=mode,
            width_level=level,
            weights_xyz=(wx, wy, wz),
            weight_abs_xy=weight_abs_xy,
            n_starts=int(config.n_starts),
            maxiter=int(config.maxiter),
            ftol=ftol,
            gtol=gtol,
            seed=None if config.seed is None else int(config.seed),
            axial_symmetry=axial_symmetry,
            binary_amplitude=binary_amplitude,
            binary_structure=binary_structure,
            binary_anneal_alpha_start=alpha_start,
            binary_anneal_alpha_end=alpha_end,
            sidelobe_penalty_weight=sidelobe_penalty_weight,
        )

    @staticmethod
    def _pack_maps(amplitude: np.ndarray, phase: np.ndarray, mode: str) -> np.ndarray:
        if mode == "amplitude":
            return np.asarray(amplitude, dtype=np.float64).ravel()
        if mode == "phase":
            return np.asarray(phase, dtype=np.float64).ravel()
        if mode == "amplitude_phase":
            amp_flat = np.asarray(amplitude, dtype=np.float64).ravel()
            phase_flat = np.asarray(phase, dtype=np.float64).ravel()
            return np.concatenate((amp_flat, phase_flat), axis=0)
        raise ValueError("Unsupported mode.")

    @staticmethod
    def _unpack_vector(
        x: np.ndarray, mode: str, var_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = int(var_shape[0] * var_shape[1])
        arr = np.asarray(x, dtype=np.float64).ravel()
        if mode == "amplitude":
            if arr.size != n:
                raise ValueError("Invalid vector length for amplitude mode.")
            amp = arr.reshape(var_shape)
            phase = np.zeros(var_shape, dtype=np.float64)
            return amp, phase
        if mode == "phase":
            if arr.size != n:
                raise ValueError("Invalid vector length for phase mode.")
            amp = np.ones(var_shape, dtype=np.float64)
            phase = arr.reshape(var_shape)
            return amp, phase
        if mode == "amplitude_phase":
            if arr.size != 2 * n:
                raise ValueError("Invalid vector length for amplitude_phase mode.")
            amp = arr[:n].reshape(var_shape)
            phase = arr[n:].reshape(var_shape)
            return amp, phase
        raise ValueError("Unsupported mode.")

    @staticmethod
    def _vector_bounds(mode: str, var_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        n = int(var_shape[0] * var_shape[1])
        if mode == "amplitude":
            return [(0.0, 1.0)] * n
        if mode == "phase":
            return [(-np.pi, np.pi)] * n
        if mode == "amplitude_phase":
            return ([(0.0, 1.0)] * n) + ([(-np.pi, np.pi)] * n)
        raise ValueError("Unsupported mode.")

    @staticmethod
    def _pupil_support_for_shape(
        pupil_mask: np.ndarray,
        var_shape: Tuple[int, int],
        threshold: float = 0.5,
    ) -> np.ndarray:
        mask = np.asarray(pupil_mask, dtype=np.float64)
        if tuple(mask.shape) != tuple(var_shape):
            mask = _resize_map(mask, var_shape)
        return np.asarray(mask, dtype=np.float64) > float(threshold)

    @staticmethod
    def _apply_fixed_outside_pupil_bounds(
        bounds: List[Tuple[float, float]],
        mode: str,
        support_mask: np.ndarray,
        binary_amplitude: bool = False,
    ) -> List[Tuple[float, float]]:
        out = list(bounds)
        support_flat = np.asarray(support_mask, dtype=bool).ravel()
        n = int(support_flat.size)

        if mode == "amplitude":
            fixed_amp = 12.0 if bool(binary_amplitude) else 1.0
            for idx in range(n):
                if not bool(support_flat[idx]):
                    out[idx] = (float(fixed_amp), float(fixed_amp))
            return out

        if mode == "phase":
            for idx in range(n):
                if not bool(support_flat[idx]):
                    out[idx] = (0.0, 0.0)
            return out

        if mode == "amplitude_phase":
            for idx in range(n):
                if not bool(support_flat[idx]):
                    out[idx] = (1.0, 1.0)
                    out[n + idx] = (0.0, 0.0)
            return out

        raise ValueError("Unsupported mode.")

    @staticmethod
    def _clip_vector_to_bounds(x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).copy()
        lo = np.asarray([b[0] for b in bounds], dtype=np.float64)
        hi = np.asarray([b[1] for b in bounds], dtype=np.float64)
        return np.clip(arr, lo, hi)

    @staticmethod
    def _init_vector(
        rng: np.random.Generator,
        start_idx: int,
        mode: str,
        var_shape: Tuple[int, int],
        binary_amplitude: bool = False,
        x_seed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = int(var_shape[0] * var_shape[1])
        if x_seed is not None and int(start_idx) == 0:
            return np.asarray(x_seed, dtype=np.float64).copy()

        if mode == "amplitude":
            if bool(binary_amplitude):
                if int(start_idx) == 0:
                    return np.full(n, 6.0, dtype=np.float64)
                return rng.uniform(-2.0, 2.0, size=n).astype(np.float64)
            if int(start_idx) == 0:
                return np.ones(n, dtype=np.float64)
            return rng.uniform(0.05, 0.95, size=n).astype(np.float64)

        if mode == "phase":
            if int(start_idx) == 0:
                return np.zeros(n, dtype=np.float64)
            return rng.uniform(-np.pi, np.pi, size=n).astype(np.float64)

        if mode == "amplitude_phase":
            if int(start_idx) == 0:
                amp = np.ones(n, dtype=np.float64)
                phase = np.zeros(n, dtype=np.float64)
                return np.concatenate((amp, phase), axis=0)
            amp = rng.uniform(0.05, 0.95, size=n).astype(np.float64)
            phase = rng.uniform(-np.pi, np.pi, size=n).astype(np.float64)
            return np.concatenate((amp, phase), axis=0)

        raise ValueError("Unsupported mode.")

    @staticmethod
    def _downsample_preview(arr: np.ndarray, max_dim: int = 120) -> np.ndarray:
        data = np.asarray(arr, dtype=np.float64)
        if data.ndim != 2:
            return np.asarray(data, dtype=np.float32)

        nx, ny = int(data.shape[0]), int(data.shape[1])
        max_dim = max(8, int(max_dim))
        if nx <= max_dim and ny <= max_dim:
            return np.asarray(data, dtype=np.float32)

        ix = np.linspace(0, max(nx - 1, 0), min(nx, max_dim)).astype(int)
        iy = np.linspace(0, max(ny - 1, 0), min(ny, max_dim)).astype(int)
        return np.asarray(data[np.ix_(ix, iy)], dtype=np.float32)

    @staticmethod
    def _estimate_evals_per_iteration(optimizer_name: str, n_variables: int) -> float:
        n = max(1, int(n_variables))
        name = str(optimizer_name).strip()
        if name in {"L-BFGS-B", "SLSQP"}:
            # SciPy finite-difference gradient uses roughly n+1 objective calls / iteration.
            return float(n + 1)
        if name == "Powell":
            # Powell is derivative-free; line searches still scale with dimensionality.
            return float(max(16, 2 * n))
        if name == "Nelder-Mead":
            # Simplex methods require multiple objective calls per iteration.
            return float(max(20, 2 * n))
        if name == "Differential Evolution":
            popsize = 8
            return float(max(20, popsize * n))
        if name == "SHGO":
            samples = max(24, 4 * n)
            return float(max(20, samples))
        return float(max(1, n))

    @staticmethod
    def _active_axis_flags(
        weights_xyz: Tuple[float, float, float],
        weight_abs_xy: float = 0.0,
        eps: float = 1e-15,
    ) -> Tuple[bool, bool, bool]:
        wx, wy, wz = (float(weights_xyz[0]), float(weights_xyz[1]), float(weights_xyz[2]))
        wxy = float(weight_abs_xy)
        need_xy_match = bool(np.isfinite(wxy) and (wxy > eps))
        return (bool((wx > eps) or need_xy_match), bool((wy > eps) or need_xy_match), bool(wz > eps))

    @staticmethod
    def _required_plane_flags(need_x: bool, need_y: bool, need_z: bool) -> Tuple[bool, bool, bool]:
        # Returns (compute_xy, compute_xz, compute_yz) with minimal planes.
        if need_z:
            if need_x and (not need_y):
                return False, True, False
            if need_y and (not need_x):
                return False, False, True
            if need_x and need_y:
                return True, True, False
            # z-only
            return False, True, False
        # no z requirement
        if need_x and need_y:
            return True, False, False
        if need_x:
            return False, True, False
        if need_y:
            return False, False, True
        # Should not happen with validated non-zero weights; keep safe fallback.
        return True, True, True

    @staticmethod
    def _sidelobe_excess_from_profile(profile: np.ndarray, level: float) -> float:
        p = np.asarray(profile, dtype=np.float64).ravel()
        if p.size < 3 or (not np.any(np.isfinite(p))):
            return np.nan
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        peak_idx = int(np.argmax(p))
        peak_val = float(p[peak_idx])
        if (not np.isfinite(peak_val)) or peak_val <= 0:
            return 0.0

        thr = float(min(1.0, max(0.0, level)))
        # Main lobe = contiguous supra-threshold region around global peak.
        left = int(peak_idx)
        right = int(peak_idx)
        while left - 1 >= 0 and float(p[left - 1]) >= thr:
            left -= 1
        while right + 1 < p.size and float(p[right + 1]) >= thr:
            right += 1

        outside = np.ones((p.size,), dtype=bool)
        outside[left : right + 1] = False
        if not np.any(outside):
            return 0.0
        sidelobe_max = float(np.max(p[outside]))
        if not np.isfinite(sidelobe_max):
            return np.nan
        return float(max(0.0, sidelobe_max - thr))

    def _compute_width_metrics_from_planes(
        self,
        res: Dict[str, Any],
        need_x: bool,
        need_y: bool,
        need_z: bool,
        level: float,
    ) -> Dict[str, float]:
        x_coords = np.asarray(res["x_coords"], dtype=np.float64).ravel()
        y_coords = np.asarray(res["y_coords"], dtype=np.float64).ravel()
        z_coords = np.asarray(res["z_coords"], dtype=np.float64).ravel()
        i_xy = res.get("I_xy_norm", None)
        i_xz = res.get("I_xz_norm", None)
        i_yz = res.get("I_yz_norm", None)
        if i_xy is not None:
            i_xy = np.asarray(i_xy, dtype=np.float64)
        if i_xz is not None:
            i_xz = np.asarray(i_xz, dtype=np.float64)
        if i_yz is not None:
            i_yz = np.asarray(i_yz, dtype=np.float64)

        peak_x_idx = int(max(0, min(len(x_coords) - 1, len(x_coords) // 2)))
        peak_y_idx = int(max(0, min(len(y_coords) - 1, len(y_coords) // 2)))
        peak_z_idx = int(max(0, min(len(z_coords) - 1, len(z_coords) // 2)))

        if i_xy is not None:
            py, px = np.unravel_index(int(np.argmax(i_xy)), i_xy.shape)
            peak_y_idx = int(py)
            peak_x_idx = int(px)
        if i_xz is not None:
            pz_xz, px_xz = np.unravel_index(int(np.argmax(i_xz)), i_xz.shape)
            peak_z_idx = int(pz_xz)
            if i_xy is None:
                peak_x_idx = int(px_xz)
        if i_yz is not None:
            pz_yz, py_yz = np.unravel_index(int(np.argmax(i_yz)), i_yz.shape)
            if i_xz is None:
                peak_z_idx = int(pz_yz)
            if i_xy is None:
                peak_y_idx = int(py_yz)

        width_x_m = np.nan
        width_y_m = np.nan
        width_z_m = np.nan

        if need_x:
            if i_xy is not None:
                x_profile = np.asarray(i_xy[peak_y_idx, :], dtype=np.float64)
            elif i_xz is not None:
                z_idx = int(max(0, min(i_xz.shape[0] - 1, peak_z_idx)))
                x_profile = np.asarray(i_xz[z_idx, :], dtype=np.float64)
            else:
                x_profile = None
            if x_profile is not None:
                width_x_m = float(self.simulator._width_at_level_1d(x_coords, x_profile, level=level))
                sidelobe_excess_x = float(self._sidelobe_excess_from_profile(x_profile, level=level))
            else:
                sidelobe_excess_x = np.nan
        else:
            sidelobe_excess_x = np.nan

        if need_y:
            if i_xy is not None:
                y_profile = np.asarray(i_xy[:, peak_x_idx], dtype=np.float64)
            elif i_yz is not None:
                z_idx = int(max(0, min(i_yz.shape[0] - 1, peak_z_idx)))
                y_profile = np.asarray(i_yz[z_idx, :], dtype=np.float64)
            else:
                y_profile = None
            if y_profile is not None:
                width_y_m = float(self.simulator._width_at_level_1d(y_coords, y_profile, level=level))
                sidelobe_excess_y = float(self._sidelobe_excess_from_profile(y_profile, level=level))
            else:
                sidelobe_excess_y = np.nan
        else:
            sidelobe_excess_y = np.nan

        if need_z:
            if i_xz is not None:
                x_idx = int(max(0, min(i_xz.shape[1] - 1, peak_x_idx)))
                z_profile = np.asarray(i_xz[:, x_idx], dtype=np.float64)
            elif i_yz is not None:
                y_idx = int(max(0, min(i_yz.shape[1] - 1, peak_y_idx)))
                z_profile = np.asarray(i_yz[:, y_idx], dtype=np.float64)
            else:
                z_profile = None
            if z_profile is not None:
                width_z_m = float(self.simulator._width_at_level_1d(z_coords, z_profile, level=level))
                sidelobe_excess_z = float(self._sidelobe_excess_from_profile(z_profile, level=level))
            else:
                sidelobe_excess_z = np.nan
        else:
            sidelobe_excess_z = np.nan

        penalty_sum = 0.0
        penalty_finite = True
        for active, excess in (
            (bool(need_x), float(sidelobe_excess_x) if np.isfinite(sidelobe_excess_x) else np.nan),
            (bool(need_y), float(sidelobe_excess_y) if np.isfinite(sidelobe_excess_y) else np.nan),
            (bool(need_z), float(sidelobe_excess_z) if np.isfinite(sidelobe_excess_z) else np.nan),
        ):
            if not active:
                continue
            if not np.isfinite(excess):
                penalty_finite = False
                break
            penalty_sum += float(excess * excess)

        return {
            "width_x_m": float(width_x_m),
            "width_y_m": float(width_y_m),
            "width_z_m": float(width_z_m),
            "width_x_um": float(width_x_m * 1e6) if np.isfinite(width_x_m) else np.nan,
            "width_y_um": float(width_y_m * 1e6) if np.isfinite(width_y_m) else np.nan,
            "width_z_um": float(width_z_m * 1e6) if np.isfinite(width_z_m) else np.nan,
            "sidelobe_excess_x": float(sidelobe_excess_x),
            "sidelobe_excess_y": float(sidelobe_excess_y),
            "sidelobe_excess_z": float(sidelobe_excess_z),
            "sidelobe_penalty": float(penalty_sum) if penalty_finite else np.nan,
        }

    @staticmethod
    def _smooth_max_torch(values: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
        v = torch.reshape(torch.as_tensor(values, dtype=torch.float64), (-1,))
        if int(v.numel()) <= 0:
            return torch.zeros((), dtype=torch.float64, device=v.device)
        t = max(1e-6, float(tau))
        vmax = torch.max(v)
        return vmax + t * torch.log(torch.sum(torch.exp((v - vmax) / t)))

    @classmethod
    def _soft_plane_profiles_torch(
        cls,
        plane: torch.Tensor,
        tau_peak: float = 0.02,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = torch.as_tensor(plane, dtype=torch.float64)
        peak = cls._smooth_max_torch(p, tau=tau_peak)
        p_norm = p / torch.clamp(peak, min=1e-12)
        weights = torch.softmax(torch.reshape(p_norm, (-1,)) / max(1e-6, float(tau_peak)), dim=0)
        weights = torch.reshape(weights, p_norm.shape)
        row_weights = torch.sum(weights, dim=1)
        col_weights = torch.sum(weights, dim=0)
        horizontal = torch.sum(row_weights[:, None] * p_norm, dim=0)
        vertical = torch.sum(p_norm * col_weights[None, :], dim=1)
        return p_norm, horizontal, vertical

    @classmethod
    def _surrogate_profile_metrics_torch(
        cls,
        coords: torch.Tensor,
        profile: torch.Tensor,
        *,
        level: float,
        tau_peak: float = 0.02,
        tau_width: float = 0.02,
        tau_excess: float = 0.02,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.reshape(torch.as_tensor(coords, dtype=torch.float64), (-1,))
        p = torch.reshape(torch.as_tensor(profile, dtype=torch.float64), (-1,))
        if int(x.numel()) != int(p.numel()) or int(x.numel()) < 2:
            zero = torch.zeros((), dtype=torch.float64, device=p.device)
            return zero, zero

        peak = cls._smooth_max_torch(p, tau=tau_peak)
        p_norm = p / torch.clamp(peak, min=1e-12)
        thr = float(min(0.99, max(0.01, float(level))))
        occ = torch.sigmoid((p_norm - thr) / max(1e-6, float(tau_width)))
        dx = torch.abs(x[1:] - x[:-1])
        occ_mid = 0.5 * (occ[1:] + occ[:-1])
        width = torch.sum(dx * occ_mid)

        excess = max(1e-6, float(tau_excess)) * F.softplus(
            (p_norm - thr) / max(1e-6, float(tau_excess))
        )
        outside = 1.0 - occ
        penalty = torch.sum(outside * excess * excess) / torch.clamp(torch.sum(outside), min=1e-12)
        return width, penalty

    def _surrogate_metrics_from_planes_torch(
        self,
        res_torch: Dict[str, Any],
        *,
        level: float,
    ) -> Dict[str, torch.Tensor]:
        tau_plane = 0.02
        tau_profile = 0.02
        tau_width = 0.02
        tau_excess = 0.02

        i_xy = res_torch.get("I_xy", None)
        i_xz = res_torch.get("I_xz", None)
        i_yz = res_torch.get("I_yz", None)
        x_coords = torch.as_tensor(res_torch["x_coords"], dtype=torch.float64)
        y_coords = torch.as_tensor(res_torch["y_coords"], dtype=torch.float64)
        z_coords = torch.as_tensor(res_torch["z_coords"], dtype=torch.float64)
        device = x_coords.device
        zero = torch.zeros((), dtype=torch.float64, device=device)

        x_profile_xy = None
        y_profile_xy = None
        x_profile_xz = None
        z_profile_xz = None
        y_profile_yz = None
        z_profile_yz = None

        if i_xy is not None:
            _, x_profile_xy, y_profile_xy = self._soft_plane_profiles_torch(i_xy, tau_peak=tau_plane)
        if i_xz is not None:
            _, x_profile_xz, z_profile_xz = self._soft_plane_profiles_torch(i_xz, tau_peak=tau_plane)
        if i_yz is not None:
            _, y_profile_yz, z_profile_yz = self._soft_plane_profiles_torch(i_yz, tau_peak=tau_plane)

        width_x_m = zero
        width_y_m = zero
        width_z_m = zero
        penalty_x = zero
        penalty_y = zero
        penalty_z = zero

        if bool(self._need_x):
            profile_x = x_profile_xy if x_profile_xy is not None else x_profile_xz
            width_x_m, penalty_x = self._surrogate_profile_metrics_torch(
                x_coords,
                profile_x if profile_x is not None else torch.zeros_like(x_coords),
                level=level,
                tau_peak=tau_profile,
                tau_width=tau_width,
                tau_excess=tau_excess,
            )
        if bool(self._need_y):
            profile_y = y_profile_xy if y_profile_xy is not None else y_profile_yz
            width_y_m, penalty_y = self._surrogate_profile_metrics_torch(
                y_coords,
                profile_y if profile_y is not None else torch.zeros_like(y_coords),
                level=level,
                tau_peak=tau_profile,
                tau_width=tau_width,
                tau_excess=tau_excess,
            )
        if bool(self._need_z):
            if z_profile_xz is not None and z_profile_yz is not None:
                profile_z = 0.5 * (z_profile_xz + z_profile_yz)
            else:
                profile_z = z_profile_xz if z_profile_xz is not None else z_profile_yz
            width_z_m, penalty_z = self._surrogate_profile_metrics_torch(
                z_coords,
                profile_z if profile_z is not None else torch.zeros_like(z_coords),
                level=level,
                tau_peak=tau_profile,
                tau_width=tau_width,
                tau_excess=tau_excess,
            )

        abs_x_minus_y_m = torch.sqrt((width_x_m - width_y_m) * (width_x_m - width_y_m) + 1e-24)
        penalty_sum = zero
        if bool(self._need_x):
            penalty_sum = penalty_sum + penalty_x
        if bool(self._need_y):
            penalty_sum = penalty_sum + penalty_y
        if bool(self._need_z):
            penalty_sum = penalty_sum + penalty_z

        return {
            "width_x_m": width_x_m,
            "width_y_m": width_y_m,
            "width_z_m": width_z_m,
            "width_x_um": width_x_m * 1e6,
            "width_y_um": width_y_m * 1e6,
            "width_z_um": width_z_m * 1e6,
            "abs_x_minus_y_um": abs_x_minus_y_m * 1e6,
            "sidelobe_penalty": penalty_sum,
        }

    def _surrogate_objective_from_torch_res(
        self,
        res_torch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        metrics = self._surrogate_metrics_from_planes_torch(
            res_torch,
            level=float(self._config.width_level),
        )
        wx, wy, wz = self._config.weights_xyz
        w_abs_xy = float(getattr(self._config, "weight_abs_xy", 0.0))
        device = metrics["width_x_um"].device
        objective = torch.zeros((), dtype=torch.float64, device=device)
        if bool(self._need_x):
            objective = objective + float(wx) * metrics["width_x_um"]
        if bool(self._need_y):
            objective = objective + float(wy) * metrics["width_y_um"]
        if bool(self._need_z):
            objective = objective + float(wz) * metrics["width_z_um"]
        if w_abs_xy > 0.0:
            objective = objective + float(w_abs_xy) * metrics["abs_x_minus_y_um"]
        objective = objective + float(self._config.sidelobe_penalty_weight) * metrics["sidelobe_penalty"]
        return objective, metrics

    @staticmethod
    def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        arr = np.clip(arr, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-arr))

    @staticmethod
    def _logit_stable(p: np.ndarray) -> np.ndarray:
        eps = 1e-6
        arr = np.asarray(p, dtype=np.float64)
        arr = np.clip(arr, eps, 1.0 - eps)
        return np.log(arr / (1.0 - arr))

    @staticmethod
    def _anneal_alpha(
        eval_in_start: int,
        est_evals_per_iter: float,
        maxiter: int,
        alpha_start: float,
        alpha_end: float,
    ) -> float:
        a0 = float(alpha_start)
        a1 = float(alpha_end)
        if a1 <= a0:
            return a0
        denom = max(1.0, float(est_evals_per_iter) * max(1.0, float(maxiter)))
        frac = min(1.0, max(0.0, float(eval_in_start) / denom))
        return float(a0 + (a1 - a0) * frac)

    @staticmethod
    def _default_radial_bin_count(var_shape: Tuple[int, int]) -> int:
        min_dim = max(1, min(int(var_shape[0]), int(var_shape[1])))
        bins = max(16, int(min_dim // 2))
        return int(min(512, max(1, bins)))

    @staticmethod
    def _build_radial_parameterization(
        var_shape: Tuple[int, int], support_mask: np.ndarray, n_bins: Optional[int] = None
    ) -> Dict[str, Any]:
        nx, ny = int(var_shape[0]), int(var_shape[1])
        support = np.asarray(support_mask, dtype=bool)
        if tuple(support.shape) != (nx, ny):
            raise ValueError("support_mask shape mismatch for radial parameterization.")

        row = np.arange(nx, dtype=np.float64)[:, None]
        col = np.arange(ny, dtype=np.float64)[None, :]
        cx = 0.5 * (float(nx) - 1.0)
        cy = 0.5 * (float(ny) - 1.0)
        radius = np.sqrt((row - cx) ** 2 + (col - cy) ** 2)

        if int(np.count_nonzero(support)) > 0:
            r_max = float(np.max(radius[support]))
        else:
            r_max = float(np.max(radius)) if radius.size > 0 else 1.0
        if (not np.isfinite(r_max)) or r_max <= 0:
            r_norm = np.zeros((nx, ny), dtype=np.float64)
        else:
            r_norm = radius / r_max

        if n_bins is None:
            n_bins = PostFresnelMaskOptimizer._default_radial_bin_count(var_shape)
        n_bins = int(max(2, n_bins))
        scale = float(max(1, n_bins - 1))
        bin_map = np.floor(r_norm * scale + 1e-12).astype(np.int32)
        np.clip(bin_map, 0, int(n_bins - 1), out=bin_map)

        if int(np.count_nonzero(support)) > 0:
            counts = np.bincount(bin_map[support].ravel(), minlength=n_bins)
            support_by_bin = np.asarray(counts > 0, dtype=bool)
        else:
            support_by_bin = np.zeros((n_bins,), dtype=bool)

        return {
            "bin_map": bin_map,
            "support_by_bin": support_by_bin,
            "n_bins": int(n_bins),
        }

    @staticmethod
    def _radial_profile_to_map(profile: np.ndarray, bin_map: np.ndarray) -> np.ndarray:
        prof = np.asarray(profile, dtype=np.float64).ravel()
        if prof.size < 1:
            raise ValueError("Radial profile must contain at least one value.")
        idx = np.asarray(bin_map, dtype=np.int64)
        idx = np.clip(idx, 0, int(prof.size - 1))
        return np.asarray(prof[idx], dtype=np.float64)

    @staticmethod
    def _map_to_radial_profile(
        arr: np.ndarray,
        bin_map: np.ndarray,
        support_mask: np.ndarray,
        n_bins: int,
        default_value: float,
    ) -> np.ndarray:
        data = np.asarray(arr, dtype=np.float64)
        idx = np.asarray(bin_map, dtype=np.int64)
        support = np.asarray(support_mask, dtype=bool)
        profile = np.full((int(n_bins),), float(default_value), dtype=np.float64)
        if int(np.count_nonzero(support)) <= 0:
            return profile
        flat_idx = idx[support].ravel()
        flat_val = data[support].ravel()
        sums = np.bincount(flat_idx, weights=flat_val, minlength=int(n_bins))
        counts = np.bincount(flat_idx, minlength=int(n_bins))
        valid = counts > 0
        profile[valid] = sums[valid] / counts[valid]
        return profile

    @staticmethod
    def _simple_ring_block_profile(
        radial_axis_norm: np.ndarray,
        inner_radius: float,
        outer_radius: float,
        support_bins: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float]:
        r = np.asarray(radial_axis_norm, dtype=np.float64).ravel()
        if r.size < 2:
            raise ValueError("radial_axis_norm must contain at least 2 samples.")
        rin = float(np.clip(min(inner_radius, outer_radius), 0.0, 1.0))
        rout = float(np.clip(max(inner_radius, outer_radius), 0.0, 1.0))
        amp = np.ones_like(r, dtype=np.float64)
        block = (r >= rin) & (r <= rout)
        amp[block] = 0.0
        if support_bins is not None:
            sb = np.asarray(support_bins, dtype=bool).ravel()
            if sb.size == amp.size:
                amp[~sb] = 1.0
        return amp, rin, rout

    @staticmethod
    def _simple_ellipse_ring_block_map(
        var_shape: Tuple[int, int],
        inner_diameter_x: float,
        inner_diameter_y: float,
        outer_diameter_x: float,
        outer_diameter_y: float,
        support_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float, float, float]:
        nx, ny = int(var_shape[0]), int(var_shape[1])
        row = np.arange(nx, dtype=np.float64)[:, None]
        col = np.arange(ny, dtype=np.float64)[None, :]
        cy = 0.5 * (float(nx) - 1.0)
        cx = 0.5 * (float(ny) - 1.0)

        support = None if support_mask is None else np.asarray(support_mask, dtype=bool)
        if support is not None and tuple(support.shape) != (nx, ny):
            raise ValueError("support_mask shape mismatch for simple ellipse ring block.")

        dx = np.abs(np.broadcast_to(col - cx, (nx, ny)))
        dy = np.abs(np.broadcast_to(row - cy, (nx, ny)))
        if support is not None and int(np.count_nonzero(support)) > 0:
            max_dx = float(np.max(dx[support]))
            max_dy = float(np.max(dy[support]))
        else:
            max_dx = float(np.max(dx)) if dx.size > 0 else 1.0
            max_dy = float(np.max(dy)) if dy.size > 0 else 1.0

        dinx = float(np.clip(inner_diameter_x, 0.0, 1.0))
        diny = float(np.clip(inner_diameter_y, 0.0, 1.0))
        doutx = float(np.clip(outer_diameter_x, 0.0, 1.0))
        douty = float(np.clip(outer_diameter_y, 0.0, 1.0))
        dinx, doutx = float(min(dinx, doutx)), float(max(dinx, doutx))
        diny, douty = float(min(diny, douty)), float(max(diny, douty))

        rx_in = float(dinx * max(max_dx, 1e-12))
        ry_in = float(diny * max(max_dy, 1e-12))
        rx_out = float(doutx * max(max_dx, 1e-12))
        ry_out = float(douty * max(max_dy, 1e-12))

        amp = np.ones((nx, ny), dtype=np.float64)
        if rx_out > 1e-12 and ry_out > 1e-12:
            outer = ((dx / rx_out) ** 2 + (dy / ry_out) ** 2) <= 1.0
            inner = np.zeros((nx, ny), dtype=bool)
            if rx_in > 1e-12 and ry_in > 1e-12:
                inner = ((dx / rx_in) ** 2 + (dy / ry_in) ** 2) <= 1.0
            ellipse = outer & (~inner)
            if support is not None:
                ellipse &= support
            amp[ellipse] = 0.0
        if support is not None:
            amp[~support] = 1.0
        return amp, dinx, diny, doutx, douty

    @staticmethod
    def _pack_stage_maps(
        amplitude: np.ndarray,
        phase: np.ndarray,
        mode: str,
        binary_amplitude: bool = False,
    ) -> np.ndarray:
        amp = np.asarray(amplitude, dtype=np.float64)
        ph = np.asarray(phase, dtype=np.float64)
        if mode == "amplitude":
            if bool(binary_amplitude):
                return PostFresnelMaskOptimizer._logit_stable(amp).ravel()
            return amp.ravel()
        if mode == "phase":
            return ph.ravel()
        if mode == "amplitude_phase":
            return np.concatenate((amp.ravel(), ph.ravel()), axis=0)
        raise ValueError("Unsupported mode.")

    @staticmethod
    def _pack_stage_profiles(
        amplitude_profile: np.ndarray,
        phase_profile: np.ndarray,
        mode: str,
        binary_amplitude: bool = False,
    ) -> np.ndarray:
        amp = np.asarray(amplitude_profile, dtype=np.float64).ravel()
        ph = np.asarray(phase_profile, dtype=np.float64).ravel()
        if mode == "amplitude":
            if bool(binary_amplitude):
                return PostFresnelMaskOptimizer._logit_stable(amp)
            return amp
        if mode == "phase":
            return ph
        if mode == "amplitude_phase":
            return np.concatenate((amp, ph), axis=0)
        raise ValueError("Unsupported mode.")

    @staticmethod
    def _unpack_stage_maps(
        x: np.ndarray,
        mode: str,
        var_shape: Tuple[int, int],
        binary_amplitude: bool = False,
        anneal_alpha: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(x, dtype=np.float64).ravel()
        n = int(var_shape[0] * var_shape[1])
        if mode == "amplitude":
            if arr.size != n:
                raise ValueError("Invalid vector length for amplitude mode.")
            if bool(binary_amplitude):
                latent = arr.reshape(var_shape)
                amp = PostFresnelMaskOptimizer._sigmoid_stable(float(anneal_alpha) * latent)
            else:
                amp = arr.reshape(var_shape)
            phase = np.zeros(var_shape, dtype=np.float64)
            return np.asarray(amp, dtype=np.float64), phase
        return PostFresnelMaskOptimizer._unpack_vector(arr, mode, var_shape)

    @staticmethod
    def _unpack_stage_profiles(
        x: np.ndarray,
        mode: str,
        n_bins: int,
        binary_amplitude: bool = False,
        anneal_alpha: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(x, dtype=np.float64).ravel()
        n = int(n_bins)
        if mode == "amplitude":
            if arr.size != n:
                raise ValueError("Invalid vector length for amplitude mode.")
            if bool(binary_amplitude):
                amp = PostFresnelMaskOptimizer._sigmoid_stable(float(anneal_alpha) * arr)
            else:
                amp = arr.copy()
            phase = np.zeros((n,), dtype=np.float64)
            return np.asarray(amp, dtype=np.float64), phase
        if mode == "phase":
            if arr.size != n:
                raise ValueError("Invalid vector length for phase mode.")
            return np.ones((n,), dtype=np.float64), arr.copy()
        if mode == "amplitude_phase":
            if arr.size != 2 * n:
                raise ValueError("Invalid vector length for amplitude_phase mode.")
            return arr[:n].copy(), arr[n:].copy()
        raise ValueError("Unsupported mode.")

    @staticmethod
    def _vector_bounds_for_stage(
        mode: str,
        var_shape: Tuple[int, int],
        binary_amplitude: bool = False,
    ) -> List[Tuple[float, float]]:
        if mode == "amplitude" and bool(binary_amplitude):
            n = int(var_shape[0] * var_shape[1])
            return [(-12.0, 12.0)] * n
        return PostFresnelMaskOptimizer._vector_bounds(mode, var_shape)

    def _seed_vector_from_full_maps(
        self,
        amp_full: np.ndarray,
        phase_full: np.ndarray,
        mode: str,
        var_shape: Tuple[int, int],
        pupil_mask: np.ndarray,
        axial_symmetry: bool,
        binary_amplitude: bool,
    ) -> np.ndarray:
        support_mask = self._pupil_support_for_shape(
            pupil_mask=pupil_mask,
            var_shape=var_shape,
            threshold=0.5,
        )
        amp_var = np.clip(_resize_map(np.asarray(amp_full, dtype=np.float64), var_shape), 0.0, 1.0)
        phase_var = np.clip(
            _resize_map(np.asarray(phase_full, dtype=np.float64), var_shape), -np.pi, np.pi
        )
        amp_var[~support_mask] = 1.0
        phase_var[~support_mask] = 0.0

        if bool(axial_symmetry):
            radial = self._build_radial_parameterization(
                var_shape,
                support_mask=support_mask,
                n_bins=int(self._radial_bins),
            )
            n_bins = int(radial["n_bins"])
            amp_profile = self._map_to_radial_profile(
                amp_var,
                bin_map=radial["bin_map"],
                support_mask=support_mask,
                n_bins=n_bins,
                default_value=1.0,
            )
            phase_profile = self._map_to_radial_profile(
                phase_var,
                bin_map=radial["bin_map"],
                support_mask=support_mask,
                n_bins=n_bins,
                default_value=0.0,
            )
            amp_profile = np.clip(amp_profile, 0.0, 1.0)
            phase_profile = np.clip(phase_profile, -np.pi, np.pi)
            support_bins = np.asarray(radial["support_by_bin"], dtype=bool)
            amp_profile[~support_bins] = 1.0
            phase_profile[~support_bins] = 0.0
            return self._pack_stage_profiles(
                amp_profile,
                phase_profile,
                mode=mode,
                binary_amplitude=bool(binary_amplitude),
            )

        return self._pack_stage_maps(
            amp_var,
            phase_var,
            mode=mode,
            binary_amplitude=bool(binary_amplitude),
        )

    def _initialize_axial_radial_fastpath(
        self,
        base_field_after_fresnel: np.ndarray,
        pupil_mask: np.ndarray,
        stop_callback=None,
    ) -> None:
        self._axial_radial_fastpath = None
        self._axial_radial_fastpath_enabled = False
        self._axial_radial_fastpath_reason = "disabled"

        self._check_stop(stop_callback)
        try:
            base = np.asarray(base_field_after_fresnel, dtype=np.complex128)
            pupil = np.asarray(pupil_mask, dtype=np.float64)
            base_field_at_pupil = base * pupil

            window = self._pupil_trim_window
            if (
                isinstance(window, (list, tuple))
                and len(window) == 4
                and int(window[1]) > int(window[0])
                and int(window[3]) > int(window[2])
            ):
                row0, row1, col0, col1 = (
                    int(window[0]),
                    int(window[1]),
                    int(window[2]),
                    int(window[3]),
                )
                base_field_for_psf = base_field_at_pupil[row0:row1, col0:col1]
            else:
                base_field_for_psf = base_field_at_pupil

            self._axial_radial_fastpath = _AxialSymmetricRadialFastPSF(
                simulator=self.simulator,
                base_field_for_psf=base_field_for_psf,
                size_x=float(self._size_x_for_psf),
                size_y=float(self._size_y_for_psf),
                entrance_pupil_diameter=float(self.params["pupil_d"]),
                range_microns=float(self.params.get("range_um", self.params.get("range_z_um", 2.0))),
                range_microns_x=float(self.params.get("range_x_um", self.params.get("range_um", 2.0))),
                range_microns_y=float(self.params.get("range_y_um", self.params.get("range_um", 2.0))),
                range_microns_z=float(self.params.get("range_z_um", self.params.get("range_um", 2.0))),
                num_points=int(self.params["points_2d"]),
                compute_xy=bool(self._compute_xy),
                compute_xz=bool(self._compute_xz),
                compute_yz=bool(self._compute_yz),
            )
            self._radial_bins = int(self._axial_radial_fastpath.radial_norm_axis.size)
            self._axial_radial_fastpath_enabled = True
            self._axial_radial_fastpath_reason = "enabled"
        except Exception as exc:
            self._axial_radial_fastpath = None
            self._axial_radial_fastpath_enabled = False
            self._axial_radial_fastpath_reason = f"fallback: {exc}"

    def _initialize_linear_fastpath(
        self,
        base_field_after_fresnel: np.ndarray,
        pupil_mask: np.ndarray,
        stop_callback=None,
    ) -> None:
        self._linear_fastpath = None
        self._linear_fastpath_enabled = False
        self._linear_fastpath_reason = "disabled"

        if str(getattr(self.simulator, "integration_mode", "")).strip().lower() != "cartesian_jacobian":
            self._linear_fastpath_reason = (
                "disabled: integration_mode is not cartesian_jacobian"
            )
            return

        self._check_stop(stop_callback)
        try:
            base = np.asarray(base_field_after_fresnel, dtype=np.complex128)
            pupil = np.asarray(pupil_mask, dtype=np.float64)
            base_field_at_pupil = base * pupil

            window = self._pupil_trim_window
            if (
                isinstance(window, (list, tuple))
                and len(window) == 4
                and int(window[1]) > int(window[0])
                and int(window[3]) > int(window[2])
            ):
                row0, row1, col0, col1 = (
                    int(window[0]),
                    int(window[1]),
                    int(window[2]),
                    int(window[3]),
                )
                base_field_for_psf = base_field_at_pupil[row0:row1, col0:col1]
            else:
                base_field_for_psf = base_field_at_pupil

            self._linear_fastpath = _LinearizedFull2DFastPSF(
                simulator=self.simulator,
                base_field_for_psf=base_field_for_psf,
                size_x=float(self._size_x_for_psf),
                size_y=float(self._size_y_for_psf),
                entrance_pupil_diameter=float(self.params["pupil_d"]),
                range_microns=float(self.params.get("range_um", self.params.get("range_z_um", 2.0))),
                range_microns_x=float(self.params.get("range_x_um", self.params.get("range_um", 2.0))),
                range_microns_y=float(self.params.get("range_y_um", self.params.get("range_um", 2.0))),
                range_microns_z=float(self.params.get("range_z_um", self.params.get("range_um", 2.0))),
                num_points=int(self.params["points_2d"]),
                compute_xy=bool(self._compute_xy),
                compute_xz=bool(self._compute_xz),
                compute_yz=bool(self._compute_yz),
            )
            self._linear_fastpath_enabled = True
            self._linear_fastpath_reason = str(self._linear_fastpath.precompute_reason)
        except Exception as exc:
            self._linear_fastpath = None
            self._linear_fastpath_enabled = False
            self._linear_fastpath_reason = f"fallback: {exc}"

    def _simulate_from_mask_maps(
        self,
        base_field_after_fresnel: np.ndarray,
        pupil_mask: np.ndarray,
        amp_var: np.ndarray,
        phase_var: np.ndarray,
        full_shape: Tuple[int, int],
        amp_profile: Optional[np.ndarray] = None,
        phase_profile: Optional[np.ndarray] = None,
        radial: Optional[Dict[str, Any]] = None,
        support_mask: Optional[np.ndarray] = None,
        axial_radial_fastpath_ok: bool = True,
        stop_callback=None,
    ) -> Tuple[float, Dict[str, float], Dict[str, Any], np.ndarray, np.ndarray]:
        self._check_stop(stop_callback)

        amp_full = _resize_map(amp_var, full_shape)
        amp_full = np.clip(amp_full, 0.0, 1.0)
        phase_full = _resize_map(phase_var, full_shape)
        phase_full = np.clip(phase_full, -np.pi, np.pi)

        use_axial_radial_fastpath = bool(
            bool(axial_radial_fastpath_ok)
            and self._axial_radial_fastpath_enabled
            and (self._axial_radial_fastpath is not None)
        )
        use_linear_fastpath = bool(
            self._linear_fastpath_enabled
            and (self._linear_fastpath is not None)
        )

        if use_axial_radial_fastpath:
            if amp_profile is None:
                if radial is not None and support_mask is not None:
                    amp_profile = self._map_to_radial_profile(
                        amp_full,
                        bin_map=radial["bin_map"],
                        support_mask=support_mask,
                        n_bins=int(radial["n_bins"]),
                        default_value=1.0,
                    )
                else:
                    amp_profile = np.ones(
                        (int(self._radial_bins),), dtype=np.float64
                    )
            if phase_profile is None:
                if radial is not None and support_mask is not None:
                    phase_profile = self._map_to_radial_profile(
                        phase_full,
                        bin_map=radial["bin_map"],
                        support_mask=support_mask,
                        n_bins=int(radial["n_bins"]),
                        default_value=0.0,
                    )
                else:
                    phase_profile = np.zeros(
                        (int(self._radial_bins),), dtype=np.float64
                    )
            res = self._axial_radial_fastpath.simulate_from_profiles(
                amplitude_profile=np.asarray(amp_profile, dtype=np.float64),
                phase_profile=np.asarray(phase_profile, dtype=np.float64),
                stop_callback=stop_callback,
            )
        elif use_linear_fastpath:
            complex_mask = amp_full * np.exp(1j * phase_full)
            window = self._pupil_trim_window
            if (
                isinstance(window, (list, tuple))
                and len(window) == 4
                and int(window[1]) > int(window[0])
                and int(window[3]) > int(window[2])
            ):
                row0, row1, col0, col1 = (
                    int(window[0]),
                    int(window[1]),
                    int(window[2]),
                    int(window[3]),
                )
                complex_mask_for_psf = np.asarray(
                    complex_mask[row0:row1, col0:col1], dtype=np.complex128
                )
            else:
                complex_mask_for_psf = np.asarray(complex_mask, dtype=np.complex128)
            res = self._linear_fastpath.simulate_from_complex_mask_map(
                complex_mask_for_psf,
                stop_callback=stop_callback,
            )
        else:
            complex_mask = amp_full * np.exp(1j * phase_full)
            field_at_pupil = base_field_after_fresnel * complex_mask * pupil_mask
            window = self._pupil_trim_window
            if (
                isinstance(window, (list, tuple))
                and len(window) == 4
                and int(window[1]) > int(window[0])
                and int(window[3]) > int(window[2])
            ):
                row0, row1, col0, col1 = (
                    int(window[0]),
                    int(window[1]),
                    int(window[2]),
                    int(window[3]),
                )
                field_for_psf = field_at_pupil[row0:row1, col0:col1]
            else:
                field_for_psf = field_at_pupil

            res = self.simulator.simulate_psf_planes_from_pupil_field(
                pupil_field=field_for_psf,
                size_x=float(self._size_x_for_psf),
                size_y=float(self._size_y_for_psf),
                entrance_pupil_diameter=float(self.params["pupil_d"]),
                range_microns=float(self.params.get("range_um", self.params.get("range_z_um", 2.0))),
                range_microns_x=float(self.params.get("range_x_um", self.params.get("range_um", 2.0))),
                range_microns_y=float(self.params.get("range_y_um", self.params.get("range_um", 2.0))),
                range_microns_z=float(self.params.get("range_z_um", self.params.get("range_um", 2.0))),
                num_points=int(self.params["points_2d"]),
                verbose=False,
                stop_callback=stop_callback,
                compute_xy=bool(self._compute_xy),
                compute_xz=bool(self._compute_xz),
                compute_yz=bool(self._compute_yz),
            )

        metrics = self._compute_width_metrics_from_planes(
            res=res,
            need_x=bool(self._need_x),
            need_y=bool(self._need_y),
            need_z=bool(self._need_z),
            level=float(self._config.width_level),
        )
        width_x_um = float(metrics["width_x_um"])
        width_y_um = float(metrics["width_y_um"])
        width_z_um = float(metrics["width_z_um"])
        abs_x_minus_y_um = (
            float(abs(width_x_um - width_y_um))
            if (np.isfinite(width_x_um) and np.isfinite(width_y_um))
            else np.nan
        )
        sidelobe_penalty = float(metrics.get("sidelobe_penalty", np.nan))

        wx, wy, wz = self._config.weights_xyz
        w_abs_xy = float(getattr(self._config, "weight_abs_xy", 0.0))
        finite_ok = True
        if bool(self._need_x) and (not np.isfinite(width_x_um)):
            finite_ok = False
        if bool(self._need_y) and (not np.isfinite(width_y_um)):
            finite_ok = False
        if bool(self._need_z) and (not np.isfinite(width_z_um)):
            finite_ok = False
        if (w_abs_xy > 0.0) and (not np.isfinite(abs_x_minus_y_um)):
            finite_ok = False
        if not np.isfinite(sidelobe_penalty):
            finite_ok = False

        if not finite_ok:
            objective = 1e12
        else:
            objective = 0.0
            if bool(self._need_x):
                objective += float(wx) * float(width_x_um)
            if bool(self._need_y):
                objective += float(wy) * float(width_y_um)
            if bool(self._need_z):
                objective += float(wz) * float(width_z_um)
            if w_abs_xy > 0.0:
                objective += float(w_abs_xy) * float(abs_x_minus_y_um)
            objective += float(self._config.sidelobe_penalty_weight) * float(sidelobe_penalty)
            objective = float(objective)

        metric_summary = {
            "width_x_um": width_x_um,
            "width_y_um": width_y_um,
            "width_z_um": width_z_um,
            "abs_x_minus_y_um": float(abs_x_minus_y_um),
            "sidelobe_excess_x": float(metrics.get("sidelobe_excess_x", np.nan)),
            "sidelobe_excess_y": float(metrics.get("sidelobe_excess_y", np.nan)),
            "sidelobe_excess_z": float(metrics.get("sidelobe_excess_z", np.nan)),
            "sidelobe_penalty": float(sidelobe_penalty),
        }
        return objective, metric_summary, res, amp_full, phase_full

    def _run_surrogate_gradient_stage(
        self,
        *,
        stage_name: str,
        base_field_after_fresnel: np.ndarray,
        pupil_mask: np.ndarray,
        optimizer_name: str,
        support_mask: np.ndarray,
        full_shape: Tuple[int, int],
        n_starts: int,
        maxiter: int,
        ftol: float,
        gtol: float,
        seed: Optional[int],
        x_seed: Optional[np.ndarray],
        progress_callback=None,
        stop_callback=None,
    ) -> Tuple[np.ndarray, float, Dict[str, float], Dict[str, Any], np.ndarray, np.ndarray]:
        if self._linear_fastpath is None or (not self._linear_fastpath_enabled):
            raise RuntimeError("Surrogate stage requires the full-2D linear fast path.")

        rng = np.random.default_rng(seed)
        support = np.asarray(support_mask, dtype=bool)
        active_flat = np.flatnonzero(support.ravel())
        if int(active_flat.size) <= 0:
            raise RuntimeError("Surrogate stage found no active in-pupil pixels to optimize.")

        device = self.simulator.device
        active_idx_t = torch.as_tensor(active_flat, dtype=torch.int64, device=device)
        total_pixels = int(full_shape[0] * full_shape[1])
        window = self._pupil_trim_window
        if (
            isinstance(window, (list, tuple))
            and len(window) == 4
            and int(window[1]) > int(window[0])
            and int(window[3]) > int(window[2])
        ):
            row0, row1, col0, col1 = (
                int(window[0]),
                int(window[1]),
                int(window[2]),
                int(window[3]),
            )
        else:
            row0, row1, col0, col1 = 0, int(full_shape[0]), 0, int(full_shape[1])

        surrogate_optimizer_name = (
            str(optimizer_name).strip()
            if str(optimizer_name).strip() in {"L-BFGS-B", "SLSQP"}
            else "L-BFGS-B"
        )
        self._surrogate_loss_used = True
        self._surrogate_optimizer_name = surrogate_optimizer_name
        if surrogate_optimizer_name == str(optimizer_name).strip():
            self._surrogate_loss_reason = (
                "enabled: smooth width surrogate with analytic gradient on the full-2D linear fast path"
            )
        else:
            self._surrogate_loss_reason = (
                "enabled: smooth width surrogate with analytic gradient on the full-2D linear fast path "
                f"(optimizer overridden from {str(optimizer_name).strip()} to {surrogate_optimizer_name})"
            )

        x_seed_arr = None
        if x_seed is not None:
            try:
                seed_arr = np.asarray(x_seed, dtype=np.float64).ravel()
                if int(seed_arr.size) == int(total_pixels):
                    x_seed_arr = np.clip(seed_arr[active_flat], 0.0, 1.0)
                elif int(seed_arr.size) == int(active_flat.size):
                    x_seed_arr = np.clip(seed_arr, 0.0, 1.0)
            except Exception:
                x_seed_arr = None

        def decode_free_vector(x_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            x_eval = np.clip(np.asarray(x_vec, dtype=np.float64).ravel(), 0.0, 1.0)
            amp_full_flat = np.ones((total_pixels,), dtype=np.float64)
            amp_full_flat[active_flat] = x_eval
            amp_full = amp_full_flat.reshape(full_shape)
            return x_eval, amp_full

        best_x = None
        best_obj = np.inf
        best_metrics = None
        best_res = None
        best_amp_full = None
        best_phase_full = None
        stage_t0 = time.time()
        eval_counter = 0
        last_preview_ts = {"t": 0.0}

        for start_idx in range(int(n_starts)):
            self._check_stop(stop_callback)
            if x_seed_arr is not None and int(start_idx) == 0:
                x0 = np.asarray(x_seed_arr, dtype=np.float64).copy()
            elif int(start_idx) == 0:
                x0 = np.ones((int(active_flat.size),), dtype=np.float64)
            else:
                x0 = rng.uniform(0.05, 0.95, size=int(active_flat.size)).astype(np.float64)
            x0 = np.clip(x0, 0.0, 1.0)

            iter_counter = {"count": 0}
            n_variables = int(x0.size)
            est_evals_per_iter = 1.0
            start_eval_base = int(eval_counter)

            if progress_callback is not None:
                progress_callback(
                    f"{stage_name}: start {start_idx + 1}/{n_starts}",
                    max(0.0, float(start_idx) / max(1.0, float(n_starts))),
                    {
                        "objective": float(best_obj) if np.isfinite(best_obj) else None,
                        "iter_index": int(iter_counter["count"]),
                        "eval_index": int(eval_counter),
                        "eval_in_start": 0,
                        "n_variables": int(n_variables),
                        "n_free_variables": int(n_variables),
                        "est_evals_per_iter": float(est_evals_per_iter),
                        "maxiter": int(maxiter),
                        "surrogate_loss_enabled": True,
                        "surrogate_optimizer_name": str(surrogate_optimizer_name),
                    },
                )

            def objective_with_grad(x):
                nonlocal eval_counter
                self._check_stop(stop_callback)
                x_eval = np.clip(np.asarray(x, dtype=np.float64).ravel(), 0.0, 1.0)
                amp_free_t = torch.as_tensor(x_eval, dtype=torch.float64, device=device)
                amp_free_t.requires_grad_(True)
                amp_full_flat_t = torch.ones((total_pixels,), dtype=torch.float64, device=device)
                amp_full_flat_t = amp_full_flat_t.scatter(0, active_idx_t, amp_free_t)
                amp_full_t = amp_full_flat_t.reshape(full_shape)
                complex_mask_t = amp_full_t.to(torch.complex128)[row0:row1, col0:col1]
                res_torch = self._linear_fastpath.simulate_surrogate_planes_from_complex_mask_tensor(
                    complex_mask_t,
                    need_xz=bool(self._need_x or self._need_z),
                    need_yz=bool(self._need_y or self._need_z),
                    stop_callback=stop_callback,
                )
                surrogate_obj_t, surrogate_metrics_t = self._surrogate_objective_from_torch_res(res_torch)
                surrogate_obj_t.backward()
                grad_t = amp_free_t.grad
                if grad_t is None:
                    grad_t = torch.zeros_like(amp_free_t)

                surrogate_obj = float(surrogate_obj_t.detach().cpu().item())
                metrics_np = {
                    "width_x_um": float(surrogate_metrics_t["width_x_um"].detach().cpu().item()),
                    "width_y_um": float(surrogate_metrics_t["width_y_um"].detach().cpu().item()),
                    "width_z_um": float(surrogate_metrics_t["width_z_um"].detach().cpu().item()),
                    "abs_x_minus_y_um": float(
                        surrogate_metrics_t["abs_x_minus_y_um"].detach().cpu().item()
                    ),
                    "sidelobe_penalty": float(
                        surrogate_metrics_t["sidelobe_penalty"].detach().cpu().item()
                    ),
                }
                eval_counter += 1
                self.history.append(
                    {
                        "stage": stage_name,
                        "start_index": int(start_idx),
                        "eval_index": int(eval_counter),
                        "objective": float(surrogate_obj),
                        "width_x_um": float(metrics_np["width_x_um"]),
                        "width_y_um": float(metrics_np["width_y_um"]),
                        "width_z_um": float(metrics_np["width_z_um"]),
                        "abs_x_minus_y_um": float(metrics_np["abs_x_minus_y_um"]),
                        "sidelobe_penalty": float(metrics_np["sidelobe_penalty"]),
                        "elapsed_s": float(time.time() - stage_t0),
                        "anneal_alpha": 1.0,
                        "binary_structure": "pixel",
                        "surrogate_loss_enabled": True,
                        "surrogate_optimizer_name": str(surrogate_optimizer_name),
                    }
                )
                if progress_callback is not None:
                    now = time.time()
                    if (now - float(last_preview_ts["t"])) >= 0.20:
                        last_preview_ts["t"] = now
                        eval_in_start = int(eval_counter - start_eval_base)
                        amp_preview = self._downsample_preview(
                            amp_full_t.detach().cpu().numpy(),
                            max_dim=120,
                        )
                        phase_preview = self._downsample_preview(
                            np.zeros(full_shape, dtype=np.float64),
                            max_dim=120,
                        )
                        progress_callback(
                            f"{stage_name}: surrogate evaluating",
                            (
                                float(start_idx)
                                + min(0.95, float(eval_in_start) / max(1.0, float(maxiter)))
                            )
                            / max(1.0, float(n_starts)),
                            {
                                "objective": float(surrogate_obj),
                                "width_x_um": float(metrics_np["width_x_um"]),
                                "width_y_um": float(metrics_np["width_y_um"]),
                                "width_z_um": float(metrics_np["width_z_um"]),
                                "abs_x_minus_y_um": float(metrics_np["abs_x_minus_y_um"]),
                                "sidelobe_penalty": float(metrics_np["sidelobe_penalty"]),
                                "iter_index": int(iter_counter["count"]),
                                "eval_index": int(eval_counter),
                                "eval_in_start": int(eval_in_start),
                                "n_variables": int(n_variables),
                                "n_free_variables": int(n_variables),
                                "est_evals_per_iter": float(est_evals_per_iter),
                                "maxiter": int(maxiter),
                                "mask_amp_preview": amp_preview,
                                "mask_phase_preview": phase_preview,
                                "preview_shape_full": [int(full_shape[0]), int(full_shape[1])],
                                "axial_symmetry": False,
                                "binary_structure": "pixel",
                                "surrogate_loss_enabled": True,
                                "surrogate_optimizer_name": str(surrogate_optimizer_name),
                            },
                        )
                return float(surrogate_obj), np.asarray(
                    grad_t.detach().cpu().numpy(),
                    dtype=np.float64,
                )

            def callback_fn(_xk):
                self._check_stop(stop_callback)
                iter_counter["count"] += 1
                if progress_callback is not None:
                    eval_in_start = int(eval_counter - start_eval_base)
                    progress_callback(
                        f"{stage_name}: surrogate iter {iter_counter['count']}",
                        (
                            float(start_idx)
                            + min(0.95, float(iter_counter["count"]) / max(1.0, float(maxiter)))
                        )
                        / max(1.0, float(n_starts)),
                        {
                            "iter_index": int(iter_counter["count"]),
                            "eval_index": int(eval_counter),
                            "eval_in_start": int(eval_in_start),
                            "n_variables": int(n_variables),
                            "n_free_variables": int(n_variables),
                            "est_evals_per_iter": float(est_evals_per_iter),
                            "maxiter": int(maxiter),
                            "surrogate_loss_enabled": True,
                            "surrogate_optimizer_name": str(surrogate_optimizer_name),
                        },
                    )
                return False

            options = {"maxiter": int(maxiter)}
            if surrogate_optimizer_name == "L-BFGS-B":
                if ftol > 0:
                    options["ftol"] = float(ftol)
                if gtol > 0:
                    options["gtol"] = float(gtol)
            elif surrogate_optimizer_name == "SLSQP":
                if ftol > 0:
                    options["ftol"] = float(ftol)

            result = minimize(
                objective_with_grad,
                x0=x0,
                method=surrogate_optimizer_name,
                jac=True,
                bounds=[(0.0, 1.0)] * int(n_variables),
                callback=callback_fn,
                options=options,
            )

            x_result = np.clip(np.asarray(getattr(result, "x", x0), dtype=np.float64).ravel(), 0.0, 1.0)
            x_final, amp_var = decode_free_vector(x_result)
            phase_var = np.zeros(full_shape, dtype=np.float64)
            obj_final, metrics_final, res_final, amp_full, phase_full = self._simulate_from_mask_maps(
                base_field_after_fresnel=base_field_after_fresnel,
                pupil_mask=pupil_mask,
                amp_var=amp_var,
                phase_var=phase_var,
                full_shape=full_shape,
                axial_radial_fastpath_ok=False,
                stop_callback=stop_callback,
            )

            if obj_final < best_obj:
                best_obj = float(obj_final)
                best_x = x_final.copy()
                best_metrics = dict(metrics_final)
                best_res = res_final
                best_amp_full = amp_full
                best_phase_full = phase_full

            if progress_callback is not None:
                frac = float(start_idx + 1) / max(1.0, float(n_starts))
                best_amp_preview = None
                best_phase_preview = None
                if best_amp_full is not None and best_phase_full is not None:
                    best_amp_preview = self._downsample_preview(best_amp_full, max_dim=120)
                    best_phase_preview = self._downsample_preview(best_phase_full, max_dim=120)
                progress_callback(
                    f"{stage_name}: done surrogate start {start_idx + 1}/{n_starts}",
                    frac,
                    {
                        "objective": float(best_obj),
                        "width_x_um": float(best_metrics["width_x_um"]) if best_metrics else None,
                        "width_y_um": float(best_metrics["width_y_um"]) if best_metrics else None,
                        "width_z_um": float(best_metrics["width_z_um"]) if best_metrics else None,
                        "abs_x_minus_y_um": (
                            float(best_metrics.get("abs_x_minus_y_um", np.nan))
                            if best_metrics
                            else None
                        ),
                        "sidelobe_penalty": (
                            float(best_metrics.get("sidelobe_penalty", np.nan))
                            if best_metrics
                            else None
                        ),
                        "iter_index": int(iter_counter["count"]),
                        "eval_index": int(eval_counter),
                        "eval_in_start": int(eval_counter - start_eval_base),
                        "n_variables": int(n_variables),
                        "n_free_variables": int(n_variables),
                        "est_evals_per_iter": float(est_evals_per_iter),
                        "maxiter": int(maxiter),
                        "mask_amp_preview": best_amp_preview,
                        "mask_phase_preview": best_phase_preview,
                        "preview_shape_full": [int(full_shape[0]), int(full_shape[1])],
                        "axial_symmetry": False,
                        "binary_structure": "pixel",
                        "surrogate_loss_enabled": True,
                        "surrogate_optimizer_name": str(surrogate_optimizer_name),
                    },
                )

        if best_x is None:
            raise RuntimeError("Surrogate optimization failed to produce any valid candidate.")

        return best_x, best_obj, best_metrics, best_res, best_amp_full, best_phase_full

    def _run_single_stage(
        self,
        *,
        stage_name: str,
        base_field_after_fresnel: np.ndarray,
        pupil_mask: np.ndarray,
        mode: str,
        optimizer_name: str,
        var_shape: Tuple[int, int],
        full_shape: Tuple[int, int],
        n_starts: int,
        maxiter: int,
        ftol: float,
        gtol: float,
        seed: Optional[int],
        x_seed: Optional[np.ndarray],
        progress_callback=None,
        stop_callback=None,
    ) -> Tuple[np.ndarray, float, Dict[str, float], Dict[str, Any], np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        best_x = None
        best_obj = np.inf
        best_metrics = None
        best_res = None
        best_amp_full = None
        best_phase_full = None
        stage_t0 = time.time()
        eval_counter = 0
        last_preview_ts = {"t": 0.0}
        binary_amplitude = bool(self._config.binary_amplitude and mode == "amplitude")
        binary_structure = str(getattr(self._config, "binary_structure", "pixel")).strip().lower()
        use_simple_ring = bool(binary_amplitude and (binary_structure == "simple_ring_block"))
        use_simple_ellipse = bool(binary_amplitude and (binary_structure == "simple_ellipse_block"))
        use_binary_latent = bool(binary_amplitude and (not use_simple_ring) and (not use_simple_ellipse))
        pupil_radius_m = 0.5 * float(self.params.get("pupil_d", 0.0))
        axial_symmetry = bool(self._config.axial_symmetry)
        support_mask = self._pupil_support_for_shape(
            pupil_mask=pupil_mask,
            var_shape=var_shape,
            threshold=0.5,
        )
        use_surrogate_gradient = bool(
            (not axial_symmetry)
            and str(mode).strip().lower() == "amplitude"
            and (not bool(binary_amplitude))
            and (not use_simple_ring)
            and (not use_simple_ellipse)
            and (self._linear_fastpath_enabled)
            and (self._linear_fastpath is not None)
        )
        if use_surrogate_gradient:
            return self._run_surrogate_gradient_stage(
                stage_name=stage_name,
                base_field_after_fresnel=base_field_after_fresnel,
                pupil_mask=pupil_mask,
                optimizer_name=optimizer_name,
                support_mask=support_mask,
                full_shape=full_shape,
                n_starts=n_starts,
                maxiter=maxiter,
                ftol=ftol,
                gtol=gtol,
                seed=seed,
                x_seed=x_seed,
                progress_callback=progress_callback,
                stop_callback=stop_callback,
            )
        radial = None
        n_bins = 0
        support_bins = None
        radial_axis_norm = None
        if bool(axial_symmetry):
            radial = self._build_radial_parameterization(
                var_shape,
                support_mask=support_mask,
                n_bins=int(self._radial_bins),
            )
            n_bins = int(radial["n_bins"])
            support_bins = np.asarray(radial["support_by_bin"], dtype=bool)
            radial_axis_norm = np.linspace(0.0, 1.0, n_bins, dtype=np.float64)
            if (
                self._axial_radial_fastpath_enabled
                and (self._axial_radial_fastpath is not None)
            ):
                axis_fast = np.asarray(
                    self._axial_radial_fastpath.radial_norm_axis, dtype=np.float64
                ).ravel()
                if int(axis_fast.size) == int(n_bins):
                    radial_axis_norm = np.clip(axis_fast, 0.0, 1.0)
        if use_simple_ring:
            bounds = [(0.0, 1.0), (0.0, 1.0)]
        elif use_simple_ellipse:
            bounds = [(0.0, 1.0)] * 4
        elif bool(axial_symmetry):
            bounds = self._vector_bounds_for_stage(
                mode=mode,
                var_shape=(n_bins, 1),
                binary_amplitude=use_binary_latent,
            )
            bounds = self._apply_fixed_outside_pupil_bounds(
                bounds=bounds,
                mode=mode,
                support_mask=support_bins[:, None],
                binary_amplitude=use_binary_latent,
            )
        else:
            bounds = self._vector_bounds_for_stage(
                mode=mode,
                var_shape=var_shape,
                binary_amplitude=use_binary_latent,
            )
            bounds = self._apply_fixed_outside_pupil_bounds(
                bounds=bounds,
                mode=mode,
                support_mask=support_mask,
                binary_amplitude=use_binary_latent,
            )
        n_free_variables = int(sum(1 for lo, hi in bounds if float(hi) > float(lo)))
        expected_len = int(len(bounds))
        x_seed_arr = None
        if x_seed is not None:
            try:
                x_seed_arr = np.asarray(x_seed, dtype=np.float64).ravel()
            except Exception:
                x_seed_arr = None
            if x_seed_arr is not None and int(x_seed_arr.size) != int(expected_len):
                x_seed_arr = None

        def structure_info_from_x_eval(x_eval: np.ndarray) -> Dict[str, Any]:
            arr = np.asarray(x_eval, dtype=np.float64).ravel()
            info: Dict[str, Any] = {
                "binary_structure": str(binary_structure),
            }
            if use_simple_ring and arr.size >= 2:
                ring_r_in = float(min(float(arr[0]), float(arr[1])))
                ring_r_out = float(max(float(arr[0]), float(arr[1])))
                info.update(
                    {
                        "ring_inner_radius_norm": ring_r_in,
                        "ring_outer_radius_norm": ring_r_out,
                        "ring_inner_radius_m": float(ring_r_in * pupil_radius_m),
                        "ring_outer_radius_m": float(ring_r_out * pupil_radius_m),
                        "ring_inner_radius_mm": float(ring_r_in * pupil_radius_m * 1e3),
                        "ring_outer_radius_mm": float(ring_r_out * pupil_radius_m * 1e3),
                    }
                )
            elif use_simple_ellipse and arr.size >= 4:
                ellipse_inner_dx = float(np.clip(arr[0], 0.0, 1.0))
                ellipse_inner_dy = float(np.clip(arr[1], 0.0, 1.0))
                ellipse_outer_dx = float(np.clip(arr[2], 0.0, 1.0))
                ellipse_outer_dy = float(np.clip(arr[3], 0.0, 1.0))
                info.update(
                    {
                        "ellipse_inner_diameter_x_norm": ellipse_inner_dx,
                        "ellipse_inner_diameter_y_norm": ellipse_inner_dy,
                        "ellipse_outer_diameter_x_norm": ellipse_outer_dx,
                        "ellipse_outer_diameter_y_norm": ellipse_outer_dy,
                        "ellipse_inner_diameter_x_m": float(ellipse_inner_dx * float(self.params.get("pupil_d", 0.0))),
                        "ellipse_inner_diameter_y_m": float(ellipse_inner_dy * float(self.params.get("pupil_d", 0.0))),
                        "ellipse_outer_diameter_x_m": float(ellipse_outer_dx * float(self.params.get("pupil_d", 0.0))),
                        "ellipse_outer_diameter_y_m": float(ellipse_outer_dy * float(self.params.get("pupil_d", 0.0))),
                        "ellipse_inner_diameter_x_mm": float(ellipse_inner_dx * float(self.params.get("pupil_d", 0.0)) * 1e3),
                        "ellipse_inner_diameter_y_mm": float(ellipse_inner_dy * float(self.params.get("pupil_d", 0.0)) * 1e3),
                        "ellipse_outer_diameter_x_mm": float(ellipse_outer_dx * float(self.params.get("pupil_d", 0.0)) * 1e3),
                        "ellipse_outer_diameter_y_mm": float(ellipse_outer_dy * float(self.params.get("pupil_d", 0.0)) * 1e3),
                    }
                )
            return info

        def decode_vector_to_stage_maps(
            x_vec: np.ndarray, anneal_alpha: float
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x_eval = self._clip_vector_to_bounds(x_vec, bounds)
            if use_simple_ring:
                amp_profile, ring_r_in, ring_r_out = self._simple_ring_block_profile(
                    radial_axis_norm=radial_axis_norm,
                    inner_radius=float(x_eval[0]),
                    outer_radius=float(x_eval[1]),
                    support_bins=support_bins,
                )
                phase_profile = np.zeros((int(radial["n_bins"]),), dtype=np.float64)
                x_eval = np.asarray([ring_r_in, ring_r_out], dtype=np.float64)
                amp_var = self._radial_profile_to_map(amp_profile, radial["bin_map"])
                phase_var = self._radial_profile_to_map(phase_profile, radial["bin_map"])
            elif use_simple_ellipse:
                amp_var, ellipse_inner_dx, ellipse_inner_dy, ellipse_outer_dx, ellipse_outer_dy = self._simple_ellipse_ring_block_map(
                    var_shape=var_shape,
                    inner_diameter_x=float(x_eval[0]),
                    inner_diameter_y=float(x_eval[1]),
                    outer_diameter_x=float(x_eval[2]),
                    outer_diameter_y=float(x_eval[3]),
                    support_mask=support_mask,
                )
                phase_var = np.zeros(var_shape, dtype=np.float64)
                amp_profile = np.asarray([], dtype=np.float64)
                phase_profile = np.asarray([], dtype=np.float64)
                x_eval = np.asarray(
                    [ellipse_inner_dx, ellipse_inner_dy, ellipse_outer_dx, ellipse_outer_dy],
                    dtype=np.float64,
                )
            elif bool(axial_symmetry):
                amp_profile, phase_profile = self._unpack_stage_profiles(
                    x_eval,
                    mode=mode,
                    n_bins=int(radial["n_bins"]),
                    binary_amplitude=use_binary_latent,
                    anneal_alpha=float(anneal_alpha),
                )
                amp_profile = np.clip(amp_profile, 0.0, 1.0)
                phase_profile = np.clip(phase_profile, -np.pi, np.pi)
                amp_profile[~support_bins] = 1.0
                phase_profile[~support_bins] = 0.0
                amp_var = self._radial_profile_to_map(amp_profile, radial["bin_map"])
                phase_var = self._radial_profile_to_map(phase_profile, radial["bin_map"])
            else:
                amp_var, phase_var = self._unpack_stage_maps(
                    x_eval,
                    mode=mode,
                    var_shape=var_shape,
                    binary_amplitude=use_binary_latent,
                    anneal_alpha=float(anneal_alpha),
                )
                amp_var = np.clip(amp_var, 0.0, 1.0)
                phase_var = np.clip(phase_var, -np.pi, np.pi)
                amp_var[~support_mask] = 1.0
                phase_var[~support_mask] = 0.0
                amp_profile = None
                phase_profile = None
            return x_eval, amp_var, phase_var, amp_profile, phase_profile

        for start_idx in range(int(n_starts)):
            self._check_stop(stop_callback)
            start_seed = x_seed_arr if (x_seed_arr is not None and int(start_idx) == 0) else None
            n = (
                int(radial["n_bins"])
                if bool(axial_symmetry) and (radial is not None)
                else int(var_shape[0] * var_shape[1])
            )
            if use_simple_ring:
                if start_seed is not None and int(np.asarray(start_seed).size) == 2:
                    x0 = np.asarray(start_seed, dtype=np.float64).copy()
                elif int(start_idx) == 0:
                    x0 = np.asarray([0.30, 0.75], dtype=np.float64)
                else:
                    x0 = rng.uniform(0.0, 1.0, size=2).astype(np.float64)
            elif use_simple_ellipse:
                if start_seed is not None and int(np.asarray(start_seed).size) == 4:
                    x0 = np.asarray(start_seed, dtype=np.float64).copy()
                elif int(start_idx) == 0:
                    x0 = np.asarray([0.20, 0.20, 0.70, 0.70], dtype=np.float64)
                else:
                    x0 = rng.uniform(0.0, 1.0, size=4).astype(np.float64)
            elif start_seed is not None:
                x0 = np.asarray(start_seed, dtype=np.float64).copy()
            elif mode == "amplitude":
                if use_binary_latent:
                    if int(start_idx) == 0:
                        x0 = np.full((n,), 6.0, dtype=np.float64)
                    else:
                        x0 = rng.uniform(-2.0, 2.0, size=n).astype(np.float64)
                else:
                    if int(start_idx) == 0:
                        x0 = np.ones((n,), dtype=np.float64)
                    else:
                        x0 = rng.uniform(0.05, 0.95, size=n).astype(np.float64)
            elif mode == "phase":
                if int(start_idx) == 0:
                    x0 = np.zeros((n,), dtype=np.float64)
                else:
                    x0 = rng.uniform(-np.pi, np.pi, size=n).astype(np.float64)
            else:
                if int(start_idx) == 0:
                    amp0 = np.ones((n,), dtype=np.float64)
                    ph0 = np.zeros((n,), dtype=np.float64)
                else:
                    amp0 = rng.uniform(0.05, 0.95, size=n).astype(np.float64)
                    ph0 = rng.uniform(-np.pi, np.pi, size=n).astype(np.float64)
                x0 = np.concatenate((amp0, ph0), axis=0)
            x0 = self._clip_vector_to_bounds(x0, bounds)
            iter_counter = {"count": 0}
            n_variables = int(np.asarray(x0).size)
            est_evals_per_iter = self._estimate_evals_per_iteration(
                optimizer_name=optimizer_name,
                n_variables=n_free_variables,
            )
            start_eval_base = int(eval_counter)

            if progress_callback is not None:
                progress_callback(
                    f"{stage_name}: start {start_idx + 1}/{n_starts}",
                    max(0.0, float(start_idx) / max(1.0, float(n_starts))),
                    {
                        "objective": float(best_obj) if np.isfinite(best_obj) else None,
                        "iter_index": int(iter_counter["count"]),
                        "eval_index": int(eval_counter),
                        "eval_in_start": 0,
                        "n_variables": int(n_variables),
                        "n_free_variables": int(n_free_variables),
                        "est_evals_per_iter": float(est_evals_per_iter),
                        "maxiter": int(maxiter),
                    },
                )

            def objective_fn(x):
                nonlocal eval_counter
                self._check_stop(stop_callback)
                eval_in_start = int(eval_counter - start_eval_base + 1)
                anneal_alpha = 1.0
                if use_binary_latent:
                    anneal_alpha = self._anneal_alpha(
                        eval_in_start=eval_in_start,
                        est_evals_per_iter=est_evals_per_iter,
                        maxiter=maxiter,
                        alpha_start=float(self._config.binary_anneal_alpha_start),
                        alpha_end=float(self._config.binary_anneal_alpha_end),
                    )
                _x_eval, amp_var, phase_var, amp_profile, phase_profile = decode_vector_to_stage_maps(
                    x, anneal_alpha=float(anneal_alpha)
                )
                objective, metrics, _res, _amp_full, _phase_full = self._simulate_from_mask_maps(
                    base_field_after_fresnel=base_field_after_fresnel,
                    pupil_mask=pupil_mask,
                    amp_var=amp_var,
                    phase_var=phase_var,
                    full_shape=full_shape,
                    amp_profile=amp_profile,
                    phase_profile=phase_profile,
                    radial=(radial if bool(axial_symmetry) else None),
                    support_mask=(support_mask if bool(axial_symmetry) else None),
                    axial_radial_fastpath_ok=bool(axial_symmetry),
                    stop_callback=stop_callback,
                )
                eval_counter += 1
                self.history.append(
                    {
                        "stage": stage_name,
                        "start_index": int(start_idx),
                        "eval_index": int(eval_counter),
                        "objective": float(objective),
                        "width_x_um": float(metrics["width_x_um"]),
                        "width_y_um": float(metrics["width_y_um"]),
                        "width_z_um": float(metrics["width_z_um"]),
                        "abs_x_minus_y_um": float(metrics.get("abs_x_minus_y_um", np.nan)),
                        "sidelobe_penalty": float(metrics.get("sidelobe_penalty", np.nan)),
                        "elapsed_s": float(time.time() - stage_t0),
                        "anneal_alpha": float(anneal_alpha),
                        **structure_info_from_x_eval(_x_eval),
                    }
                )
                if progress_callback is not None:
                    now = time.time()
                    if (now - float(last_preview_ts["t"])) >= 0.20:
                        last_preview_ts["t"] = now
                        eval_in_start = int(eval_counter - start_eval_base)
                        iter_like = float(iter_counter["count"])
                        if np.isfinite(est_evals_per_iter) and est_evals_per_iter > 0:
                            iter_like = max(
                                iter_like,
                                min(float(maxiter), float(eval_in_start) / float(est_evals_per_iter)),
                            )
                        stage_progress = (
                            float(start_idx)
                            + min(0.95, float(iter_like) / max(1.0, float(maxiter)))
                        )
                        frac = stage_progress / max(1.0, float(n_starts))
                        amp_preview = self._downsample_preview(_amp_full, max_dim=120)
                        phase_preview = self._downsample_preview(_phase_full, max_dim=120)
                        progress_callback(
                            f"{stage_name}: evaluating",
                            frac,
                            {
                                "objective": float(objective),
                                "width_x_um": float(metrics["width_x_um"]),
                                "width_y_um": float(metrics["width_y_um"]),
                                "width_z_um": float(metrics["width_z_um"]),
                                "abs_x_minus_y_um": float(metrics.get("abs_x_minus_y_um", np.nan)),
                                "sidelobe_penalty": float(metrics.get("sidelobe_penalty", np.nan)),
                                "iter_index": int(iter_counter["count"]),
                                "eval_index": int(eval_counter),
                                "eval_in_start": int(eval_in_start),
                                "n_variables": int(n_variables),
                                "n_free_variables": int(n_free_variables),
                                "est_evals_per_iter": float(est_evals_per_iter),
                                "maxiter": int(maxiter),
                                "mask_amp_preview": amp_preview,
                                "mask_phase_preview": phase_preview,
                                "preview_shape_full": [int(full_shape[0]), int(full_shape[1])],
                                "anneal_alpha": float(anneal_alpha),
                                "axial_symmetry": bool(axial_symmetry),
                                **structure_info_from_x_eval(_x_eval),
                            },
                        )
                return float(objective)

            def callback_fn(_xk):
                self._check_stop(stop_callback)
                iter_counter["count"] += 1
                if progress_callback is not None:
                    eval_in_start = int(eval_counter - start_eval_base)
                    stage_progress = (
                        float(start_idx)
                        + min(0.95, float(iter_counter["count"]) / max(1.0, float(maxiter)))
                    )
                    frac = stage_progress / max(1.0, float(n_starts))
                    progress_callback(
                        f"{stage_name}: iter {iter_counter['count']}",
                        frac,
                        {
                            "iter_index": int(iter_counter["count"]),
                            "eval_index": int(eval_counter),
                            "eval_in_start": int(eval_in_start),
                            "n_variables": int(n_variables),
                            "n_free_variables": int(n_free_variables),
                            "est_evals_per_iter": float(est_evals_per_iter),
                            "maxiter": int(maxiter),
                        },
                    )
                return False

            def callback_da_like():
                self._check_stop(stop_callback)
                iter_counter["count"] += 1
                if progress_callback is not None:
                    eval_in_start = int(eval_counter - start_eval_base)
                    stage_progress = (
                        float(start_idx)
                        + min(0.95, float(iter_counter["count"]) / max(1.0, float(maxiter)))
                    )
                    frac = stage_progress / max(1.0, float(n_starts))
                    progress_callback(
                        f"{stage_name}: iter {iter_counter['count']}",
                        frac,
                        {
                            "iter_index": int(iter_counter["count"]),
                            "eval_index": int(eval_counter),
                            "eval_in_start": int(eval_in_start),
                            "n_variables": int(n_variables),
                            "n_free_variables": int(n_free_variables),
                            "est_evals_per_iter": float(est_evals_per_iter),
                            "maxiter": int(maxiter),
                        },
                    )

            def callback_differential_evolution(_xk, _convergence=0.0):
                callback_da_like()
                return False

            def callback_shgo(_x):
                callback_da_like()

            options = {"maxiter": int(maxiter)}
            if ftol > 0:
                options["ftol"] = float(ftol)
            if optimizer_name == "L-BFGS-B" and gtol > 0:
                options["gtol"] = float(gtol)
            if optimizer_name == "SLSQP":
                options["eps"] = 1e-4
            if optimizer_name == "Differential Evolution":
                if differential_evolution is None:
                    raise ValueError("Differential Evolution is unavailable in this SciPy build.")
                de_kwargs = {
                    "func": objective_fn,
                    "bounds": bounds,
                    "maxiter": int(maxiter),
                    "popsize": 8,
                    "polish": False,
                    "updating": "immediate",
                    "workers": 1,
                    "callback": callback_differential_evolution,
                    "x0": np.asarray(x0, dtype=np.float64).copy(),
                    "rng": np.random.default_rng(int(rng.integers(0, np.iinfo(np.int32).max))),
                }
                if ftol > 0:
                    de_kwargs["tol"] = float(ftol)
                result = differential_evolution(**de_kwargs)
            elif optimizer_name == "SHGO":
                if shgo is None:
                    raise ValueError("SHGO is unavailable in this SciPy build.")
                shgo_options = {"disp": False}
                if ftol > 0:
                    shgo_options["f_tol"] = float(ftol)
                result = shgo(
                    objective_fn,
                    bounds=bounds,
                    n=int(max(24, min(256, 4 * max(1, n_free_variables)))),
                    iters=int(maxiter),
                    options=shgo_options,
                    callback=callback_shgo,
                    workers=1,
                )
            elif optimizer_name == "Nelder-Mead":
                # Nelder-Mead is derivative-free and does not require bounds
                # (we enforce bounds by clipping in objective_fn).
                options = {"maxiter": int(maxiter)}
                if ftol > 0:
                    options["xatol"] = float(ftol)
                    options["fatol"] = float(ftol)
                result = minimize(
                    objective_fn,
                    x0=x0,
                    method=optimizer_name,
                    callback=callback_fn,
                    options=options,
                )
            else:
                result = minimize(
                    objective_fn,
                    x0=x0,
                    method=optimizer_name,
                    bounds=bounds,
                    callback=callback_fn,
                    options=options,
                )

            x_result = np.asarray(getattr(result, "x", x0), dtype=np.float64).ravel()
            if int(x_result.size) != int(len(bounds)):
                x_result = np.asarray(x0, dtype=np.float64).ravel()
            x_final = self._clip_vector_to_bounds(x_result, bounds)
            final_alpha = 1.0
            if use_binary_latent:
                final_alpha = float(self._config.binary_anneal_alpha_end)
            _x_eval, amp_var, phase_var, amp_profile, phase_profile = decode_vector_to_stage_maps(
                x_final, anneal_alpha=float(final_alpha)
            )
            obj_final, metrics_final, res_final, amp_full, phase_full = self._simulate_from_mask_maps(
                base_field_after_fresnel=base_field_after_fresnel,
                pupil_mask=pupil_mask,
                amp_var=amp_var,
                phase_var=phase_var,
                full_shape=full_shape,
                amp_profile=amp_profile,
                phase_profile=phase_profile,
                radial=(radial if bool(axial_symmetry) else None),
                support_mask=(support_mask if bool(axial_symmetry) else None),
                axial_radial_fastpath_ok=bool(axial_symmetry),
                stop_callback=stop_callback,
            )

            if obj_final < best_obj:
                best_obj = float(obj_final)
                best_x = x_final.copy()
                best_metrics = dict(metrics_final)
                best_res = res_final
                best_amp_full = amp_full
                best_phase_full = phase_full

            if progress_callback is not None:
                frac = float(start_idx + 1) / max(1.0, float(n_starts))
                best_amp_preview = None
                best_phase_preview = None
                if best_amp_full is not None and best_phase_full is not None:
                    best_amp_preview = self._downsample_preview(best_amp_full, max_dim=120)
                    best_phase_preview = self._downsample_preview(best_phase_full, max_dim=120)
                progress_callback(
                    f"{stage_name}: done start {start_idx + 1}/{n_starts}",
                    frac,
                    {
                        "objective": float(best_obj),
                        "width_x_um": float(best_metrics["width_x_um"]) if best_metrics else None,
                        "width_y_um": float(best_metrics["width_y_um"]) if best_metrics else None,
                        "width_z_um": float(best_metrics["width_z_um"]) if best_metrics else None,
                        "abs_x_minus_y_um": (
                            float(best_metrics.get("abs_x_minus_y_um", np.nan))
                            if best_metrics
                            else None
                        ),
                        "sidelobe_penalty": (
                            float(best_metrics.get("sidelobe_penalty", np.nan))
                            if best_metrics
                            else None
                        ),
                        "iter_index": int(iter_counter["count"]),
                        "eval_index": int(eval_counter),
                        "eval_in_start": int(eval_counter - start_eval_base),
                        "n_variables": int(n_variables),
                        "n_free_variables": int(n_free_variables),
                        "est_evals_per_iter": float(est_evals_per_iter),
                        "maxiter": int(maxiter),
                        "mask_amp_preview": best_amp_preview,
                        "mask_phase_preview": best_phase_preview,
                        "preview_shape_full": [int(full_shape[0]), int(full_shape[1])],
                        "axial_symmetry": bool(axial_symmetry),
                        **(
                            structure_info_from_x_eval(best_x)
                            if best_x is not None
                            else {"binary_structure": str(binary_structure)}
                        ),
                    },
                )

        if best_x is None:
            raise RuntimeError("Optimization failed to produce any valid candidate.")

        return best_x, best_obj, best_metrics, best_res, best_amp_full, best_phase_full

    def optimize(
        self,
        input_intensity: np.ndarray,
        config: OptimizerConfig,
        progress_callback=None,
        stop_callback=None,
    ) -> OptimizerResult:
        self._check_stop(stop_callback)
        self.history = []
        self._surrogate_loss_used = False
        self._surrogate_loss_reason = "disabled"
        self._surrogate_optimizer_name = None

        mask = np.asarray(input_intensity, dtype=np.float64)
        if mask.ndim != 2:
            raise ValueError("input_intensity must be a 2D array.")
        if not np.all(np.isfinite(mask)):
            raise ValueError("input_intensity contains NaN/Inf.")
        if float(mask.min()) < 0.0 or float(mask.max()) > 1.0:
            raise ValueError("input_intensity must be normalized in [0, 1].")

        full_shape = tuple(int(v) for v in mask.shape)
        self._config = self._sanitize_config(config, full_shape=full_shape)
        self._need_x, self._need_y, self._need_z = self._active_axis_flags(
            self._config.weights_xyz,
            weight_abs_xy=float(getattr(self._config, "weight_abs_xy", 0.0)),
        )
        self._compute_xy, self._compute_xz, self._compute_yz = self._required_plane_flags(
            self._need_x, self._need_y, self._need_z
        )

        self._check_stop(stop_callback)
        if progress_callback is not None:
            progress_callback("Preparing post-Fresnel field", 0.01, None)

        prep = self.simulator.prepare_pupil_field_from_input(
            input_intensity=mask,
            size_x=float(self.params["size_x"]),
            size_y=float(self.params["size_y"]),
            entrance_pupil_diameter=float(self.params["pupil_d"]),
            fresnel_distance=float(self.params["fresnel_z"]),
            use_gaussian_profile=bool(self.params["use_gaussian"]),
            gaussian_waist_x=(
                float(self.params["gauss_wx"]) if bool(self.params["use_gaussian"]) else None
            ),
            gaussian_waist_y=(
                float(self.params["gauss_wy"]) if bool(self.params["use_gaussian"]) else None
            ),
            custom_mask=None,
            ring_inner_diameter=None,
            ring_outer_diameter=None,
            stop_callback=stop_callback,
        )

        base_field_after_fresnel = np.asarray(prep["field_after_fresnel"], dtype=np.complex128)
        pupil_mask = np.asarray(prep["entrance_pupil_mask"], dtype=np.float64)
        trim_window = prep.get("pupil_trim_window")
        if (
            isinstance(trim_window, (list, tuple))
            and len(trim_window) == 4
            and int(trim_window[1]) > int(trim_window[0])
            and int(trim_window[3]) > int(trim_window[2])
        ):
            self._pupil_trim_window = [
                int(trim_window[0]),
                int(trim_window[1]),
                int(trim_window[2]),
                int(trim_window[3]),
            ]
        else:
            self._pupil_trim_window = None
        self._size_x_for_psf = float(prep.get("size_x_for_psf", self.params["size_x"]))
        self._size_y_for_psf = float(prep.get("size_y_for_psf", self.params["size_y"]))
        if bool(self._config.axial_symmetry):
            self._linear_fastpath = None
            self._linear_fastpath_enabled = False
            self._linear_fastpath_reason = "disabled: axial symmetry on"
            if progress_callback is not None:
                progress_callback("Preparing axial-symmetry precomputed field", 0.03, None)
            self._initialize_axial_radial_fastpath(
                base_field_after_fresnel=base_field_after_fresnel,
                pupil_mask=pupil_mask,
                stop_callback=stop_callback,
            )
            if progress_callback is not None:
                progress_callback(
                    (
                        "Axial-symmetry fast path enabled"
                        if self._axial_radial_fastpath_enabled
                        else "Axial-symmetry fast path fallback"
                    ),
                    0.05,
                    {
                        "axial_fastpath_enabled": bool(self._axial_radial_fastpath_enabled),
                        "axial_fastpath_reason": str(self._axial_radial_fastpath_reason),
                        "radial_bins": int(self._radial_bins),
                    },
                )
        else:
            self._axial_radial_fastpath = None
            self._axial_radial_fastpath_enabled = False
            self._axial_radial_fastpath_reason = "disabled: axial symmetry off"
            if progress_callback is not None:
                progress_callback(
                    "Preparing full-2D precomputed field",
                    0.03,
                    {
                        "axial_fastpath_enabled": False,
                        "axial_fastpath_reason": str(self._axial_radial_fastpath_reason),
                        "radial_bins": int(self._radial_bins),
                    },
                )
            self._initialize_linear_fastpath(
                base_field_after_fresnel=base_field_after_fresnel,
                pupil_mask=pupil_mask,
                stop_callback=stop_callback,
            )
            if progress_callback is not None:
                progress_callback(
                    (
                        "Full-2D linear fast path enabled"
                        if self._linear_fastpath_enabled
                        else "Full-2D linear fast path fallback"
                    ),
                    0.05,
                    {
                        "axial_fastpath_enabled": False,
                        "axial_fastpath_reason": str(self._axial_radial_fastpath_reason),
                        "linear_fastpath_enabled": bool(self._linear_fastpath_enabled),
                        "linear_fastpath_reason": str(self._linear_fastpath_reason),
                        "radial_bins": int(self._radial_bins),
                    },
                )

        baseline_amp = np.ones(full_shape, dtype=np.float64)
        baseline_phase = np.zeros(full_shape, dtype=np.float64)
        baseline_t0 = time.time()
        baseline_obj, baseline_metrics, baseline_res, _, _ = self._simulate_from_mask_maps(
            base_field_after_fresnel=base_field_after_fresnel,
            pupil_mask=pupil_mask,
            amp_var=baseline_amp,
            phase_var=baseline_phase,
            full_shape=full_shape,
            axial_radial_fastpath_ok=bool(self._config.axial_symmetry),
            stop_callback=stop_callback,
        )
        baseline_eval_s = max(1e-9, float(time.time() - baseline_t0))

        if progress_callback is not None:
            progress_callback(
                "Baseline computed",
                0.06,
                {
                    "objective": float(baseline_obj),
                    "width_x_um": float(baseline_metrics["width_x_um"]),
                    "width_y_um": float(baseline_metrics["width_y_um"]),
                    "width_z_um": float(baseline_metrics["width_z_um"]),
                    "abs_x_minus_y_um": float(baseline_metrics.get("abs_x_minus_y_um", np.nan)),
                    "sidelobe_penalty": float(baseline_metrics.get("sidelobe_penalty", np.nan)),
                    "objective_eval_s": float(baseline_eval_s),
                },
            )

        def emit_stage_cost_hint(stage_name: str, stage_shape: Tuple[int, int], frac: float):
            if progress_callback is None:
                return
            support = self._pupil_support_for_shape(
                pupil_mask=pupil_mask,
                var_shape=stage_shape,
                threshold=0.5,
            )
            n_support_pixels = int(np.count_nonzero(support))
            use_simple_ring = bool(
                str(self._config.mode).strip().lower() == "amplitude"
                and bool(self._config.binary_amplitude)
                and str(getattr(self._config, "binary_structure", "pixel")).strip().lower()
                == "simple_ring_block"
            )
            use_simple_ellipse = bool(
                str(self._config.mode).strip().lower() == "amplitude"
                and bool(self._config.binary_amplitude)
                and str(getattr(self._config, "binary_structure", "pixel")).strip().lower()
                == "simple_ellipse_block"
            )
            if use_simple_ring:
                n_vars = 2
            elif use_simple_ellipse:
                n_vars = 4
            elif bool(self._config.axial_symmetry):
                radial = self._build_radial_parameterization(
                    stage_shape,
                    support_mask=support,
                    n_bins=int(self._radial_bins),
                )
                n_active_bins = int(np.count_nonzero(np.asarray(radial["support_by_bin"], dtype=bool)))
                n_vars = int(n_active_bins)
                if self._config.mode == "amplitude_phase":
                    n_vars = 2 * int(n_vars)
            else:
                n_vars = int(n_support_pixels)
                if self._config.mode == "amplitude_phase":
                    n_vars = 2 * int(n_vars)
            surrogate_enabled = bool(
                (not bool(self._config.axial_symmetry))
                and str(self._config.mode).strip().lower() == "amplitude"
                and (not bool(self._config.binary_amplitude))
                and str(getattr(self._config, "binary_structure", "pixel")).strip().lower() == "pixel"
                and bool(self._linear_fastpath_enabled)
            )
            effective_optimizer = str(self._config.optimizer_name)
            if surrogate_enabled:
                if effective_optimizer not in {"L-BFGS-B", "SLSQP"}:
                    effective_optimizer = "L-BFGS-B"
                est_per_iter = 1.0
            else:
                est_per_iter = self._estimate_evals_per_iteration(
                    optimizer_name=effective_optimizer,
                    n_variables=n_vars,
                )
            progress_callback(
                f"{stage_name}: estimated objective-eval cost",
                float(frac),
                {
                    "n_variables": int(n_vars),
                    "n_support_pixels": int(n_support_pixels),
                    "est_evals_per_iter": float(est_per_iter),
                    "objective_eval_s": float(baseline_eval_s),
                    "est_iter_s": float(est_per_iter * baseline_eval_s),
                    "effective_optimizer_name": str(effective_optimizer),
                    "surrogate_loss_enabled": bool(surrogate_enabled),
                    "axial_symmetry": bool(self._config.axial_symmetry),
                    "binary_structure": str(getattr(self._config, "binary_structure", "pixel")),
                },
            )

        emit_stage_cost_hint("Optimization stage", full_shape, 0.07)
        final_x, final_obj, final_metrics, final_res, final_amp, final_phase = self._run_single_stage(
            stage_name="Optimization stage",
            base_field_after_fresnel=base_field_after_fresnel,
            pupil_mask=pupil_mask,
            mode=self._config.mode,
            optimizer_name=self._config.optimizer_name,
            var_shape=full_shape,
            full_shape=full_shape,
            n_starts=self._config.n_starts,
            maxiter=self._config.maxiter,
            ftol=self._config.ftol,
            gtol=self._config.gtol,
            seed=self._config.seed,
            x_seed=None,
            progress_callback=(
                None
                if progress_callback is None
                else lambda stage, frac, info: progress_callback(
                    stage, 0.08 + 0.90 * float(frac), info
                )
            ),
            stop_callback=stop_callback,
        )

        if final_x is None or final_metrics is None or final_res is None:
            raise RuntimeError("Optimization did not return a final solution.")

        amp_out = np.asarray(final_amp, dtype=np.float64)
        phase_out = np.asarray(final_phase, dtype=np.float64)
        amp_out = np.clip(amp_out, 0.0, 1.0)
        phase_out = np.clip(phase_out, -np.pi, np.pi)
        if bool(self._config.binary_amplitude and self._config.mode == "amplitude"):
            amp_out = np.asarray(amp_out >= 0.5, dtype=np.float64)
            phase_out = np.zeros_like(amp_out, dtype=np.float64)
        complex_out = amp_out * np.exp(1j * phase_out)
        use_simple_ring = bool(
            str(self._config.mode).strip().lower() == "amplitude"
            and bool(self._config.binary_amplitude)
            and str(getattr(self._config, "binary_structure", "pixel")).strip().lower()
            == "simple_ring_block"
        )
        use_simple_ellipse = bool(
            str(self._config.mode).strip().lower() == "amplitude"
            and bool(self._config.binary_amplitude)
            and str(getattr(self._config, "binary_structure", "pixel")).strip().lower()
            == "simple_ellipse_block"
        )

        settings = {
            "optimizer_name": self._config.optimizer_name,
            "optimizer_category": (
                str(self.get_optimizer_category_for_name(self._config.optimizer_name) or "Uncategorized")
            ),
            "mode": self._config.mode,
            "width_level": float(self._config.width_level),
            "weights_xyz": [float(v) for v in self._config.weights_xyz],
            "weight_abs_xy": float(getattr(self._config, "weight_abs_xy", 0.0)),
            "n_starts": int(self._config.n_starts),
            "maxiter": int(self._config.maxiter),
            "ftol": float(self._config.ftol),
            "gtol": float(self._config.gtol),
            "seed": None if self._config.seed is None else int(self._config.seed),
            "axial_symmetry": bool(self._config.axial_symmetry),
            "binary_amplitude": bool(self._config.binary_amplitude),
            "binary_structure": str(getattr(self._config, "binary_structure", "pixel")),
            "binary_anneal_alpha_start": float(self._config.binary_anneal_alpha_start),
            "binary_anneal_alpha_end": float(self._config.binary_anneal_alpha_end),
            "sidelobe_penalty_weight": float(self._config.sidelobe_penalty_weight),
            "radial_bins": int(self._radial_bins),
            "axial_fastpath_enabled": bool(self._axial_radial_fastpath_enabled),
            "axial_fastpath_reason": str(self._axial_radial_fastpath_reason),
            "linear_fastpath_enabled": bool(self._linear_fastpath_enabled),
            "linear_fastpath_reason": str(self._linear_fastpath_reason),
            "linear_fastpath_mode": (
                None
                if self._linear_fastpath is None
                else str(getattr(self._linear_fastpath, "precompute_mode", "factorized"))
            ),
            "surrogate_loss_enabled": bool(self._surrogate_loss_used),
            "surrogate_loss_reason": str(self._surrogate_loss_reason),
            "surrogate_optimizer_name": (
                None
                if self._surrogate_optimizer_name is None
                else str(self._surrogate_optimizer_name)
            ),
            "amplitude_fastpath_enabled": bool(self._linear_fastpath_enabled),
            "amplitude_fastpath_reason": str(self._linear_fastpath_reason),
            "active_axes": {
                "x": bool(self._need_x),
                "y": bool(self._need_y),
                "z": bool(self._need_z),
            },
            "computed_planes": {
                "xy": bool(self._compute_xy),
                "xz": bool(self._compute_xz),
                "yz": bool(self._compute_yz),
            },
        }
        if use_simple_ring and final_x is not None and int(np.asarray(final_x).size) >= 2:
            ring_inner_norm = float(min(float(final_x[0]), float(final_x[1])))
            ring_outer_norm = float(max(float(final_x[0]), float(final_x[1])))
            ring_inner_m = float(ring_inner_norm * 0.5 * float(self.params.get("pupil_d", 0.0)))
            ring_outer_m = float(ring_outer_norm * 0.5 * float(self.params.get("pupil_d", 0.0)))
            settings["ring_inner_radius_norm"] = ring_inner_norm
            settings["ring_outer_radius_norm"] = ring_outer_norm
            settings["ring_inner_radius_m"] = ring_inner_m
            settings["ring_outer_radius_m"] = ring_outer_m
            settings["ring_inner_radius_mm"] = float(ring_inner_m * 1e3)
            settings["ring_outer_radius_mm"] = float(ring_outer_m * 1e3)
        elif use_simple_ellipse and final_x is not None and int(np.asarray(final_x).size) >= 4:
            pupil_d_m = float(self.params.get("pupil_d", 0.0))
            ellipse_inner_dx_norm = float(np.clip(final_x[0], 0.0, 1.0))
            ellipse_inner_dy_norm = float(np.clip(final_x[1], 0.0, 1.0))
            ellipse_outer_dx_norm = float(np.clip(final_x[2], 0.0, 1.0))
            ellipse_outer_dy_norm = float(np.clip(final_x[3], 0.0, 1.0))
            settings["ellipse_inner_diameter_x_norm"] = ellipse_inner_dx_norm
            settings["ellipse_inner_diameter_y_norm"] = ellipse_inner_dy_norm
            settings["ellipse_outer_diameter_x_norm"] = ellipse_outer_dx_norm
            settings["ellipse_outer_diameter_y_norm"] = ellipse_outer_dy_norm
            settings["ellipse_inner_diameter_x_m"] = float(ellipse_inner_dx_norm * pupil_d_m)
            settings["ellipse_inner_diameter_y_m"] = float(ellipse_inner_dy_norm * pupil_d_m)
            settings["ellipse_outer_diameter_x_m"] = float(ellipse_outer_dx_norm * pupil_d_m)
            settings["ellipse_outer_diameter_y_m"] = float(ellipse_outer_dy_norm * pupil_d_m)
            settings["ellipse_inner_diameter_x_mm"] = float(ellipse_inner_dx_norm * pupil_d_m * 1e3)
            settings["ellipse_inner_diameter_y_mm"] = float(ellipse_inner_dy_norm * pupil_d_m * 1e3)
            settings["ellipse_outer_diameter_x_mm"] = float(ellipse_outer_dx_norm * pupil_d_m * 1e3)
            settings["ellipse_outer_diameter_y_mm"] = float(ellipse_outer_dy_norm * pupil_d_m * 1e3)

        if progress_callback is not None:
            progress_callback("Optimization complete", 1.0, {"objective": float(final_obj)})

        return OptimizerResult(
            best_objective=float(final_obj),
            baseline_objective=float(baseline_obj),
            best_metrics_um={
                "width_x_um": float(final_metrics["width_x_um"]),
                "width_y_um": float(final_metrics["width_y_um"]),
                "width_z_um": float(final_metrics["width_z_um"]),
                "abs_x_minus_y_um": float(final_metrics.get("abs_x_minus_y_um", np.nan)),
            },
            baseline_metrics_um={
                "width_x_um": float(baseline_metrics["width_x_um"]),
                "width_y_um": float(baseline_metrics["width_y_um"]),
                "width_z_um": float(baseline_metrics["width_z_um"]),
                "abs_x_minus_y_um": float(baseline_metrics.get("abs_x_minus_y_um", np.nan)),
            },
            amplitude_mask=amp_out,
            phase_mask_rad=phase_out,
            complex_mask=np.asarray(complex_out, dtype=np.complex128),
            history=list(self.history),
            settings=settings,
            baseline_result_2d=baseline_res,
            optimized_result_2d=final_res,
        )


def save_result_npz(path: str, result: OptimizerResult, extra_metadata: Optional[Dict[str, Any]] = None):
    metadata = {
        "settings": result.settings,
        "best_objective": float(result.best_objective),
        "baseline_objective": float(result.baseline_objective),
        "best_metrics_um": dict(result.best_metrics_um),
        "baseline_metrics_um": dict(result.baseline_metrics_um),
        "history_length": int(len(result.history)),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    np.savez_compressed(
        path,
        amplitude_mask=np.asarray(result.amplitude_mask, dtype=np.float64),
        phase_mask_rad=np.asarray(result.phase_mask_rad, dtype=np.float64),
        complex_mask_real=np.asarray(np.real(result.complex_mask), dtype=np.float64),
        complex_mask_imag=np.asarray(np.imag(result.complex_mask), dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )
