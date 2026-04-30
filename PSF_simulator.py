import argparse
import hashlib
import threading
import traceback
from collections import OrderedDict
from math import asin, isclose, isfinite, log2, pi, prod

import numpy as np
import torch
import torch.nn.functional as F

from plot_style import (
    add_compact_colorbar,
    add_panel_labels,
    apply_publication_style,
    style_image_axis,
    style_profile_axis,
)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:  # pragma: no cover - optional for headless runs
    tk = None
    filedialog = None
    messagebox = None
    ttk = None


apply_publication_style()


class SimulationStoppedError(RuntimeError):
    """Raised when simulation is cancelled by a stop callback."""


_FRESNEL_KERNEL_CACHE = OrderedDict()
_FRESNEL_KERNEL_CACHE_MAX = 16


def _resolve_torch_device(device=None):
    if device is None or str(device).strip().lower() in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available.")
    return dev


def _torch_float64(arr, device):
    if torch.is_tensor(arr):
        return arr.to(device=device, dtype=torch.float64)
    return torch.as_tensor(arr, dtype=torch.float64, device=device)


def _torch_complex128(arr, device):
    if torch.is_tensor(arr):
        return arr.to(device=device, dtype=torch.complex128)
    return torch.as_tensor(arr, dtype=torch.complex128, device=device)


def _torch_linspace_no_endpoint(start, stop, steps, device, dtype=torch.float64):
    steps = int(steps)
    if steps <= 0:
        return torch.empty((0,), dtype=dtype, device=device)
    delta = (float(stop) - float(start)) / float(steps)
    idx = torch.arange(steps, device=device, dtype=dtype)
    return idx * delta + float(start)


def _torch_to_numpy(arr):
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _fresnel_kernel_cache_key(nx, ny, dx, dy, z, wavelength, device):
    return (
        str(device),
        int(nx),
        int(ny),
        float(np.float64(dx)),
        float(np.float64(dy)),
        float(np.float64(z)),
        float(np.float64(wavelength)),
    )


def _get_fresnel_transfer_kernel(nx, ny, dx, dy, z, wavelength, device):
    key = _fresnel_kernel_cache_key(nx, ny, dx, dy, z, wavelength, device)
    cached = _FRESNEL_KERNEL_CACHE.get(key)
    if cached is not None:
        _FRESNEL_KERNEL_CACHE.move_to_end(key)
        return cached

    k = 2.0 * pi / float(wavelength)
    fx = torch.fft.fftfreq(int(nx), d=float(dx), device=device, dtype=torch.float64)
    fy = torch.fft.fftfreq(int(ny), d=float(dy), device=device, dtype=torch.float64)
    kx = 2.0 * pi * fx[:, None]
    ky = 2.0 * pi * fy[None, :]
    kz = torch.sqrt((k**2 - kx**2 - ky**2).to(torch.complex128))
    h = torch.exp(1j * kz * float(z))

    _FRESNEL_KERNEL_CACHE[key] = h
    _FRESNEL_KERNEL_CACHE.move_to_end(key)
    while len(_FRESNEL_KERNEL_CACHE) > int(_FRESNEL_KERNEL_CACHE_MAX):
        _FRESNEL_KERNEL_CACHE.popitem(last=False)
    return h


def propagate_fresnel_field(intensity, size_x, size_y, z, wavelength, device=None, return_torch=False):
    """
    Propagate a 2D input field with exact scalar free-space propagation.

    The input is a normalized intensity map in [0, 1]. The function assumes
    zero initial phase, i.e. U0 = sqrt(intensity), and returns the propagated
    complex field U(x, y, z).

    Parameters
    ----------
    intensity : (nx, ny) ndarray
        2D intensity array normalized to [0, 1].
    size_x : float
        Physical size of the grid along x (meters).
    size_y : float
        Physical size of the grid along y (meters).
    z : float
        Propagation distance (meters).
    wavelength : float
        Wavelength (meters).

    Returns
    -------
    propagated_field : (nx, ny) complex ndarray
        Complex propagated field at distance z.
    """
    dev = _resolve_torch_device(device)
    intensity_t = _torch_float64(intensity, device=dev)

    if intensity_t.ndim != 2:
        raise ValueError("intensity must be a 2D array.")
    if intensity_t.numel() == 0:
        raise ValueError("intensity cannot be empty.")
    if size_x <= 0 or size_y <= 0:
        raise ValueError("size_x and size_y must be positive.")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")

    min_i = float(torch.min(intensity_t).item())
    max_i = float(torch.max(intensity_t).item())
    if min_i < 0.0 or max_i > 1.0:
        raise ValueError("intensity values must be normalized within [0, 1].")

    u0 = torch.sqrt(intensity_t).to(torch.complex128)

    if isclose(z, 0.0):
        return u0 if return_torch else _torch_to_numpy(u0)

    nx, ny = intensity_t.shape
    dx = size_x / nx
    dy = size_y / ny

    h = _get_fresnel_transfer_kernel(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        z=z,
        wavelength=wavelength,
        device=dev,
    )

    u0_f = torch.fft.fft2(u0)
    uz = torch.fft.ifft2(u0_f * h)
    return uz if return_torch else _torch_to_numpy(uz)


def apply_gaussian_profile(intensity, size_x, size_y, wx, wy, device=None, return_torch=False):
    """
    Apply a centered Gaussian profile to a 2D intensity array.

    The Gaussian factor is:
        G(x, y) = exp(-((x / wx)^2 + (y / wy)^2))

    Parameters
    ----------
    intensity : (nx, ny) ndarray
        Input 2D intensity array.
    size_x : float
        Physical size along x (meters).
    size_y : float
        Physical size along y (meters).
    wx : float
        Gaussian width along x (meters).
    wy : float
        Gaussian width along y (meters).

    Returns
    -------
    processed_intensity : (nx, ny) ndarray
        Intensity after Gaussian weighting.
    """
    dev = _resolve_torch_device(device)
    intensity_t = _torch_float64(intensity, device=dev)

    if intensity_t.ndim != 2:
        raise ValueError("intensity must be a 2D array.")
    if intensity_t.numel() == 0:
        raise ValueError("intensity cannot be empty.")
    if size_x <= 0 or size_y <= 0:
        raise ValueError("size_x and size_y must be positive.")
    if wx <= 0 or wy <= 0:
        raise ValueError("wx and wy must be positive.")

    nx, ny = intensity_t.shape
    x = _torch_linspace_no_endpoint(-size_x / 2, size_x / 2, nx, device=dev)
    y = _torch_linspace_no_endpoint(-size_y / 2, size_y / 2, ny, device=dev)
    xg, yg = torch.meshgrid(x, y, indexing="ij")

    gaussian = torch.exp(-((xg / wx) ** 2 + (yg / wy) ** 2))
    out = intensity_t * gaussian
    return out if return_torch else _torch_to_numpy(out)


class CartesianPSFSimulator:
    """
    Cartesian PSF simulator following PSF_x_z_cartesian.m equations.

    Processing chain:
    1) input intensity -> optional Gaussian envelope
    2) optional custom/ring mask
    3) Fresnel propagation
    4) entrance pupil diameter mask
    5) vectorial objective PSF calculation on XZ, YZ, and XY planes
    """

    def __init__(
        self,
        wavelength=780e-9,
        NA=1.4,
        theta_min=0.0,
        theta_samples=81,
        phi_samples=160,
        t_gd=0.17e-3,
        n_gd=1.515,
        t_id=0.09e-3,
        n_id=1.515,
        n_sd=1.515,
        t_g=0.17e-3,
        n_g=1.515,
        t_s=1e-6,
        n_s=1.474,
        n_i=1.515,
        integration_mode="cartesian_jacobian",
        torch_device="cpu",
        trim_pupil_for_psf=True,
        pupil_trim_margin_pixels=2,
    ):
        self.device = _resolve_torch_device(torch_device)
        self.wavelength = float(wavelength)
        self.NA = float(NA)
        self.theta_min = float(theta_min)
        self.theta_samples = int(theta_samples)
        self.phi_samples = int(phi_samples)

        self.t_gd = float(t_gd)
        self.n_gd = float(n_gd)
        self.t_id = float(t_id)
        self.n_id = float(n_id)
        self.n_sd = float(n_sd)

        self.t_g = float(t_g)
        self.n_g = float(n_g)
        self.t_s = float(t_s)
        self.n_s = float(n_s)
        self.n_i = float(n_i)
        self.integration_mode = str(integration_mode).strip().lower()
        self.trim_pupil_for_psf = bool(trim_pupil_for_psf)
        self.pupil_trim_margin_pixels = int(pupil_trim_margin_pixels)

        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive.")
        if self.theta_samples < 3:
            raise ValueError("theta_samples must be >= 3.")
        if self.phi_samples < 8:
            raise ValueError("phi_samples must be >= 8.")
        if self.NA <= 0:
            raise ValueError("NA must be positive.")
        if self.NA > self.n_i:
            raise ValueError("NA must be <= n_i for theta_max = asin(NA/n_i).")
        if self.integration_mode not in {"cartesian_jacobian", "polar_interp"}:
            raise ValueError("integration_mode must be 'cartesian_jacobian' or 'polar_interp'.")
        if self.pupil_trim_margin_pixels < 0:
            raise ValueError("pupil_trim_margin_pixels must be >= 0.")

        self.k = 2.0 * pi / self.wavelength
        self.theta_max_nominal = asin(self.NA / self.n_i)
        # Effective angular support is limited by any lower-index downstream medium.
        self.NA_eff = min(self.NA, self.n_i, self.n_g, self.n_s)
        self.theta_max = asin(min(1.0, max(-1.0, self.NA_eff / self.n_i)))
        if self.theta_min < 0:
            raise ValueError("theta_min must be >= 0.")
        if self.theta_min >= self.theta_max:
            raise ValueError("theta_min must be < theta_max after effective NA limiting.")

        # MATLAB line 21: required immersion oil thickness for paraxial focus.
        self.t_i = self.t_s + self.n_i * (
            -(self.t_s / self.n_s)
            - (self.t_g / self.n_g)
            + (self.t_gd / self.n_gd)
            + (self.t_id / self.n_id)
        )

        self.h_1 = self.t_s + self.t_g
        self.h_2 = self.t_s
        self.h_1d = self.t_gd

        self.C = torch.tensor(
            -1j * self.n_i * self.k / (2.0 * pi),
            dtype=torch.complex128,
            device=self.device,
        )

        # Small LRU caches for expensive precomputations.
        self._geometry_cache = OrderedDict()
        self._geometry_cache_max = 8
        self._integration_table_cache = OrderedDict()
        self._integration_table_cache_max = 8

    def _to_torch_float(self, arr):
        return _torch_float64(arr, device=self.device)

    def _to_torch_complex(self, arr):
        return _torch_complex128(arr, device=self.device)

    @staticmethod
    def _to_numpy(arr):
        return _torch_to_numpy(arr)

    @staticmethod
    def _normalize_intensity(arr):
        if torch.is_tensor(arr):
            max_val = torch.max(arr)
            if float(max_val.item()) <= 0.0:
                return arr
            return arr / max_val
        arr_np = np.asarray(arr)
        max_val = float(np.max(arr_np))
        if max_val <= 0:
            return arr_np
        return arr_np / max_val

    @staticmethod
    def _array_hash(arr):
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().contiguous().numpy()
        else:
            arr = np.ascontiguousarray(arr)
        return hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()

    @staticmethod
    def _float_key(value):
        return float(np.float64(value))

    @staticmethod
    def _resolve_axis_ranges_m(
        range_microns=2.0,
        range_microns_x=None,
        range_microns_y=None,
        range_microns_z=None,
    ):
        base_um = float(range_microns)
        rx_um = base_um if range_microns_x is None else float(range_microns_x)
        ry_um = base_um if range_microns_y is None else float(range_microns_y)
        rz_um = base_um if range_microns_z is None else float(range_microns_z)

        for value, axis in ((rx_um, "x"), (ry_um, "y"), (rz_um, "z")):
            if (not np.isfinite(value)) or value <= 0.0:
                raise ValueError(f"PSF range_{axis} must be finite and > 0 microns.")

        return rx_um * 1e-6, ry_um * 1e-6, rz_um * 1e-6

    def _lru_get(self, cache, key):
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    @staticmethod
    def _lru_put(cache, key, value, max_size):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > int(max_size):
            cache.popitem(last=False)

    def _table_base_key(self, mode, shape, size_x, size_y, entrance_pupil_diameter):
        return (
            str(mode),
            tuple(int(v) for v in shape),
            self._float_key(size_x),
            self._float_key(size_y),
            self._float_key(entrance_pupil_diameter),
        )

    def _integration_table_cache_key(
        self, pupil_field, mode, size_x, size_y, entrance_pupil_diameter
    ):
        if torch.is_tensor(pupil_field):
            arr = pupil_field.detach().cpu().numpy().astype(np.complex128, copy=False)
        else:
            arr = np.asarray(pupil_field, dtype=np.complex128)
        base = self._table_base_key(mode, arr.shape, size_x, size_y, entrance_pupil_diameter)
        # Hash complex values directly; _array_hash enforces contiguity before byte-viewing.
        field_hash = self._array_hash(arr)
        return ("table",) + base + (field_hash,)

    @staticmethod
    def _centered_pupil_trim_window(
        shape,
        size_x,
        size_y,
        entrance_pupil_diameter,
        margin_pixels=2,
    ):
        nx, ny = int(shape[0]), int(shape[1])
        if nx <= 0 or ny <= 0:
            raise ValueError("shape values must be positive.")
        if size_x <= 0 or size_y <= 0:
            raise ValueError("size_x and size_y must be positive.")
        if entrance_pupil_diameter <= 0:
            raise ValueError("entrance_pupil_diameter must be positive.")

        dx = float(size_x) / float(nx)
        dy = float(size_y) / float(ny)
        margin = max(0, int(margin_pixels))
        target_span_m = float(entrance_pupil_diameter) + 2.0 * float(margin) * max(dx, dy)
        n_target_x = int(np.ceil(target_span_m / max(dx, 1e-30)))
        n_target_y = int(np.ceil(target_span_m / max(dy, 1e-30)))

        def fit_count(n_total, n_target):
            n_keep = max(3, min(int(n_total), int(n_target)))
            if (n_keep % 2) != (int(n_total) % 2):
                if n_keep + 1 <= int(n_total):
                    n_keep += 1
                elif n_keep - 1 >= 3:
                    n_keep -= 1
            return max(3, min(int(n_total), int(n_keep)))

        n_keep_x = fit_count(nx, n_target_x)
        n_keep_y = fit_count(ny, n_target_y)
        row0 = int((nx - n_keep_x) // 2)
        col0 = int((ny - n_keep_y) // 2)
        row1 = int(row0 + n_keep_x)
        col1 = int(col0 + n_keep_y)

        size_x_keep = float(dx * n_keep_x)
        size_y_keep = float(dy * n_keep_y)
        trimmed = bool(n_keep_x < nx or n_keep_y < ny)

        return {
            "row0": row0,
            "row1": row1,
            "col0": col0,
            "col1": col1,
            "size_x": size_x_keep,
            "size_y": size_y_keep,
            "trimmed": trimmed,
        }

    @staticmethod
    def _width_at_level_1d(x, y, level=0.5):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        if x.size != y.size or x.size < 3:
            return np.nan

        max_idx = int(np.argmax(y))
        max_y = float(y[max_idx])
        if not np.isfinite(max_y) or max_y <= 0:
            return np.nan

        level = float(level)
        level = min(1.0, max(0.0, level))
        target = level * max_y

        left_candidates = np.where(y[: max_idx + 1] <= target)[0]
        if left_candidates.size == 0:
            return np.nan
        left_idx = int(left_candidates[-1])
        if left_idx >= max_idx:
            return np.nan

        right_candidates = np.where(y[max_idx:] <= target)[0]
        if right_candidates.size == 0:
            return np.nan
        right_idx = int(max_idx + right_candidates[0])
        if right_idx <= max_idx:
            return np.nan

        x1_l, y1_l = x[left_idx], y[left_idx]
        x2_l, y2_l = x[left_idx + 1], y[left_idx + 1]
        if y2_l == y1_l:
            x_left = 0.5 * (x1_l + x2_l)
        else:
            x_left = x1_l + (target - y1_l) * (x2_l - x1_l) / (y2_l - y1_l)

        x1_r, y1_r = x[right_idx - 1], y[right_idx - 1]
        x2_r, y2_r = x[right_idx], y[right_idx]
        if y2_r == y1_r:
            x_right = 0.5 * (x1_r + x2_r)
        else:
            x_right = x1_r + (target - y1_r) * (x2_r - x1_r) / (y2_r - y1_r)

        return float(x_right - x_left)

    @staticmethod
    def _fwhm_1d(x, y):
        return CartesianPSFSimulator._width_at_level_1d(x, y, level=0.5)

    @staticmethod
    def _safe_width_ratio(numerator, denominator):
        num = float(numerator)
        den = float(denominator)
        if (not np.isfinite(num)) or (not np.isfinite(den)) or den == 0.0:
            return np.nan
        return num / den

    @staticmethod
    def _pack_peak_width_metrics(
        *,
        level,
        x_profile,
        y_profile,
        z_profile,
        width_x_m,
        width_y_m,
        width_z_m,
        peak_x_index,
        peak_y_index,
    ):
        return {
            "level": float(level),
            "x_profile_peak": np.asarray(x_profile, dtype=float),
            "y_profile_peak": np.asarray(y_profile, dtype=float),
            "z_profile_peak": np.asarray(z_profile, dtype=float),
            "width_x_m": float(width_x_m),
            "width_y_m": float(width_y_m),
            "width_z_m": float(width_z_m),
            "width_x_um": float(width_x_m * 1e6) if np.isfinite(width_x_m) else np.nan,
            "width_y_um": float(width_y_m * 1e6) if np.isfinite(width_y_m) else np.nan,
            "width_z_um": float(width_z_m * 1e6) if np.isfinite(width_z_m) else np.nan,
            "width_ratio_zx": CartesianPSFSimulator._safe_width_ratio(width_z_m, width_x_m),
            "width_ratio_zy": CartesianPSFSimulator._safe_width_ratio(width_z_m, width_y_m),
            "peak_x_index": int(peak_x_index),
            "peak_y_index": int(peak_y_index),
        }

    @staticmethod
    def _extract_peak_width_profiles(i_xy_norm, i_xz_norm, i_yz_norm=None):
        i_xy_arr = np.asarray(i_xy_norm, dtype=float)
        i_xz_arr = np.asarray(i_xz_norm, dtype=float)
        i_yz_arr = None if i_yz_norm is None else np.asarray(i_yz_norm, dtype=float)

        peak_y_idx, peak_x_idx = np.unravel_index(int(np.argmax(i_xy_arr)), i_xy_arr.shape)
        x_profile = np.asarray(i_xy_arr[peak_y_idx, :], dtype=float)
        use_yz = (
            i_yz_arr is not None
            and i_yz_arr.ndim == 2
            and i_yz_arr.shape[0] > 0
            and i_yz_arr.shape[1] > 0
        )
        if use_yz:
            peak_z_idx_yz, peak_y_idx_yz = np.unravel_index(int(np.argmax(i_yz_arr)), i_yz_arr.shape)
            y_profile = np.asarray(i_yz_arr[peak_z_idx_yz, :], dtype=float)
            z_profile = np.asarray(i_yz_arr[:, peak_y_idx_yz], dtype=float)
        else:
            y_profile = np.asarray(i_xy_arr[:, peak_x_idx], dtype=float)
            x_idx_for_z = min(max(int(peak_x_idx), 0), i_xz_arr.shape[1] - 1)
            z_profile = np.asarray(i_xz_arr[:, x_idx_for_z], dtype=float)

        return {
            "x_profile_peak": x_profile,
            "y_profile_peak": y_profile,
            "z_profile_peak": z_profile,
            "peak_x_index": int(peak_x_idx),
            "peak_y_index": int(peak_y_idx),
        }

    def _compute_fwhm_metrics(self, x_coords, y_coords, z_coords, i_xy_norm, i_xz_norm):
        center_idx = int(len(x_coords) // 2)

        x_profile = i_xy_norm[center_idx, :]
        y_profile = i_xy_norm[:, center_idx]
        z_profile = i_xz_norm[:, center_idx]

        fwhm_x_m = self._fwhm_1d(x_coords, x_profile)
        fwhm_y_m = self._fwhm_1d(y_coords, y_profile)
        fwhm_z_m = self._fwhm_1d(z_coords, z_profile)

        return {
            "center_index": center_idx,
            "x_profile_at_y0": x_profile,
            "y_profile_at_x0": y_profile,
            "z_profile_at_x0": z_profile,
            "fwhm_x_m": fwhm_x_m,
            "fwhm_y_m": fwhm_y_m,
            "fwhm_z_m": fwhm_z_m,
            "fwhm_x_um": float(fwhm_x_m * 1e6) if np.isfinite(fwhm_x_m) else np.nan,
            "fwhm_y_um": float(fwhm_y_m * 1e6) if np.isfinite(fwhm_y_m) else np.nan,
            "fwhm_z_um": float(fwhm_z_m * 1e6) if np.isfinite(fwhm_z_m) else np.nan,
        }

    def compute_peak_width_metrics(
        self, x_coords, y_coords, z_coords, i_xy_norm, i_xz_norm, i_yz_norm=None, level=0.5
    ):
        """
        Compute width metrics at an arbitrary normalized level around peak profiles.

        Parameters
        ----------
        x_coords, y_coords, z_coords : ndarray
            Coordinate axes in meters.
        i_xy_norm, i_xz_norm : ndarray
            Normalized XY and XZ intensity maps in [0, 1].
        i_yz_norm : ndarray or None
            Optional normalized YZ map. If present, Y/Z peak profiles are taken
            from this plane; otherwise Y comes from XY and Z comes from XZ.
        level : float
            Width level in [0, 1], where 0.5 is classic FWHM.
        """
        level = float(level)
        level = min(1.0, max(0.0, level))

        peak_profiles = self._extract_peak_width_profiles(
            i_xy_norm=i_xy_norm,
            i_xz_norm=i_xz_norm,
            i_yz_norm=i_yz_norm,
        )
        return self.peak_width_metrics_from_profiles(
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
            x_profile=peak_profiles["x_profile_peak"],
            y_profile=peak_profiles["y_profile_peak"],
            z_profile=peak_profiles["z_profile_peak"],
            level=level,
            peak_x_index=int(peak_profiles["peak_x_index"]),
            peak_y_index=int(peak_profiles["peak_y_index"]),
        )

    def peak_width_metrics_from_profiles(
        self,
        x_coords,
        y_coords,
        z_coords,
        x_profile,
        y_profile,
        z_profile,
        level=0.5,
        peak_x_index=0,
        peak_y_index=0,
    ):
        level = float(level)
        level = min(1.0, max(0.0, level))

        x_profile = np.asarray(x_profile, dtype=float)
        y_profile = np.asarray(y_profile, dtype=float)
        z_profile = np.asarray(z_profile, dtype=float)

        width_x_m = self._width_at_level_1d(x_coords, x_profile, level=level)
        width_y_m = self._width_at_level_1d(y_coords, y_profile, level=level)
        width_z_m = self._width_at_level_1d(z_coords, z_profile, level=level)
        return self._pack_peak_width_metrics(
            level=level,
            x_profile=x_profile,
            y_profile=y_profile,
            z_profile=z_profile,
            width_x_m=width_x_m,
            width_y_m=width_y_m,
            width_z_m=width_z_m,
            peak_x_index=peak_x_index,
            peak_y_index=peak_y_index,
        )

    def precompute_peak_width_metrics(
        self,
        x_coords,
        y_coords,
        z_coords,
        i_xy_norm,
        i_xz_norm,
        i_yz_norm=None,
        level_min=0.01,
        level_max=0.99,
        level_step=0.001,
    ):
        level_min = float(level_min)
        level_max = float(level_max)
        level_step = float(level_step)
        if level_step <= 0.0:
            raise ValueError("level_step must be positive.")
        if level_max < level_min:
            raise ValueError("level_max must be >= level_min.")

        peak_profiles = self._extract_peak_width_profiles(
            i_xy_norm=i_xy_norm,
            i_xz_norm=i_xz_norm,
            i_yz_norm=i_yz_norm,
        )
        x_profile = peak_profiles["x_profile_peak"]
        y_profile = peak_profiles["y_profile_peak"]
        z_profile = peak_profiles["z_profile_peak"]

        step_count = int(round((level_max - level_min) / level_step))
        levels = level_min + level_step * np.arange(step_count + 1, dtype=np.float64)
        levels = np.clip(np.round(levels, 6), level_min, level_max)

        width_x_m = np.empty(levels.shape, dtype=np.float64)
        width_y_m = np.empty(levels.shape, dtype=np.float64)
        width_z_m = np.empty(levels.shape, dtype=np.float64)
        for idx, level_value in enumerate(levels):
            width_x_m[idx] = self._width_at_level_1d(x_coords, x_profile, level=float(level_value))
            width_y_m[idx] = self._width_at_level_1d(y_coords, y_profile, level=float(level_value))
            width_z_m[idx] = self._width_at_level_1d(z_coords, z_profile, level=float(level_value))

        width_ratio_zx = np.full(levels.shape, np.nan, dtype=np.float64)
        width_ratio_zy = np.full(levels.shape, np.nan, dtype=np.float64)
        valid_zx = np.isfinite(width_z_m) & np.isfinite(width_x_m) & (width_x_m != 0.0)
        valid_zy = np.isfinite(width_z_m) & np.isfinite(width_y_m) & (width_y_m != 0.0)
        width_ratio_zx[valid_zx] = width_z_m[valid_zx] / width_x_m[valid_zx]
        width_ratio_zy[valid_zy] = width_z_m[valid_zy] / width_y_m[valid_zy]

        return {
            "levels": levels,
            "level_min": level_min,
            "level_max": level_max,
            "level_step": level_step,
            "x_profile_peak": x_profile,
            "y_profile_peak": y_profile,
            "z_profile_peak": z_profile,
            "width_x_m": width_x_m,
            "width_y_m": width_y_m,
            "width_z_m": width_z_m,
            "width_x_um": width_x_m * 1e6,
            "width_y_um": width_y_m * 1e6,
            "width_z_um": width_z_m * 1e6,
            "width_ratio_zx": width_ratio_zx,
            "width_ratio_zy": width_ratio_zy,
            "peak_x_index": int(peak_profiles["peak_x_index"]),
            "peak_y_index": int(peak_profiles["peak_y_index"]),
        }

    @staticmethod
    def peak_width_metrics_from_precomputed(precomputed, level):
        levels = np.asarray(precomputed["levels"], dtype=float).ravel()
        if levels.size == 0:
            raise ValueError("precomputed width metrics contain no levels.")
        idx = int(np.argmin(np.abs(levels - float(level))))

        return CartesianPSFSimulator._pack_peak_width_metrics(
            level=levels[idx],
            x_profile=precomputed["x_profile_peak"],
            y_profile=precomputed["y_profile_peak"],
            z_profile=precomputed["z_profile_peak"],
            width_x_m=np.asarray(precomputed["width_x_m"], dtype=float)[idx],
            width_y_m=np.asarray(precomputed["width_y_m"], dtype=float)[idx],
            width_z_m=np.asarray(precomputed["width_z_m"], dtype=float)[idx],
            peak_x_index=precomputed["peak_x_index"],
            peak_y_index=precomputed["peak_y_index"],
        )

    @staticmethod
    def apply_user_mask(intensity, mask):
        intensity_t = _torch_float64(intensity, device=torch.device("cpu"))
        mask_t = _torch_float64(mask, device=torch.device("cpu"))
        if tuple(int(v) for v in intensity_t.shape) != tuple(int(v) for v in mask_t.shape):
            raise ValueError("mask shape must match intensity shape.")
        out = intensity_t * mask_t
        return _torch_to_numpy(out)

    @staticmethod
    def apply_mask_offset(mask, size_x, size_y, offset_x=0.0, offset_y=0.0):
        """
        Shift a 2D transmission mask by physical x/y offsets (meters).

        Out-of-bounds pixels created by the shift are filled with 1.0
        (all-pass transmission).
        """
        mask_t = _torch_float64(mask, device=torch.device("cpu"))
        if mask_t.ndim != 2:
            raise ValueError("mask must be a 2D array.")
        if size_x <= 0 or size_y <= 0:
            raise ValueError("size_x and size_y must be positive.")
        if not isfinite(offset_x) or not isfinite(offset_y):
            raise ValueError("offset_x and offset_y must be finite.")

        mask_t = torch.clamp(mask_t, 0.0, 1.0)
        if isclose(offset_x, 0.0) and isclose(offset_y, 0.0):
            return _torch_to_numpy(mask_t)

        nx, ny = int(mask_t.shape[0]), int(mask_t.shape[1])
        dx = float(size_x) / float(nx)
        dy = float(size_y) / float(ny)
        shift_x_px = float(offset_x) / dx
        shift_y_px = float(offset_y) / dy

        row = torch.arange(nx, dtype=torch.float64, device=mask_t.device)
        col = torch.arange(ny, dtype=torch.float64, device=mask_t.device)
        row_grid, col_grid = torch.meshgrid(row, col, indexing="ij")

        src_row = row_grid - shift_x_px
        src_col = col_grid - shift_y_px

        if nx > 1:
            norm_row = (2.0 * src_row / float(nx - 1)) - 1.0
        else:
            norm_row = torch.zeros_like(src_row)
        if ny > 1:
            norm_col = (2.0 * src_col / float(ny - 1)) - 1.0
        else:
            norm_col = torch.zeros_like(src_col)

        grid = torch.stack((norm_col, norm_row), dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            mask_t[None, None, :, :],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]

        valid = (
            (src_row >= 0.0)
            & (src_row <= float(nx - 1))
            & (src_col >= 0.0)
            & (src_col <= float(ny - 1))
        )
        shifted = torch.where(valid, sampled, torch.ones_like(sampled))
        shifted = torch.clamp(shifted, 0.0, 1.0)
        return _torch_to_numpy(shifted)

    @staticmethod
    def create_ring_block_mask(shape, size_x, size_y, inner_diameter, outer_diameter):
        nx, ny = int(shape[0]), int(shape[1])
        if inner_diameter < 0 or outer_diameter <= inner_diameter:
            raise ValueError("Ring diameters must satisfy 0 <= inner < outer.")

        dev = torch.device("cpu")
        x = _torch_linspace_no_endpoint(-size_x / 2, size_x / 2, nx, device=dev)
        y = _torch_linspace_no_endpoint(-size_y / 2, size_y / 2, ny, device=dev)
        xg, yg = torch.meshgrid(x, y, indexing="ij")
        r = torch.sqrt(xg**2 + yg**2)

        ring_inner_radius = 0.5 * inner_diameter
        ring_outer_radius = 0.5 * outer_diameter

        mask = torch.ones((nx, ny), dtype=torch.float64, device=dev)
        ring_region = (r >= ring_inner_radius) & (r <= ring_outer_radius)
        mask[ring_region] = 0.0
        return _torch_to_numpy(mask)

    @staticmethod
    def create_circular_pupil_mask(shape, size_x, size_y, pupil_diameter):
        nx, ny = int(shape[0]), int(shape[1])
        if pupil_diameter <= 0:
            raise ValueError("pupil_diameter must be positive.")

        dev = torch.device("cpu")
        x = _torch_linspace_no_endpoint(-size_x / 2, size_x / 2, nx, device=dev)
        y = _torch_linspace_no_endpoint(-size_y / 2, size_y / 2, ny, device=dev)
        xg, yg = torch.meshgrid(x, y, indexing="ij")
        r = torch.sqrt(xg**2 + yg**2)
        mask = (r <= 0.5 * pupil_diameter).to(dtype=torch.float64)
        return _torch_to_numpy(mask)

    def apply_gaussian_envelope(self, intensity, size_x, size_y, wx, wy):
        as_torch = torch.is_tensor(intensity)
        return apply_gaussian_profile(
            intensity,
            size_x=size_x,
            size_y=size_y,
            wx=wx,
            wy=wy,
            device=self.device,
            return_torch=as_torch,
        )

    def fresnel_propagate(self, intensity, size_x, size_y, z):
        return propagate_fresnel_field(
            intensity,
            size_x=size_x,
            size_y=size_y,
            z=z,
            wavelength=self.wavelength,
            device=self.device,
            return_torch=torch.is_tensor(intensity),
        )

    def apply_entrance_pupil_diameter_mask(
        self, field, size_x, size_y, entrance_pupil_diameter
    ):
        as_torch = torch.is_tensor(field)
        field_t = self._to_torch_complex(field)
        pupil_mask_t = self._to_torch_float(
            self.create_circular_pupil_mask(
                field_t.shape,
                size_x=size_x,
                size_y=size_y,
                pupil_diameter=entrance_pupil_diameter,
            )
        )
        out = field_t * pupil_mask_t
        return out if as_torch else self._to_numpy(out)

    @staticmethod
    def _check_stop(stop_callback):
        if stop_callback is not None and bool(stop_callback()):
            raise SimulationStoppedError("Simulation stopped by user.")

    @staticmethod
    def _emit_progress(progress_callback, stage, frac):
        if progress_callback is None:
            return
        progress_callback(stage, float(min(1.0, max(0.0, frac))))

    @staticmethod
    def _coord_count(coords):
        if torch.is_tensor(coords):
            return int(coords.numel())
        return int(np.asarray(coords).size)

    @staticmethod
    def _intensity_from_components(ex, ey, ez):
        return torch.abs(ex) ** 2 + torch.abs(ey) ** 2 + torch.abs(ez) ** 2

    def _make_scan_coords(self, range_x_m, range_y_m, range_z_m, num_points):
        num_points = int(num_points)
        return (
            torch.linspace(-range_x_m, range_x_m, num_points, dtype=torch.float64, device=self.device),
            torch.linspace(-range_y_m, range_y_m, num_points, dtype=torch.float64, device=self.device),
            torch.linspace(-range_z_m, range_z_m, num_points, dtype=torch.float64, device=self.device),
        )

    def prepare_pupil_field_from_input(
        self,
        input_intensity,
        size_x,
        size_y,
        entrance_pupil_diameter,
        fresnel_distance,
        use_gaussian_profile=None,
        gaussian_waist_x=None,
        gaussian_waist_y=None,
        custom_mask=None,
        ring_inner_diameter=None,
        ring_outer_diameter=None,
        progress_callback=None,
        stop_callback=None,
    ):
        """
        Pre-PSF pipeline:
        input intensity -> optional gaussian -> optional mask(s) -> Fresnel -> entrance pupil.
        Returns intermediate fields/intensities and coordinates for plotting.
        """

        self._check_stop(stop_callback)
        self._emit_progress(progress_callback, "Validating input", 0.02)
        intensity = self._to_torch_float(input_intensity)
        if intensity.ndim != 2:
            raise ValueError("input_intensity must be a 2D array.")

        min_i = float(torch.min(intensity).item())
        max_i = float(torch.max(intensity).item())
        if min_i < 0.0 or max_i > 1.0:
            raise ValueError("input_intensity must be normalized in [0, 1].")

        if use_gaussian_profile is None:
            use_gaussian_profile = (
                gaussian_waist_x is not None or gaussian_waist_y is not None
            )

        self._check_stop(stop_callback)
        self._emit_progress(progress_callback, "Applying source envelope", 0.18)
        intensity_after_gaussian = intensity.clone()
        if use_gaussian_profile:
            if gaussian_waist_x is None or gaussian_waist_y is None:
                raise ValueError("Both gaussian_waist_x and gaussian_waist_y must be provided.")
            if gaussian_waist_x <= 0 or gaussian_waist_y <= 0:
                raise ValueError("gaussian_waist_x and gaussian_waist_y must be positive.")
            intensity_after_gaussian = self.apply_gaussian_envelope(
                intensity_after_gaussian,
                size_x=size_x,
                size_y=size_y,
                wx=gaussian_waist_x,
                wy=gaussian_waist_y,
            )

        self._check_stop(stop_callback)
        self._emit_progress(progress_callback, "Applying masks", 0.33)
        intensity_after_mask = intensity_after_gaussian.clone()

        if custom_mask is not None:
            custom_mask_t = self._to_torch_float(custom_mask)
            if tuple(int(v) for v in custom_mask_t.shape) != tuple(int(v) for v in intensity_after_mask.shape):
                raise ValueError("custom_mask shape must match input_intensity shape.")
            intensity_after_mask = intensity_after_mask * custom_mask_t

        ring_mask = None
        if ring_inner_diameter is not None or ring_outer_diameter is not None:
            if ring_inner_diameter is None or ring_outer_diameter is None:
                raise ValueError("Both ring_inner_diameter and ring_outer_diameter are required.")
            ring_mask = self._to_torch_float(
                self.create_ring_block_mask(
                intensity.shape,
                size_x=size_x,
                size_y=size_y,
                inner_diameter=ring_inner_diameter,
                outer_diameter=ring_outer_diameter,
            )
            )
            intensity_after_mask = intensity_after_mask * ring_mask

        max_masked = float(torch.max(intensity_after_mask).item())
        if max_masked <= 0:
            raise ValueError("Intensity is all zeros after masking.")
        intensity_after_mask = intensity_after_mask / max_masked

        self._check_stop(stop_callback)
        self._emit_progress(progress_callback, "Propagating to entrance pupil", 0.52)
        if isclose(fresnel_distance, 0.0):
            # z=0: skip propagation and keep the mask-plane field.
            field_after_fresnel = torch.sqrt(intensity_after_mask).to(torch.complex128)
        else:
            field_after_fresnel = self.fresnel_propagate(
                intensity_after_mask, size_x=size_x, size_y=size_y, z=fresnel_distance
            )

        self._check_stop(stop_callback)
        self._emit_progress(progress_callback, "Applying entrance pupil", 0.74)
        pupil_mask = self._to_torch_float(
            self.create_circular_pupil_mask(
            tuple(int(v) for v in intensity.shape),
            size_x=size_x,
            size_y=size_y,
            pupil_diameter=entrance_pupil_diameter,
        )
        )
        field_at_pupil = field_after_fresnel * pupil_mask

        entrance_pupil_intensity = torch.abs(field_after_fresnel) ** 2
        intensity_after_pupil_mask = torch.abs(field_at_pupil) ** 2
        entrance_pupil_intensity_norm = self._normalize_intensity(entrance_pupil_intensity.clone())
        intensity_after_pupil_mask_norm = self._normalize_intensity(
            intensity_after_pupil_mask.clone()
        )

        trim_info = self._centered_pupil_trim_window(
            shape=tuple(int(v) for v in intensity.shape),
            size_x=size_x,
            size_y=size_y,
            entrance_pupil_diameter=entrance_pupil_diameter,
            margin_pixels=self.pupil_trim_margin_pixels,
        )
        if self.trim_pupil_for_psf and bool(trim_info["trimmed"]):
            row0 = int(trim_info["row0"])
            row1 = int(trim_info["row1"])
            col0 = int(trim_info["col0"])
            col1 = int(trim_info["col1"])
            field_at_pupil_for_psf = field_at_pupil[row0:row1, col0:col1]
            size_x_for_psf = float(trim_info["size_x"])
            size_y_for_psf = float(trim_info["size_y"])
            pupil_trimmed_for_psf = True
        else:
            row0, row1 = 0, int(field_at_pupil.shape[0])
            col0, col1 = 0, int(field_at_pupil.shape[1])
            field_at_pupil_for_psf = field_at_pupil
            size_x_for_psf = float(size_x)
            size_y_for_psf = float(size_y)
            pupil_trimmed_for_psf = False

        nx, ny = int(intensity.shape[0]), int(intensity.shape[1])
        entrance_pupil_x_coords = _torch_linspace_no_endpoint(
            -size_x / 2, size_x / 2, nx, device=self.device
        )
        entrance_pupil_y_coords = _torch_linspace_no_endpoint(
            -size_y / 2, size_y / 2, ny, device=self.device
        )

        self._emit_progress(progress_callback, "Pupil field ready", 1.0)
        return {
            "input_intensity": self._to_numpy(intensity),
            "intensity_after_gaussian": self._to_numpy(intensity_after_gaussian),
            "intensity_after_mask": self._to_numpy(intensity_after_mask),
            "ring_mask": None if ring_mask is None else self._to_numpy(ring_mask),
            "entrance_pupil_mask": self._to_numpy(pupil_mask),
            "field_after_fresnel": self._to_numpy(field_after_fresnel),
            "field_at_pupil": self._to_numpy(field_at_pupil),
            "field_at_pupil_for_psf": self._to_numpy(field_at_pupil_for_psf),
            "size_x_for_psf": float(size_x_for_psf),
            "size_y_for_psf": float(size_y_for_psf),
            "pupil_trimmed_for_psf": bool(pupil_trimmed_for_psf),
            "pupil_trim_window": [int(row0), int(row1), int(col0), int(col1)],
            "intensity_at_entrance_pupil": self._to_numpy(entrance_pupil_intensity),
            "intensity_at_entrance_pupil_norm": self._to_numpy(entrance_pupil_intensity_norm),
            "intensity_after_pupil_mask": self._to_numpy(intensity_after_pupil_mask),
            "intensity_after_pupil_mask_norm": self._to_numpy(intensity_after_pupil_mask_norm),
            "entrance_pupil_x_coords": self._to_numpy(entrance_pupil_x_coords),
            "entrance_pupil_y_coords": self._to_numpy(entrance_pupil_y_coords),
        }

    def _safe_asin(self, x):
        x_t = self._to_torch_float(x)
        over = (x_t > 1.0 + 1e-12) | (x_t < -1.0 - 1e-12)
        if bool(torch.any(over).item()):
            # Keep evanescent/over-critical angles as complex values.
            return torch.asin(x_t.to(torch.complex128))
        return torch.asin(torch.clamp(x_t, -1.0, 1.0))

    def _theta_3(self, theta):
        theta_t = self._to_torch_float(theta)
        return self._safe_asin(self.n_i * torch.sin(theta_t) / self.n_s)

    def _theta_2(self, theta):
        theta_t = self._to_torch_float(theta)
        return self._safe_asin(self.n_i * torch.sin(theta_t) / self.n_g)

    def _w_abb(self, theta):
        theta_t = self._to_torch_float(theta)
        theta_3 = self._theta_3(theta_t)
        theta_id = self._safe_asin(self.n_i * torch.sin(theta_t) / self.n_id)
        return (
            self.h_2 * self.n_s * torch.cos(theta_3)
            - self.h_1 * self.n_i * torch.cos(theta_t)
            + self.h_1d * self.n_id * torch.cos(theta_id)
        )

    def _transmission_terms(self, theta):
        theta_t = self._to_torch_float(theta)
        theta_2 = self._theta_2(theta_t)
        theta_3 = self._theta_3(theta_t)

        cos_theta = torch.cos(theta_t)
        cos_theta_2 = torch.cos(theta_2)
        cos_theta_3 = torch.cos(theta_3)

        t_1_s = (2 * self.n_i * cos_theta) / (
            self.n_i * cos_theta + self.n_g * cos_theta_2
        )
        t_1_p = (2 * self.n_i * cos_theta) / (
            self.n_g * cos_theta + self.n_i * cos_theta_2
        )
        r_1_s = (self.n_i * cos_theta - self.n_g * cos_theta_2) / (
            self.n_i * cos_theta + self.n_g * cos_theta_2
        )
        r_1_p = (self.n_g * cos_theta - self.n_i * cos_theta_2) / (
            self.n_g * cos_theta + self.n_i * cos_theta_2
        )

        t_2_s = (2 * self.n_g * cos_theta_2) / (
            self.n_g * cos_theta_2 + self.n_s * cos_theta_3
        )
        t_2_p = (2 * self.n_g * cos_theta_2) / (
            self.n_s * cos_theta_2 + self.n_g * cos_theta_3
        )
        r_2_s = (self.n_g * cos_theta_2 - self.n_s * cos_theta_3) / (
            self.n_g * cos_theta_2 + self.n_s * cos_theta_3
        )
        r_2_p = (self.n_s * cos_theta_2 - self.n_g * cos_theta_3) / (
            self.n_s * cos_theta_2 + self.n_g * cos_theta_3
        )

        beta_2 = self.k * self.n_g * self.t_g * cos_theta_2
        theta_beta_d = self._safe_asin(self.n_i * torch.sin(theta_t) / self.n_gd)
        beta_2d = self.k * self.n_gd * self.t_gd * torch.cos(theta_beta_d)

        phase = torch.exp(1j * (beta_2 - beta_2d))
        d_2_s = 1 + r_1_s * r_2_s * torch.exp(2j * (beta_2 - beta_2d))
        d_2_p = 1 + r_1_p * r_2_p * torch.exp(2j * (beta_2 - beta_2d))

        t_s = t_2_s * t_1_s * phase / d_2_s
        t_p = t_2_p * t_1_p * phase / d_2_p

        return t_s, t_p, theta_3

    def _prepare_integration_tables_polar_interp(
        self, pupil_field, size_x, size_y, entrance_pupil_diameter
    ):
        mode = "polar_interp"
        pupil_field = self._to_torch_complex(pupil_field)

        table_key = self._integration_table_cache_key(
            pupil_field,
            mode=mode,
            size_x=size_x,
            size_y=size_y,
            entrance_pupil_diameter=entrance_pupil_diameter,
        )
        cached_table = self._lru_get(self._integration_table_cache, table_key)
        if cached_table is not None:
            return cached_table

        base_key = self._table_base_key(
            mode, pupil_field.shape, size_x, size_y, entrance_pupil_diameter
        )
        geom_key = ("geom",) + base_key
        geom = self._lru_get(self._geometry_cache, geom_key)
        if geom is None:
            nx, ny = int(pupil_field.shape[0]), int(pupil_field.shape[1])

            theta = torch.linspace(
                self.theta_min,
                self.theta_max,
                int(self.theta_samples),
                dtype=torch.float64,
                device=self.device,
            )
            phi = _torch_linspace_no_endpoint(
                0.0, 2.0 * pi, int(self.phi_samples), device=self.device
            )

            sin_theta_max = torch.sin(
                torch.tensor(self.theta_max, dtype=torch.float64, device=self.device)
            )
            rho = (torch.sin(theta) / sin_theta_max) * (0.5 * entrance_pupil_diameter)
            xp = rho[:, None] * torch.cos(phi)[None, :]
            yp = rho[:, None] * torch.sin(phi)[None, :]

            t_s, t_p, theta_3 = self._transmission_terms(theta)
            w_abb = self._w_abb(theta)

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            sqrt_cos_theta = torch.sqrt(cos_theta)
            cos_theta_3 = torch.cos(theta_3)
            sin_theta_3 = torch.sin(theta_3)

            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            t_s_col = t_s[:, None]
            t_p_col = t_p[:, None]
            cos_phi_grid = cos_phi[None, :]
            sin_phi_grid = sin_phi[None, :]

            geom_ex = (
                t_p_col * cos_theta_3[:, None] * cos_phi_grid**2
                + t_s_col * sin_phi_grid**2
            )
            geom_ey = (
                t_p_col * cos_theta_3[:, None] * sin_phi_grid * cos_phi_grid
                - t_s_col * sin_phi_grid * cos_phi_grid
            )
            geom_ez = -t_p_col * sin_theta_3[:, None] * cos_phi_grid

            common_prefactor = (
                sqrt_cos_theta[:, None]
                * torch.exp(1j * self.k * w_abb)[:, None]
                * sin_theta[:, None]
            )

            dtheta = theta[1] - theta[0]
            dphi = (2.0 * pi) / float(phi.numel())
            theta_weights = torch.ones(theta.numel(), dtype=torch.float64, device=self.device)
            theta_weights[0] = 0.5
            theta_weights[-1] = 0.5
            theta_weights *= dtheta

            geom = {
                "xp": xp,
                "yp": yp,
                "theta_3": theta_3,
                "cos_theta_3": cos_theta_3,
                "dphi": dphi,
                "theta_weights": theta_weights,
                "sin_phi": sin_phi,
                "cos_phi": cos_phi,
                "phase_scale": self.n_i * self.k * sin_theta,
                "component_factors": torch.stack(
                    (
                        common_prefactor * geom_ex,
                        common_prefactor * geom_ey,
                        common_prefactor * geom_ez,
                    ),
                    dim=0,
                ),
            }
            self._lru_put(self._geometry_cache, geom_key, geom, self._geometry_cache_max)

        nx, ny = int(pupil_field.shape[0]), int(pupil_field.shape[1])
        dx = float(size_x) / float(nx)
        dy = float(size_y) / float(ny)

        src_row = (geom["xp"] + 0.5 * float(size_x)) / dx
        src_col = (geom["yp"] + 0.5 * float(size_y)) / dy

        if nx > 1:
            norm_row = (2.0 * src_row / float(nx - 1)) - 1.0
        else:
            norm_row = torch.zeros_like(src_row)
        if ny > 1:
            norm_col = (2.0 * src_col / float(ny - 1)) - 1.0
        else:
            norm_col = torch.zeros_like(src_col)

        grid = torch.stack((norm_col, norm_row), dim=-1).unsqueeze(0)

        pupil_real = pupil_field.real[None, None, :, :]
        pupil_imag = pupil_field.imag[None, None, :, :]
        e_real = F.grid_sample(
            pupil_real,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]
        e_imag = F.grid_sample(
            pupil_imag,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]
        e_inf = e_real + 1j * e_imag

        tables = {
            "mode": mode,
            "cache_token": table_key,
            "theta_3": geom["theta_3"],
            "cos_theta_3": geom["cos_theta_3"],
            "dphi": geom["dphi"],
            "theta_weights": geom["theta_weights"],
            "sin_phi": geom["sin_phi"],
            "cos_phi": geom["cos_phi"],
            "phase_scale": geom["phase_scale"],
            "field_components_static": geom["component_factors"] * e_inf[None, :, :],
        }
        self._lru_put(
            self._integration_table_cache,
            table_key,
            tables,
            self._integration_table_cache_max,
        )
        return tables

    def _prepare_integration_tables_cartesian_jacobian(
        self, pupil_field, size_x, size_y, entrance_pupil_diameter
    ):
        mode = "cartesian_jacobian"
        pupil_field = self._to_torch_complex(pupil_field)

        table_key = self._integration_table_cache_key(
            pupil_field,
            mode=mode,
            size_x=size_x,
            size_y=size_y,
            entrance_pupil_diameter=entrance_pupil_diameter,
        )
        cached_table = self._lru_get(self._integration_table_cache, table_key)
        if cached_table is not None:
            return cached_table

        base_key = self._table_base_key(
            mode, pupil_field.shape, size_x, size_y, entrance_pupil_diameter
        )
        geom_key = ("geom",) + base_key
        geom = self._lru_get(self._geometry_cache, geom_key)
        if geom is None:
            nx, ny = int(pupil_field.shape[0]), int(pupil_field.shape[1])
            x = _torch_linspace_no_endpoint(-size_x / 2, size_x / 2, nx, device=self.device)
            y = _torch_linspace_no_endpoint(-size_y / 2, size_y / 2, ny, device=self.device)
            xg, yg = torch.meshgrid(x, y, indexing="ij")

            r = torch.sqrt(xg**2 + yg**2)
            pupil_radius = 0.5 * float(entrance_pupil_diameter)
            if pupil_radius <= 0:
                raise ValueError("entrance_pupil_diameter must be positive.")

            inside = r <= pupil_radius
            if not bool(torch.any(inside).item()):
                raise ValueError("No Cartesian grid points lie inside the entrance pupil.")

            xp = xg[inside]
            yp = yg[inside]
            flat_inside = torch.nonzero(inside.reshape(-1), as_tuple=False).reshape(-1).to(torch.int64)

            sin_theta_max = torch.sin(
                torch.tensor(self.theta_max, dtype=torch.float64, device=self.device)
            )
            sin_theta = torch.clamp((r[inside] / pupil_radius) * sin_theta_max, 0.0, sin_theta_max)
            if self.theta_min > 0:
                sin_theta_min = torch.sin(
                    torch.tensor(self.theta_min, dtype=torch.float64, device=self.device)
                )
                keep = sin_theta >= sin_theta_min
                xp = xp[keep]
                yp = yp[keep]
                flat_inside = flat_inside[keep]
                sin_theta = sin_theta[keep]
                if int(sin_theta.numel()) == 0:
                    raise ValueError("No Cartesian points remain after theta_min filtering.")

            theta = torch.asin(torch.clamp(sin_theta, -1.0, 1.0))
            cos_theta = torch.cos(theta)
            phi = torch.atan2(yp, xp)
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            t_s, t_p, theta_3 = self._transmission_terms(theta)
            w_abb = self._w_abb(theta)

            cos_theta_3 = torch.cos(theta_3)
            sin_theta_3 = torch.sin(theta_3)

            geom_ex = t_p * cos_theta_3 * cos_phi**2 + t_s * sin_phi**2
            geom_ey = t_p * cos_theta_3 * sin_phi * cos_phi - t_s * sin_phi * cos_phi
            geom_ez = -t_p * sin_theta_3 * cos_phi

            dx = size_x / nx
            dy = size_y / ny
            jacobian = (sin_theta_max**2 / (pupil_radius**2)) * (dx * dy) / torch.clamp(
                cos_theta, 1e-12, None
            )
            prefactor = torch.sqrt(cos_theta) * torch.exp(1j * self.k * w_abb) * jacobian

            geom = {
                "flat_indices": flat_inside,
                "cos_theta_3": cos_theta_3,
                "phase_scale_x": self.n_i * self.k * sin_theta * cos_phi,
                "phase_scale_y": self.n_i * self.k * sin_theta * sin_phi,
                "component_factors": torch.stack(
                    (
                        prefactor * geom_ex,
                        prefactor * geom_ey,
                        prefactor * geom_ez,
                    ),
                    dim=0,
                ),
            }
            self._lru_put(self._geometry_cache, geom_key, geom, self._geometry_cache_max)

        e_inf = pupil_field.reshape(-1)[geom["flat_indices"]]
        tables = {
            "mode": mode,
            "cache_token": table_key,
            "cos_theta_3": geom["cos_theta_3"],
            "phase_scale_x": geom["phase_scale_x"],
            "phase_scale_y": geom["phase_scale_y"],
            "flat_indices": geom["flat_indices"],
            "field_components_static": geom["component_factors"] * e_inf[None, :],
        }
        self._lru_put(
            self._integration_table_cache,
            table_key,
            tables,
            self._integration_table_cache_max,
        )
        return tables

    def _prepare_integration_tables(
        self, pupil_field, size_x, size_y, entrance_pupil_diameter
    ):
        if self.integration_mode == "cartesian_jacobian":
            return self._prepare_integration_tables_cartesian_jacobian(
                pupil_field,
                size_x=size_x,
                size_y=size_y,
                entrance_pupil_diameter=entrance_pupil_diameter,
            )
        return self._prepare_integration_tables_polar_interp(
            pupil_field,
            size_x=size_x,
            size_y=size_y,
            entrance_pupil_diameter=entrance_pupil_diameter,
        )

    def _weighted_components(self, tables):
        if tables["mode"] == "cartesian_jacobian":
            return self.C * tables["field_components_static"]
        return (
            self.C
            * tables["dphi"]
            * tables["theta_weights"][None, :, None]
            * tables["field_components_static"]
        )

    def _axial_phase_for_z(self, z_values, tables):
        z_values = self._to_torch_float(z_values)
        return torch.exp(
            1j
            * self.k
            * self.n_s
            * tables["cos_theta_3"][:, None]
            * z_values[None, :]
        )

    def _precompute_exp_x(self, x_points, tables):
        x_points = self._to_torch_float(x_points)
        if tables["mode"] == "cartesian_jacobian":
            return torch.exp(1j * tables["phase_scale_x"][:, None] * x_points[None, :])
        x_cos_phi = x_points[:, None] * tables["cos_phi"][None, :]
        return torch.exp(1j * tables["phase_scale"][:, None, None] * x_cos_phi[None, :, :])

    def _precompute_exp_y(self, y_points, tables):
        y_points = self._to_torch_float(y_points)
        if tables["mode"] == "cartesian_jacobian":
            return torch.exp(1j * tables["phase_scale_y"][:, None] * y_points[None, :])
        y_sin_phi = y_points[:, None] * tables["sin_phi"][None, :]
        return torch.exp(1j * tables["phase_scale"][:, None, None] * y_sin_phi[None, :, :])

    def _get_phase_terms(self, tables, x_coords=None, y_coords=None, z_coords=None):
        terms = {}
        if x_coords is not None:
            terms["exp_x"] = self._precompute_exp_x(x_coords, tables)
        if y_coords is not None:
            terms["exp_y"] = self._precompute_exp_y(y_coords, tables)
        if z_coords is not None:
            terms["axial_by_z"] = self._axial_phase_for_z(z_coords, tables)
        return terms

    def _integrate_line_points(self, x_points, y_points, z_val, tables):
        """
        Integrate vector field components for a line of Cartesian points at fixed z.

        This keeps the same numerical integration rule as the scalar per-point form,
        but batches all points on the line to reduce Python-loop overhead.
        """
        x_points = self._to_torch_float(x_points)
        y_points = self._to_torch_float(y_points)
        if x_points.shape != y_points.shape:
            raise ValueError("x_points and y_points must have matching shapes.")

        weighted_components = self._weighted_components(tables)
        if tables["mode"] == "cartesian_jacobian":
            radial_phase = torch.exp(
                1j
                * (
                    tables["phase_scale_x"][:, None] * x_points[None, :]
                    + tables["phase_scale_y"][:, None] * y_points[None, :]
                )
            )
            axial_phase = torch.exp(
                1j * self.k * self.n_s * float(z_val) * tables["cos_theta_3"]
            )[:, None]
            kernel = radial_phase * axial_phase
            # (3, p) @ (p, x) -> (3, x)
            field_components = torch.matmul(weighted_components, kernel)
        else:
            xy_projection = (
                x_points[None, :, None] * tables["cos_phi"][None, None, :]
                + y_points[None, :, None] * tables["sin_phi"][None, None, :]
            )
            radial_phase = torch.exp(1j * tables["phase_scale"][:, None, None] * xy_projection)

            axial_phase = torch.exp(
                1j * self.k * self.n_s * z_val * tables["cos_theta_3"]
            )[:, None, None]

            kernel = radial_phase * axial_phase
            field_components = torch.einsum(
                "txp,ctp->cx",
                kernel,
                weighted_components,
            )
        return field_components[0], field_components[1], field_components[2]

    @staticmethod
    def _integration_point_count(tables):
        static = tables["field_components_static"]
        return int(prod(static.shape[1:]))

    def _compute_xz_plane(
        self,
        x_coords,
        z_coords,
        tables,
        verbose=False,
        progress_callback=None,
        stop_callback=None,
        z_chunk_size=8,
        weighted_components=None,
        exp_x=None,
        axial_by_z=None,
    ):
        if weighted_components is None:
            weighted_components = self._weighted_components(tables)
        if exp_x is None:
            exp_x = self._precompute_exp_x(x_coords, tables)
        if axial_by_z is None:
            axial_by_z = self._axial_phase_for_z(z_coords, tables)
        if verbose:
            print("XZ progress: vectorized contraction")

        nz = self._coord_count(z_coords)
        nx = self._coord_count(x_coords)
        i_xz = torch.empty((nz, nx), dtype=torch.float64, device=self.device)

        z_chunk_size = max(1, int(z_chunk_size))
        wx_t = None
        if tables["mode"] == "cartesian_jacobian":
            # Precompute component-weighted X-phase once:
            # weighted_components: (3, p), exp_x: (p, x) -> wx_t: (3, x, p)
            wx_t = (weighted_components[:, :, None] * exp_x[None, :, :]).permute(0, 2, 1)
        for start in range(0, nz, z_chunk_size):
            self._check_stop(stop_callback)
            end = min(nz, start + z_chunk_size)
            axial_chunk = axial_by_z[:, start:end]

            if tables["mode"] == "cartesian_jacobian":
                # (3, x, p) @ (p, zc) -> (3, x, zc) -> (3, zc, x)
                field_chunk = torch.matmul(wx_t, axial_chunk).permute(0, 2, 1)
            else:
                field_chunk = torch.einsum(
                    "ctp,txp,tz->czx",
                    weighted_components,
                    exp_x,
                    axial_chunk,
                )

            ex = field_chunk[0]
            ey = field_chunk[1]
            ez = field_chunk[2]
            i_xz[start:end, :] = self._intensity_from_components(ex, ey, ez)

            if progress_callback is not None:
                progress_callback(float(end) / float(nz))

        return i_xz

    def _compute_xy_plane(
        self,
        x_coords,
        y_coords,
        z_val,
        tables,
        verbose=False,
        progress_callback=None,
        stop_callback=None,
        y_chunk_size=8,
        weighted_components=None,
        exp_x=None,
        exp_y=None,
        axial=None,
    ):
        if weighted_components is None:
            weighted_components = self._weighted_components(tables)
        if exp_x is None:
            exp_x = self._precompute_exp_x(x_coords, tables)
        if exp_y is None:
            exp_y = self._precompute_exp_y(y_coords, tables)
        if axial is None:
            axial = self._axial_phase_for_z([z_val], tables)[:, 0]
        if verbose:
            print("XY progress: vectorized contraction")

        ny = self._coord_count(y_coords)
        nx = self._coord_count(x_coords)
        i_xy = torch.empty((ny, nx), dtype=torch.float64, device=self.device)
        y_chunk_size = max(1, int(y_chunk_size))

        if tables["mode"] == "cartesian_jacobian":
            weighted_z = weighted_components * axial[None, :]
            # Precompute component-weighted X-phase once:
            # weighted_z: (3, p), exp_x: (p, x) -> wx_t: (3, x, p)
            wx_t = (weighted_z[:, :, None] * exp_x[None, :, :]).permute(0, 2, 1)
            for start in range(0, ny, y_chunk_size):
                self._check_stop(stop_callback)
                end = min(ny, start + y_chunk_size)
                exp_y_chunk = exp_y[:, start:end]
                # (3, x, p) @ (p, yc) -> (3, x, yc) -> (3, yc, x)
                field_chunk = torch.matmul(wx_t, exp_y_chunk).permute(0, 2, 1)
                ex = field_chunk[0]
                ey = field_chunk[1]
                ez = field_chunk[2]
                i_xy[start:end, :] = self._intensity_from_components(ex, ey, ez)
                if progress_callback is not None:
                    progress_callback(float(end) / float(ny))
        else:
            weighted_z = weighted_components * axial[None, :, None]
            for start in range(0, ny, y_chunk_size):
                self._check_stop(stop_callback)
                end = min(ny, start + y_chunk_size)
                exp_y_chunk = exp_y[:, start:end, :]
                field_chunk = torch.einsum(
                    "ctp,txp,typ->cyx",
                    weighted_z,
                    exp_x,
                    exp_y_chunk,
                )
                ex = field_chunk[0]
                ey = field_chunk[1]
                ez = field_chunk[2]
                i_xy[start:end, :] = self._intensity_from_components(ex, ey, ez)
                if progress_callback is not None:
                    progress_callback(float(end) / float(ny))

        return i_xy

    def _compute_yz_plane(
        self,
        y_coords,
        z_coords,
        tables,
        verbose=False,
        progress_callback=None,
        stop_callback=None,
        z_chunk_size=8,
        weighted_components=None,
        exp_y=None,
        axial_by_z=None,
    ):
        if weighted_components is None:
            weighted_components = self._weighted_components(tables)
        if exp_y is None:
            exp_y = self._precompute_exp_y(y_coords, tables)
        if axial_by_z is None:
            axial_by_z = self._axial_phase_for_z(z_coords, tables)
        if verbose:
            print("YZ progress: vectorized contraction")

        nz = self._coord_count(z_coords)
        ny = self._coord_count(y_coords)
        i_yz = torch.empty((nz, ny), dtype=torch.float64, device=self.device)

        z_chunk_size = max(1, int(z_chunk_size))
        wy_t = None
        if tables["mode"] == "cartesian_jacobian":
            # Precompute component-weighted Y-phase once:
            # weighted_components: (3, p), exp_y: (p, y) -> wy_t: (3, y, p)
            wy_t = (weighted_components[:, :, None] * exp_y[None, :, :]).permute(0, 2, 1)
        for start in range(0, nz, z_chunk_size):
            self._check_stop(stop_callback)
            end = min(nz, start + z_chunk_size)
            axial_chunk = axial_by_z[:, start:end]

            if tables["mode"] == "cartesian_jacobian":
                # (3, y, p) @ (p, zc) -> (3, y, zc) -> (3, zc, y)
                field_chunk = torch.matmul(wy_t, axial_chunk).permute(0, 2, 1)
            else:
                field_chunk = torch.einsum(
                    "ctp,typ,tz->czy",
                    weighted_components,
                    exp_y,
                    axial_chunk,
                )

            ex = field_chunk[0]
            ey = field_chunk[1]
            ez = field_chunk[2]
            i_yz[start:end, :] = self._intensity_from_components(ex, ey, ez)

            if progress_callback is not None:
                progress_callback(float(end) / float(nz))

        return i_yz

    def _compute_xyz_volume(
        self,
        x_coords,
        y_coords,
        z_coords,
        tables,
        z_chunk_size=8,
        verbose=False,
        progress_callback=None,
        stop_callback=None,
        weighted_components=None,
        exp_x=None,
        exp_y=None,
        axial_by_z=None,
    ):
        """
        Compute full 3D PSF intensity volume I(z, y, x) with the same integral kernel
        used by the 2D plane solvers. Chunking over z limits peak memory usage.
        """
        if weighted_components is None:
            weighted_components = self._weighted_components(tables)
        if exp_x is None:
            exp_x = self._precompute_exp_x(x_coords, tables)
        if exp_y is None:
            exp_y = self._precompute_exp_y(y_coords, tables)
        if axial_by_z is None:
            axial_by_z = self._axial_phase_for_z(z_coords, tables)

        nz = self._coord_count(z_coords)
        ny = self._coord_count(y_coords)
        nx = self._coord_count(x_coords)
        i_xyz = torch.empty((nz, ny, nx), dtype=torch.float64, device=self.device)

        z_chunk_size = max(1, int(z_chunk_size))
        use_x_first = True
        wx_t = None
        wy_t = None
        if tables["mode"] == "cartesian_jacobian":
            # Choose the smaller lateral axis for temporary (zc, axis, p) work arrays.
            use_x_first = nx <= ny
            if use_x_first:
                # weighted_components: (3, p), exp_x: (p, x) -> wx_t: (3, x, p)
                wx_t = (weighted_components[:, :, None] * exp_x[None, :, :]).permute(0, 2, 1)
            else:
                # weighted_components: (3, p), exp_y: (p, y) -> wy_t: (3, y, p)
                wy_t = (weighted_components[:, :, None] * exp_y[None, :, :]).permute(0, 2, 1)
        for start in range(0, nz, z_chunk_size):
            self._check_stop(stop_callback)
            end = min(nz, start + z_chunk_size)
            axial_chunk = axial_by_z[:, start:end]

            if tables["mode"] == "cartesian_jacobian":
                # Avoid building a large (3, zc, x, p) intermediate by contracting
                # one vector component at a time and accumulating intensity directly.
                axial_t = axial_chunk.permute(1, 0)
                zc = int(end - start)
                i_chunk = torch.zeros((zc, ny, nx), dtype=torch.float64, device=self.device)
                if use_x_first:
                    for c_idx in range(3):
                        # (zc, x, p) @ (p, y) -> (zc, x, y) -> (zc, y, x)
                        left = wx_t[c_idx][None, :, :] * axial_t[:, None, :]
                        comp_field = torch.matmul(left, exp_y).permute(0, 2, 1)
                        i_chunk += torch.abs(comp_field) ** 2
                else:
                    for c_idx in range(3):
                        # (zc, y, p) @ (p, x) -> (zc, y, x)
                        left = wy_t[c_idx][None, :, :] * axial_t[:, None, :]
                        comp_field = torch.matmul(left, exp_x)
                        i_chunk += torch.abs(comp_field) ** 2
                i_xyz[start:end, :, :] = i_chunk
            else:
                field_chunk = torch.einsum(
                    "ctp,txp,typ,tz->czyx",
                    weighted_components,
                    exp_x,
                    exp_y,
                    axial_chunk,
                )
                ex = field_chunk[0]
                ey = field_chunk[1]
                ez = field_chunk[2]
                i_xyz[start:end, :, :] = self._intensity_from_components(ex, ey, ez)

            if verbose:
                pct = 100.0 * end / nz
                print(f"XYZ progress: {end}/{nz} ({pct:.1f}%)")
            if progress_callback is not None:
                progress_callback(float(end) / float(nz))

        return i_xyz

    def simulate_psf_volume_from_pupil_field(
        self,
        pupil_field,
        size_x,
        size_y,
        entrance_pupil_diameter,
        range_microns=2.0,
        range_microns_x=None,
        range_microns_y=None,
        range_microns_z=None,
        num_points=41,
        z_chunk_size=8,
        verbose=False,
        progress_callback=None,
        stop_callback=None,
    ):
        pupil_field = self._to_torch_complex(pupil_field)
        if pupil_field.ndim != 2:
            raise ValueError("pupil_field must be a 2D array.")
        if int(num_points) < 3:
            raise ValueError("num_points must be >= 3 for 3D simulation.")

        range_x_m, range_y_m, range_z_m = self._resolve_axis_ranges_m(
            range_microns=range_microns,
            range_microns_x=range_microns_x,
            range_microns_y=range_microns_y,
            range_microns_z=range_microns_z,
        )
        x_coords, y_coords, z_coords = self._make_scan_coords(
            range_x_m, range_y_m, range_z_m, num_points
        )

        self._check_stop(stop_callback)
        self._emit_progress(progress_callback, "Preparing 3D integration tables", 0.0)
        tables = self._prepare_integration_tables(
            pupil_field,
            size_x=size_x,
            size_y=size_y,
            entrance_pupil_diameter=entrance_pupil_diameter,
        )
        weighted_components = self._weighted_components(tables)
        phase_terms = self._get_phase_terms(
            tables,
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
        )
        exp_x = phase_terms["exp_x"]
        exp_y = phase_terms["exp_y"]
        axial_by_z = phase_terms["axial_by_z"]
        pint = max(1, self._integration_point_count(tables))
        nx = int(x_coords.numel())
        ny = int(y_coords.numel())
        nz = int(z_coords.numel())
        table_ops = float(pint)
        vol_ops = float(nx * ny * nz * pint)
        post_ops = float(nx * ny * nz)
        total_ops = max(1.0, table_ops + vol_ops + post_ops)

        self._check_stop(stop_callback)
        self._emit_progress(progress_callback, "Computing 3D volume", table_ops / total_ops)

        i_xyz = self._compute_xyz_volume(
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
            tables=tables,
            z_chunk_size=z_chunk_size,
            verbose=verbose,
            progress_callback=lambda f: self._emit_progress(
                progress_callback,
                "Computing 3D volume", (table_ops + vol_ops * float(f)) / total_ops
            ),
            stop_callback=stop_callback,
            weighted_components=weighted_components,
            exp_x=exp_x,
            exp_y=exp_y,
            axial_by_z=axial_by_z,
        )
        self._check_stop(stop_callback)
        self._emit_progress(progress_callback, "Extracting 3D slices", (table_ops + vol_ops) / total_ops)
        i_xyz_norm = self._normalize_intensity(i_xyz.clone())

        max_flat_idx = int(torch.argmax(i_xyz).item())
        plane = ny * nx
        peak_z_idx = max_flat_idx // plane
        rem = max_flat_idx % plane
        peak_y_idx = rem // nx
        peak_x_idx = rem % nx

        i_xy = i_xyz[peak_z_idx, :, :]
        i_xz = i_xyz[:, peak_y_idx, :]
        i_yz = i_xyz[:, :, peak_x_idx]

        i_xy_norm = self._normalize_intensity(i_xy.clone())
        i_xz_norm = self._normalize_intensity(i_xz.clone())
        i_yz_norm = self._normalize_intensity(i_yz.clone())

        x_coords_np = self._to_numpy(x_coords)
        y_coords_np = self._to_numpy(y_coords)
        z_coords_np = self._to_numpy(z_coords)
        i_xyz_np = self._to_numpy(i_xyz)
        i_xyz_norm_np = self._to_numpy(i_xyz_norm)
        i_xy_np = self._to_numpy(i_xy)
        i_xz_np = self._to_numpy(i_xz)
        i_yz_np = self._to_numpy(i_yz)
        i_xy_norm_np = self._to_numpy(i_xy_norm)
        i_xz_norm_np = self._to_numpy(i_xz_norm)
        i_yz_norm_np = self._to_numpy(i_yz_norm)

        fwhm_metrics = self._compute_fwhm_metrics(
            x_coords=x_coords_np,
            y_coords=y_coords_np,
            z_coords=z_coords_np,
            i_xy_norm=i_xy_norm_np,
            i_xz_norm=i_xz_norm_np,
        )
        self._emit_progress(progress_callback, "Finalizing 3D results", 1.0)

        return {
            "x_coords": x_coords_np,
            "y_coords": y_coords_np,
            "z_coords": z_coords_np,
            "I_xyz": i_xyz_np,
            "I_xyz_norm": i_xyz_norm_np,
            "I_xy": i_xy_np,
            "I_xz": i_xz_np,
            "I_yz": i_yz_np,
            "I_xy_norm": i_xy_norm_np,
            "I_xz_norm": i_xz_norm_np,
            "I_yz_norm": i_yz_norm_np,
            "peak_x_index": int(peak_x_idx),
            "peak_y_index": int(peak_y_idx),
            "peak_z_index": int(peak_z_idx),
            "z_focus": float(z_coords_np[peak_z_idx]),
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

    def simulate_psf_planes_from_pupil_field(
        self,
        pupil_field,
        size_x,
        size_y,
        entrance_pupil_diameter,
        range_microns=2.0,
        range_microns_x=None,
        range_microns_y=None,
        range_microns_z=None,
        num_points=21,
        verbose=False,
        progress_callback=None,
        stop_callback=None,
        compute_xy=True,
        compute_xz=True,
        compute_yz=True,
    ):
        pupil_field = self._to_torch_complex(pupil_field)
        if pupil_field.ndim != 2:
            raise ValueError("pupil_field must be a 2D array.")
        compute_xy = bool(compute_xy)
        compute_xz = bool(compute_xz)
        compute_yz = bool(compute_yz)
        if not (compute_xy or compute_xz or compute_yz):
            raise ValueError("At least one of compute_xy/compute_xz/compute_yz must be True.")

        range_x_m, range_y_m, range_z_m = self._resolve_axis_ranges_m(
            range_microns=range_microns,
            range_microns_x=range_microns_x,
            range_microns_y=range_microns_y,
            range_microns_z=range_microns_z,
        )
        x_coords, y_coords, z_coords = self._make_scan_coords(
            range_x_m, range_y_m, range_z_m, num_points
        )

        self._check_stop(stop_callback)
        self._emit_progress(progress_callback, "Preparing 2D integration tables", 0.0)
        tables = self._prepare_integration_tables(
            pupil_field,
            size_x=size_x,
            size_y=size_y,
            entrance_pupil_diameter=entrance_pupil_diameter,
        )
        weighted_components = self._weighted_components(tables)
        phase_terms = self._get_phase_terms(
            tables,
            x_coords=x_coords,
            y_coords=y_coords,
            z_coords=z_coords,
        )
        exp_x = phase_terms["exp_x"]
        exp_y = phase_terms["exp_y"]
        axial_by_z = phase_terms["axial_by_z"]
        pint = max(1, self._integration_point_count(tables))
        nx = int(x_coords.numel())
        ny = int(y_coords.numel())
        nz = int(z_coords.numel())
        w_prep = float(pint)
        w_xz = float(nx * nz * pint) if compute_xz else 0.0
        w_xy = float(nx * ny * pint) if compute_xy else 0.0
        w_yz = float(ny * nz * pint) if compute_yz else 0.0
        w_post = (
            (float(nx * nz) if compute_xz else 0.0)
            + (float(nx * ny) if compute_xy else 0.0)
            + (float(ny * nz) if compute_yz else 0.0)
        )
        total = max(1.0, w_prep + w_xz + w_xy + w_yz + w_post)
        done = w_prep

        i_xz = None
        i_xy = None
        i_yz = None
        z_focus = None
        max_i_z = None

        if compute_xz:
            self._check_stop(stop_callback)
            self._emit_progress(progress_callback, "Computing XZ plane", done / total)
            i_xz = self._compute_xz_plane(
                x_coords,
                z_coords,
                tables,
                verbose=verbose,
                progress_callback=lambda f: self._emit_progress(
                    progress_callback,
                    "Computing XZ plane",
                    (done + w_xz * float(f)) / total,
                ),
                stop_callback=stop_callback,
                weighted_components=weighted_components,
                exp_x=exp_x,
                axial_by_z=axial_by_z,
            )
            done += w_xz
            max_idx = int(torch.argmax(i_xz).item())
            max_i_z = max_idx // nx
            z_focus = z_coords[max_i_z]

        # When XY is requested but XZ is skipped, compute YZ first if available
        # so z_focus can be inferred without computing the full XZ plane.
        if compute_yz and ((not compute_xy) or (z_focus is None)):
            self._check_stop(stop_callback)
            self._emit_progress(progress_callback, "Computing YZ plane", done / total)
            i_yz = self._compute_yz_plane(
                y_coords,
                z_coords,
                tables,
                verbose=verbose,
                progress_callback=lambda f: self._emit_progress(
                    progress_callback,
                    "Computing YZ plane",
                    (done + w_yz * float(f)) / total,
                ),
                stop_callback=stop_callback,
                weighted_components=weighted_components,
                exp_y=exp_y,
                axial_by_z=axial_by_z,
            )
            done += w_yz
            if z_focus is None:
                max_idx_yz = int(torch.argmax(i_yz).item())
                max_i_z = max_idx_yz // ny
                z_focus = z_coords[max_i_z]

        if compute_xy:
            if z_focus is None:
                # If neither XZ nor YZ was requested, estimate focus from on-axis
                # intensity along z with a 1-column XZ evaluation.
                self._check_stop(stop_callback)
                x_center = torch.zeros((1,), dtype=torch.float64, device=self.device)
                i_xz_center = self._compute_xz_plane(
                    x_center,
                    z_coords,
                    tables,
                    verbose=False,
                    progress_callback=None,
                    stop_callback=stop_callback,
                    weighted_components=weighted_components,
                    exp_x=None,
                    axial_by_z=axial_by_z,
                )
                max_i_z = int(torch.argmax(i_xz_center).item())
                z_focus = z_coords[max_i_z]
            if max_i_z is None:
                max_i_z = int(torch.argmin(torch.abs(z_coords - z_focus)).item())

            self._check_stop(stop_callback)
            self._emit_progress(progress_callback, "Computing XY plane", done / total)
            i_xy = self._compute_xy_plane(
                x_coords,
                y_coords,
                z_focus,
                tables,
                verbose=verbose,
                progress_callback=lambda f: self._emit_progress(
                    progress_callback,
                    "Computing XY plane",
                    (done + w_xy * float(f)) / total,
                ),
                stop_callback=stop_callback,
                weighted_components=weighted_components,
                exp_x=exp_x,
                exp_y=exp_y,
                axial=axial_by_z[:, max_i_z],
            )
            done += w_xy

        # If YZ has not yet been computed but is requested, do it now.
        if compute_yz and (i_yz is None):
            self._check_stop(stop_callback)
            self._emit_progress(progress_callback, "Computing YZ plane", done / total)
            i_yz = self._compute_yz_plane(
                y_coords,
                z_coords,
                tables,
                verbose=verbose,
                progress_callback=lambda f: self._emit_progress(
                    progress_callback,
                    "Computing YZ plane",
                    (done + w_yz * float(f)) / total,
                ),
                stop_callback=stop_callback,
                weighted_components=weighted_components,
                exp_y=exp_y,
                axial_by_z=axial_by_z,
            )
            done += w_yz
            if z_focus is None:
                max_idx_yz = int(torch.argmax(i_yz).item())
                max_i_z = max_idx_yz // ny
                z_focus = z_coords[max_i_z]

        i_xz_norm = None if i_xz is None else self._normalize_intensity(i_xz.clone())
        i_xy_norm = None if i_xy is None else self._normalize_intensity(i_xy.clone())
        i_yz_norm = None if i_yz is None else self._normalize_intensity(i_yz.clone())

        x_coords_np = self._to_numpy(x_coords)
        y_coords_np = self._to_numpy(y_coords)
        z_coords_np = self._to_numpy(z_coords)
        if z_focus is None:
            z_focus = z_coords[int(len(z_coords) // 2)]
        z_focus_np = float(z_focus.item()) if torch.is_tensor(z_focus) else float(z_focus)
        i_xz_np = None if i_xz is None else self._to_numpy(i_xz)
        i_xy_np = None if i_xy is None else self._to_numpy(i_xy)
        i_yz_np = None if i_yz is None else self._to_numpy(i_yz)
        i_xz_norm_np = None if i_xz_norm is None else self._to_numpy(i_xz_norm)
        i_xy_norm_np = None if i_xy_norm is None else self._to_numpy(i_xy_norm)
        i_yz_norm_np = None if i_yz_norm is None else self._to_numpy(i_yz_norm)

        if (i_xy_norm_np is not None) and (i_xz_norm_np is not None):
            fwhm_metrics = self._compute_fwhm_metrics(
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
        done += w_post
        self._emit_progress(progress_callback, "Finalizing 2D results", 1.0)

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

    def simulate_psf_profile_from_pupil_field(
        self,
        pupil_field,
        size_x,
        size_y,
        entrance_pupil_diameter,
        axis,
        range_microns=2.0,
        range_microns_x=None,
        range_microns_y=None,
        range_microns_z=None,
        num_points=101,
        z_position=None,
        width_level=0.5,
        verbose=False,
        progress_callback=None,
        stop_callback=None,
    ):
        """
        Compute one on-axis PSF intensity profile without calculating full 2D planes.

        axis="z" computes I(0, 0, z). axis="x" computes I(x, 0, z0), and
        axis="y" computes I(0, y, z0). If z_position is None for x/y, z0 is
        found from the peak of the on-axis z profile.
        """
        axis = str(axis).strip().lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be one of: x, y, z.")
        if int(num_points) < 3:
            raise ValueError("num_points must be >= 3 for profile simulation.")

        pupil_field = self._to_torch_complex(pupil_field)
        if pupil_field.ndim != 2:
            raise ValueError("pupil_field must be a 2D array.")

        range_x_m, range_y_m, range_z_m = self._resolve_axis_ranges_m(
            range_microns=range_microns,
            range_microns_x=range_microns_x,
            range_microns_y=range_microns_y,
            range_microns_z=range_microns_z,
        )
        x_coords, y_coords, z_coords = self._make_scan_coords(
            range_x_m, range_y_m, range_z_m, num_points
        )

        self._check_stop(stop_callback)
        self._emit_progress(progress_callback, "Preparing profile integration tables", 0.0)
        tables = self._prepare_integration_tables(
            pupil_field,
            size_x=size_x,
            size_y=size_y,
            entrance_pupil_diameter=entrance_pupil_diameter,
        )
        weighted_components = self._weighted_components(tables)

        self._check_stop(stop_callback)
        axial_by_z = self._axial_phase_for_z(z_coords, tables)
        x_center = torch.zeros((1,), dtype=torch.float64, device=self.device)
        y_center = torch.zeros((1,), dtype=torch.float64, device=self.device)

        z_profile = None
        z_focus = None
        need_focus_scan = bool(axis == "z" or z_position is None)
        if need_focus_scan:
            self._emit_progress(progress_callback, "Computing on-axis Z profile", 0.25)
            z_profile_map = self._compute_xz_plane(
                x_center,
                z_coords,
                tables,
                verbose=verbose,
                progress_callback=lambda f: self._emit_progress(
                    progress_callback, "Computing on-axis Z profile", 0.25 + 0.45 * float(f)
                ),
                stop_callback=stop_callback,
                weighted_components=weighted_components,
                exp_x=None,
                axial_by_z=axial_by_z,
            )
            z_profile = z_profile_map[:, 0]
            z_focus_idx = int(torch.argmax(z_profile).item())
            z_focus = z_coords[z_focus_idx]

        if z_position is not None:
            z_eval = self._to_torch_float([float(z_position)])[0]
        elif z_focus is not None:
            z_eval = z_focus
        else:
            z_eval = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        if axis == "z":
            profile_t = z_profile
            coord_t = z_coords
            z_position_used = z_focus
        elif axis == "x":
            self._emit_progress(progress_callback, "Computing X profile", 0.72)
            axial_eval = self._axial_phase_for_z([float(z_eval.item())], tables)
            x_profile_map = self._compute_xz_plane(
                x_coords,
                self._to_torch_float([float(z_eval.item())]),
                tables,
                verbose=verbose,
                progress_callback=lambda f: self._emit_progress(
                    progress_callback, "Computing X profile", 0.72 + 0.25 * float(f)
                ),
                stop_callback=stop_callback,
                weighted_components=weighted_components,
                exp_x=None,
                axial_by_z=axial_eval,
            )
            profile_t = x_profile_map[0, :]
            coord_t = x_coords
            z_position_used = z_eval
        else:
            self._emit_progress(progress_callback, "Computing Y profile", 0.72)
            axial_eval = self._axial_phase_for_z([float(z_eval.item())], tables)
            y_profile_map = self._compute_yz_plane(
                y_coords,
                self._to_torch_float([float(z_eval.item())]),
                tables,
                verbose=verbose,
                progress_callback=lambda f: self._emit_progress(
                    progress_callback, "Computing Y profile", 0.72 + 0.25 * float(f)
                ),
                stop_callback=stop_callback,
                weighted_components=weighted_components,
                exp_y=None,
                axial_by_z=axial_eval,
            )
            profile_t = y_profile_map[0, :]
            coord_t = y_coords
            z_position_used = z_eval

        profile_np = self._to_numpy(profile_t)
        coord_np = self._to_numpy(coord_t)
        max_val = float(np.max(profile_np)) if np.size(profile_np) else 0.0
        profile_norm = profile_np / max_val if max_val > 0.0 else profile_np.copy()
        peak_index = int(np.argmax(profile_np)) if np.size(profile_np) else 0
        width_m = self._width_at_level_1d(coord_np, profile_np, level=width_level)

        if z_focus is None:
            z_focus_value = float(z_position_used.item()) if torch.is_tensor(z_position_used) else float(z_position_used)
        else:
            z_focus_value = float(z_focus.item()) if torch.is_tensor(z_focus) else float(z_focus)
        z_position_value = (
            float(z_position_used.item()) if torch.is_tensor(z_position_used) else float(z_position_used)
        )

        self._emit_progress(progress_callback, "Finalizing profile", 1.0)
        return {
            "axis": axis,
            "coords": coord_np,
            "coords_um": coord_np * 1e6,
            "intensity": np.asarray(profile_np, dtype=float),
            "intensity_norm": np.asarray(profile_norm, dtype=float),
            "width_level": float(width_level),
            "width_m": float(width_m),
            "width_um": float(width_m * 1e6) if np.isfinite(width_m) else np.nan,
            "peak_index": int(peak_index),
            "peak_coord_m": float(coord_np[peak_index]) if np.size(coord_np) else np.nan,
            "peak_coord_um": float(coord_np[peak_index] * 1e6) if np.size(coord_np) else np.nan,
            "peak_intensity": float(profile_np[peak_index]) if np.size(profile_np) else np.nan,
            "z_focus": float(z_focus_value),
            "z_focus_um": float(z_focus_value * 1e6),
            "z_position": float(z_position_value),
            "z_position_um": float(z_position_value * 1e6),
        }

    def simulate_psf_profile_full_pipeline(
        self,
        input_intensity,
        size_x,
        size_y,
        entrance_pupil_diameter,
        fresnel_distance,
        axis,
        use_gaussian_profile=None,
        gaussian_waist_x=None,
        gaussian_waist_y=None,
        custom_mask=None,
        ring_inner_diameter=None,
        ring_outer_diameter=None,
        range_microns=2.0,
        range_microns_x=None,
        range_microns_y=None,
        range_microns_z=None,
        num_points=101,
        z_position=None,
        width_level=0.5,
        verbose=False,
        progress_callback=None,
        stop_callback=None,
    ):
        self._emit_progress(progress_callback, "Preparing pupil field", 0.0)
        prep = self.prepare_pupil_field_from_input(
            input_intensity=input_intensity,
            size_x=size_x,
            size_y=size_y,
            entrance_pupil_diameter=entrance_pupil_diameter,
            fresnel_distance=fresnel_distance,
            use_gaussian_profile=use_gaussian_profile,
            gaussian_waist_x=gaussian_waist_x,
            gaussian_waist_y=gaussian_waist_y,
            custom_mask=custom_mask,
            ring_inner_diameter=ring_inner_diameter,
            ring_outer_diameter=ring_outer_diameter,
            progress_callback=lambda stage, frac: self._emit_progress(
                progress_callback, f"Pupil prep: {stage}", 0.45 * float(frac)
            ),
            stop_callback=stop_callback,
        )
        result = self.simulate_psf_profile_from_pupil_field(
            pupil_field=prep.get("field_at_pupil_for_psf", prep["field_at_pupil"]),
            size_x=float(prep.get("size_x_for_psf", size_x)),
            size_y=float(prep.get("size_y_for_psf", size_y)),
            entrance_pupil_diameter=entrance_pupil_diameter,
            axis=axis,
            range_microns=range_microns,
            range_microns_x=range_microns_x,
            range_microns_y=range_microns_y,
            range_microns_z=range_microns_z,
            num_points=num_points,
            z_position=z_position,
            width_level=width_level,
            verbose=verbose,
            progress_callback=lambda stage, frac: self._emit_progress(
                progress_callback, f"Profile: {stage}", 0.45 + 0.55 * float(frac)
            ),
            stop_callback=stop_callback,
        )
        result.update(
            {
                "pupil_trimmed_for_psf": bool(prep.get("pupil_trimmed_for_psf", False)),
                "pupil_trim_window": prep.get("pupil_trim_window"),
                "size_x_for_psf": float(prep.get("size_x_for_psf", size_x)),
                "size_y_for_psf": float(prep.get("size_y_for_psf", size_y)),
            }
        )
        self._emit_progress(progress_callback, "Done", 1.0)
        return result

    def simulate_full_pipeline(
        self,
        input_intensity,
        size_x,
        size_y,
        entrance_pupil_diameter,
        fresnel_distance,
        use_gaussian_profile=None,
        gaussian_waist_x=None,
        gaussian_waist_y=None,
        custom_mask=None,
        ring_inner_diameter=None,
        ring_outer_diameter=None,
        range_microns=2.0,
        range_microns_x=None,
        range_microns_y=None,
        range_microns_z=None,
        num_points=21,
        verbose=False,
        progress_callback=None,
        stop_callback=None,
    ):
        in_arr = np.asarray(input_intensity)
        if in_arr.ndim != 2:
            raise ValueError("input_intensity must be a 2D array.")
        nx_in, ny_in = in_arr.shape
        max_n = max(nx_in, ny_in, 2)
        prep_est = float(nx_in * ny_in) * (1.0 if isclose(fresnel_distance, 0.0) else log2(max_n))
        if self.integration_mode == "cartesian_jacobian":
            pupil_radius = 0.5 * float(entrance_pupil_diameter)
            area_ratio = pi * pupil_radius * pupil_radius / max(float(size_x * size_y), 1e-30)
            area_ratio = min(1.0, max(0.0, area_ratio))
            pint_est = max(1.0, area_ratio * nx_in * ny_in)
        else:
            pint_est = float(max(1, self.theta_samples * self.phi_samples))
        npnts = max(1, int(num_points))
        psf_est = float(3 * npnts * npnts) * pint_est
        total_est = max(1.0, prep_est + psf_est)
        prep_share = prep_est / total_est

        self._emit_progress(progress_callback, "Preparing pupil field", 0.0)
        prep = self.prepare_pupil_field_from_input(
            input_intensity=input_intensity,
            size_x=size_x,
            size_y=size_y,
            entrance_pupil_diameter=entrance_pupil_diameter,
            fresnel_distance=fresnel_distance,
            use_gaussian_profile=use_gaussian_profile,
            gaussian_waist_x=gaussian_waist_x,
            gaussian_waist_y=gaussian_waist_y,
            custom_mask=custom_mask,
            ring_inner_diameter=ring_inner_diameter,
            ring_outer_diameter=ring_outer_diameter,
            progress_callback=lambda stage, frac: self._emit_progress(
                progress_callback,
                f"Pupil prep: {stage}", prep_share * float(frac)
            ),
            stop_callback=stop_callback,
        )

        self._emit_progress(progress_callback, "Computing PSF planes", prep_share)
        pupil_field_for_psf = prep.get("field_at_pupil_for_psf", prep["field_at_pupil"])
        size_x_for_psf = float(prep.get("size_x_for_psf", size_x))
        size_y_for_psf = float(prep.get("size_y_for_psf", size_y))
        psf_results = self.simulate_psf_planes_from_pupil_field(
            pupil_field_for_psf,
            size_x=size_x_for_psf,
            size_y=size_y_for_psf,
            entrance_pupil_diameter=entrance_pupil_diameter,
            range_microns=range_microns,
            range_microns_x=range_microns_x,
            range_microns_y=range_microns_y,
            range_microns_z=range_microns_z,
            num_points=num_points,
            verbose=verbose,
            progress_callback=lambda stage, frac: self._emit_progress(
                progress_callback,
                f"PSF planes: {stage}", prep_share + (1.0 - prep_share) * float(frac)
            ),
            stop_callback=stop_callback,
        )

        psf_results.update(prep)
        self._emit_progress(progress_callback, "Done", 1.0)

        return psf_results


class AxialSymmetricPSFSimulator(CartesianPSFSimulator):
    """
    Axially symmetric (radial) PSF simulator for optimization.

    Assumptions
    ----------
    1) pupil modulation is radial only
    2) integration over phi is done analytically with Bessel terms
    3) numerical quadrature is performed only over theta

    This class is intended for fast axial-symmetry mask optimization where the
    mask variable is a 1D radial profile instead of a 2D map.
    """

    @staticmethod
    def _theta_trapezoid_weights(theta):
        theta = torch.as_tensor(theta, dtype=torch.float64)
        n = int(theta.numel())
        if n < 2:
            raise ValueError("theta grid must have at least 2 samples.")
        w = torch.zeros_like(theta)
        d = theta[1:] - theta[:-1]
        w[0] = 0.5 * d[0]
        w[-1] = 0.5 * d[-1]
        if n > 2:
            w[1:-1] = 0.5 * (d[:-1] + d[1:])
        return w

    @staticmethod
    def _bessel_j2(beta, j0=None, j1=None):
        # Stable J2 from recurrence with small-argument fallback.
        if j0 is None:
            j0 = torch.special.bessel_j0(beta)
        if j1 is None:
            j1 = torch.special.bessel_j1(beta)
        eps = torch.tensor(1e-10, dtype=beta.dtype, device=beta.device)
        beta_abs = torch.abs(beta)
        beta_safe = torch.where(beta_abs > eps, beta, torch.ones_like(beta))
        j2_rec = 2.0 * j1 / beta_safe - j0
        j2_small = (beta * beta) / 8.0 - (beta**4) / 96.0
        return torch.where(beta_abs > eps, j2_rec, j2_small)

    @staticmethod
    def _resample_profile_1d(profile, out_len, device):
        prof = _torch_float64(profile, device=device).reshape(-1)
        out_len = int(out_len)
        if out_len < 1:
            raise ValueError("out_len must be >= 1.")
        if int(prof.numel()) == out_len:
            return prof
        if int(prof.numel()) < 2:
            return prof.repeat(out_len)
        prof3 = prof[None, None, :]
        return F.interpolate(
            prof3,
            size=out_len,
            mode="linear",
            align_corners=True,
        )[0, 0]

    def _extract_radial_complex_profile(
        self,
        pupil_field,
        size_x,
        size_y,
        entrance_pupil_diameter,
        n_samples,
    ):
        field = self._to_torch_complex(pupil_field)
        if field.ndim != 2:
            raise ValueError("pupil_field must be 2D.")
        n_samples = int(n_samples)
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2.")
        if float(entrance_pupil_diameter) <= 0.0:
            raise ValueError("entrance_pupil_diameter must be > 0.")

        nx, ny = int(field.shape[0]), int(field.shape[1])
        x = _torch_linspace_no_endpoint(-size_x / 2, size_x / 2, nx, device=self.device)
        y = _torch_linspace_no_endpoint(-size_y / 2, size_y / 2, ny, device=self.device)
        xg, yg = torch.meshgrid(x, y, indexing="ij")

        pupil_radius = 0.5 * float(entrance_pupil_diameter)
        r_norm = torch.sqrt(xg * xg + yg * yg) / max(pupil_radius, 1e-30)
        inside = r_norm <= 1.0
        if not bool(torch.any(inside).item()):
            raise ValueError("No points found inside entrance pupil for radial profile extraction.")

        r_use = torch.clamp(r_norm[inside], 0.0, 1.0)
        idx = torch.round(r_use * float(max(1, n_samples - 1))).to(torch.int64)
        idx = torch.clamp(idx, 0, int(n_samples - 1))
        vals = field[inside]

        ones = torch.ones_like(r_use, dtype=torch.float64)
        counts = torch.zeros((n_samples,), dtype=torch.float64, device=self.device)
        counts.index_add_(0, idx, ones)

        sum_real = torch.zeros((n_samples,), dtype=torch.float64, device=self.device)
        sum_imag = torch.zeros((n_samples,), dtype=torch.float64, device=self.device)
        sum_real.index_add_(0, idx, vals.real.to(torch.float64))
        sum_imag.index_add_(0, idx, vals.imag.to(torch.float64))

        prof_real = torch.zeros((n_samples,), dtype=torch.float64, device=self.device)
        prof_imag = torch.zeros((n_samples,), dtype=torch.float64, device=self.device)
        valid = counts > 0
        prof_real[valid] = sum_real[valid] / counts[valid]
        prof_imag[valid] = sum_imag[valid] / counts[valid]

        if not bool(torch.all(valid).item()):
            valid_np = valid.detach().cpu().numpy()
            idx_np = np.arange(n_samples, dtype=np.float64)
            real_np = prof_real.detach().cpu().numpy()
            imag_np = prof_imag.detach().cpu().numpy()
            if int(np.count_nonzero(valid_np)) < 1:
                raise ValueError("Failed to construct radial complex profile from pupil field.")
            real_np[~valid_np] = np.interp(
                idx_np[~valid_np],
                idx_np[valid_np],
                real_np[valid_np],
                left=float(real_np[valid_np][0]),
                right=float(real_np[valid_np][-1]),
            )
            imag_np[~valid_np] = np.interp(
                idx_np[~valid_np],
                idx_np[valid_np],
                imag_np[valid_np],
                left=float(imag_np[valid_np][0]),
                right=float(imag_np[valid_np][-1]),
            )
            prof_real = torch.as_tensor(real_np, dtype=torch.float64, device=self.device)
            prof_imag = torch.as_tensor(imag_np, dtype=torch.float64, device=self.device)

        return prof_real.to(torch.complex128) + 1j * prof_imag.to(torch.complex128)

    def prepare_axial_symmetric_precomputed(
        self,
        pupil_field,
        size_x,
        size_y,
        entrance_pupil_diameter,
        range_microns=2.0,
        range_microns_x=None,
        range_microns_y=None,
        range_microns_z=None,
        num_points=21,
    ):
        pupil_field = self._to_torch_complex(pupil_field)
        if pupil_field.ndim != 2:
            raise ValueError("pupil_field must be 2D.")
        if int(num_points) < 3:
            raise ValueError("num_points must be >= 3.")

        range_x_m, range_y_m, range_z_m = self._resolve_axis_ranges_m(
            range_microns=range_microns,
            range_microns_x=range_microns_x,
            range_microns_y=range_microns_y,
            range_microns_z=range_microns_z,
        )
        x_coords, y_coords, z_coords = self._make_scan_coords(
            range_x_m, range_y_m, range_z_m, num_points
        )

        theta = torch.linspace(
            float(self.theta_min),
            float(self.theta_max),
            int(self.theta_samples),
            dtype=torch.float64,
            device=self.device,
        )
        theta_w = self._theta_trapezoid_weights(theta).to(device=self.device, dtype=torch.float64)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_theta_max = torch.sin(torch.tensor(float(self.theta_max), dtype=torch.float64, device=self.device))
        if float(torch.abs(sin_theta_max).item()) <= 0.0:
            raise ValueError("Invalid theta_max for axial-symmetric precomputation.")
        radial_norm = torch.clamp(sin_theta / sin_theta_max, 0.0, 1.0)

        base_field_radial = self._extract_radial_complex_profile(
            pupil_field=pupil_field,
            size_x=float(size_x),
            size_y=float(size_y),
            entrance_pupil_diameter=float(entrance_pupil_diameter),
            n_samples=int(theta.numel()),
        )

        t_s, t_p, theta_3 = self._transmission_terms(theta)
        cos_theta_3 = torch.cos(theta_3)
        sin_theta_3 = torch.sin(theta_3)
        w_abb = self._w_abb(theta)

        base_weight = (
            self.C
            * torch.sqrt(cos_theta)
            * torch.exp(1j * self.k * w_abb)
            * base_field_radial
            * sin_theta
            * theta_w
        ).to(torch.complex128)

        coeff_a = (t_p * cos_theta_3 + t_s).to(torch.complex128)
        coeff_b = (t_s - t_p * cos_theta_3).to(torch.complex128)
        coeff_z = ((-2.0j * pi) * t_p * sin_theta_3).to(torch.complex128)

        phase_scale = (self.n_i * self.k * sin_theta).to(torch.float64)
        rho_x = torch.abs(x_coords).to(torch.float64)
        rho_y = torch.abs(y_coords).to(torch.float64)
        beta_x = phase_scale[:, None] * rho_x[None, :]
        beta_y = phase_scale[:, None] * rho_y[None, :]

        j0_x = torch.special.bessel_j0(beta_x)
        j1_x = torch.special.bessel_j1(beta_x)
        j2_x = self._bessel_j2(beta_x, j0=j0_x, j1=j1_x)
        j0_y = torch.special.bessel_j0(beta_y)
        j1_y = torch.special.bessel_j1(beta_y)
        j2_y = self._bessel_j2(beta_y, j0=j0_y, j1=j1_y)

        mx_x = base_weight[:, None] * pi * (coeff_a[:, None] * j0_x + coeff_b[:, None] * j2_x)
        mz_x = base_weight[:, None] * coeff_z[:, None] * j1_x
        mx_y = base_weight[:, None] * pi * (coeff_a[:, None] * j0_y + coeff_b[:, None] * j2_y)
        mz_y = base_weight[:, None] * coeff_z[:, None] * j1_y

        axial_by_z = torch.exp(
            1j * self.k * self.n_s * cos_theta_3[:, None] * z_coords[None, :]
        ).to(torch.complex128)

        center_x = int(torch.argmin(torch.abs(x_coords)).item())
        center_y = int(torch.argmin(torch.abs(y_coords)).item())

        return {
            "radial_norm_axis": radial_norm,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "z_coords": z_coords,
            "axial_by_z": axial_by_z,
            "mx_x": mx_x.to(torch.complex128).contiguous(),
            "mz_x": mz_x.to(torch.complex128).contiguous(),
            "mx_y": mx_y.to(torch.complex128).contiguous(),
            "mz_y": mz_y.to(torch.complex128).contiguous(),
            "center_x_index": int(center_x),
            "center_y_index": int(center_y),
        }

    @staticmethod
    def _xy_from_radial_line_numpy(radial_axis_abs, radial_intensity, x_coords, y_coords):
        r = np.abs(np.asarray(radial_axis_abs, dtype=np.float64).ravel())
        i = np.asarray(radial_intensity, dtype=np.float64).ravel()
        if r.size != i.size or r.size < 2:
            raise ValueError("Invalid radial line input for XY reconstruction.")

        order = np.argsort(r)
        r = r[order]
        i = i[order]
        ur, inv = np.unique(r, return_inverse=True)
        sums = np.bincount(inv, weights=i, minlength=int(ur.size))
        cnts = np.bincount(inv, minlength=int(ur.size))
        ui = sums / np.maximum(cnts, 1)

        x = np.asarray(x_coords, dtype=np.float64).ravel()
        y = np.asarray(y_coords, dtype=np.float64).ravel()
        xg, yg = np.meshgrid(x, y, indexing="xy")
        rho = np.sqrt(xg * xg + yg * yg)
        out = np.interp(
            rho.ravel(),
            ur,
            ui,
            left=float(ui[0]),
            right=float(ui[-1]),
        ).reshape(y.size, x.size)
        return np.asarray(out, dtype=np.float64)

    def simulate_psf_planes_from_radial_profiles(
        self,
        radial_amplitude_profile,
        radial_phase_profile=None,
        precomputed=None,
        stop_callback=None,
        compute_xy=True,
        compute_xz=True,
        compute_yz=True,
    ):
        if precomputed is None:
            raise ValueError("precomputed must be provided. Call prepare_axial_symmetric_precomputed first.")
        compute_xy = bool(compute_xy)
        compute_xz = bool(compute_xz)
        compute_yz = bool(compute_yz)
        if not (compute_xy or compute_xz or compute_yz):
            raise ValueError("At least one PSF plane must be enabled.")

        self._check_stop(stop_callback)
        x_coords = precomputed["x_coords"]
        y_coords = precomputed["y_coords"]
        z_coords = precomputed["z_coords"]
        mx_x = precomputed["mx_x"]
        mz_x = precomputed["mz_x"]
        mx_y = precomputed["mx_y"]
        mz_y = precomputed["mz_y"]
        axial_by_z = precomputed["axial_by_z"]
        n_theta = int(axial_by_z.shape[0])

        amp = self._resample_profile_1d(radial_amplitude_profile, n_theta, self.device)
        amp = torch.clamp(amp, 0.0, 1.0)
        if radial_phase_profile is None:
            phase = torch.zeros_like(amp, dtype=torch.float64)
        else:
            phase = self._resample_profile_1d(radial_phase_profile, n_theta, self.device)
            phase = torch.clamp(phase, -pi, pi)
        modulation = amp.to(torch.complex128) * torch.exp(1j * phase.to(torch.complex128))

        wz = modulation[:, None] * axial_by_z  # (theta, z)
        wz_t = wz.permute(1, 0).contiguous()  # (z, theta)

        i_xz = None
        i_yz = None
        i_xy = None
        z_focus = None
        max_i_z = None

        if compute_xz:
            self._check_stop(stop_callback)
            ex_xz = torch.matmul(wz_t, mx_x)
            ez_xz = torch.matmul(wz_t, mz_x)
            i_xz = torch.abs(ex_xz) ** 2 + torch.abs(ez_xz) ** 2
            nx = int(x_coords.numel())
            max_idx = int(torch.argmax(i_xz).item())
            max_i_z = max_idx // max(1, nx)
            z_focus = z_coords[max_i_z]

        if compute_yz and ((not compute_xy) or (z_focus is None)):
            self._check_stop(stop_callback)
            ex_yz = torch.matmul(wz_t, mx_y)
            ez_yz = torch.matmul(wz_t, mz_y)
            i_yz = torch.abs(ex_yz) ** 2 + torch.abs(ez_yz) ** 2
            if z_focus is None:
                ny = int(y_coords.numel())
                max_idx_y = int(torch.argmax(i_yz).item())
                max_i_z = max_idx_y // max(1, ny)
                z_focus = z_coords[max_i_z]

        if compute_xy:
            self._check_stop(stop_callback)
            if z_focus is None:
                ix0 = int(precomputed["center_x_index"])
                ex_on = torch.matmul(wz_t, mx_x[:, ix0])
                ez_on = torch.matmul(wz_t, mz_x[:, ix0])
                i_on = torch.abs(ex_on) ** 2 + torch.abs(ez_on) ** 2
                max_i_z = int(torch.argmax(i_on).item())
                z_focus = z_coords[max_i_z]
            if max_i_z is None:
                max_i_z = int(torch.argmin(torch.abs(z_coords - z_focus)).item())

            if i_xz is not None:
                radial_axis = torch.abs(x_coords)
                radial_line = i_xz[max_i_z, :]
            elif i_yz is not None:
                radial_axis = torch.abs(y_coords)
                radial_line = i_yz[max_i_z, :]
            else:
                wz0 = wz[:, max_i_z]
                ex_line = torch.matmul(wz0[None, :], mx_x)[0]
                ez_line = torch.matmul(wz0[None, :], mz_x)[0]
                radial_axis = torch.abs(x_coords)
                radial_line = torch.abs(ex_line) ** 2 + torch.abs(ez_line) ** 2

            i_xy_np = self._xy_from_radial_line_numpy(
                radial_axis_abs=self._to_numpy(radial_axis),
                radial_intensity=self._to_numpy(radial_line),
                x_coords=self._to_numpy(x_coords),
                y_coords=self._to_numpy(y_coords),
            )
            i_xy = self._to_torch_float(i_xy_np)

        if compute_yz and (i_yz is None):
            self._check_stop(stop_callback)
            ex_yz = torch.matmul(wz_t, mx_y)
            ez_yz = torch.matmul(wz_t, mz_y)
            i_yz = torch.abs(ex_yz) ** 2 + torch.abs(ez_yz) ** 2
            if z_focus is None:
                ny = int(y_coords.numel())
                max_idx_y = int(torch.argmax(i_yz).item())
                max_i_z = max_idx_y // max(1, ny)
                z_focus = z_coords[max_i_z]

        i_xz_norm = None if i_xz is None else self._normalize_intensity(i_xz.clone())
        i_xy_norm = None if i_xy is None else self._normalize_intensity(i_xy.clone())
        i_yz_norm = None if i_yz is None else self._normalize_intensity(i_yz.clone())

        x_coords_np = self._to_numpy(x_coords)
        y_coords_np = self._to_numpy(y_coords)
        z_coords_np = self._to_numpy(z_coords)
        if z_focus is None:
            z_focus = z_coords[int(len(z_coords) // 2)]
        z_focus_np = float(z_focus.item()) if torch.is_tensor(z_focus) else float(z_focus)

        i_xz_np = None if i_xz is None else self._to_numpy(i_xz)
        i_xy_np = None if i_xy is None else self._to_numpy(i_xy)
        i_yz_np = None if i_yz is None else self._to_numpy(i_yz)
        i_xz_norm_np = None if i_xz_norm is None else self._to_numpy(i_xz_norm)
        i_xy_norm_np = None if i_xy_norm is None else self._to_numpy(i_xy_norm)
        i_yz_norm_np = None if i_yz_norm is None else self._to_numpy(i_yz_norm)

        if (i_xy_norm_np is not None) and (i_xz_norm_np is not None):
            fwhm_metrics = self._compute_fwhm_metrics(
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

    def simulate_psf_planes_from_radial_intensity(
        self,
        radial_intensity_profile,
        radial_phase_profile=None,
        precomputed=None,
        stop_callback=None,
        compute_xy=True,
        compute_xz=True,
        compute_yz=True,
    ):
        intensity = self._to_torch_float(radial_intensity_profile).reshape(-1)
        intensity = torch.clamp(intensity, 0.0, 1.0)
        amp = torch.sqrt(intensity)
        return self.simulate_psf_planes_from_radial_profiles(
            radial_amplitude_profile=amp,
            radial_phase_profile=radial_phase_profile,
            precomputed=precomputed,
            stop_callback=stop_callback,
            compute_xy=compute_xy,
            compute_xz=compute_xz,
            compute_yz=compute_yz,
        )


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    nx, ny = 256, 256
    size_x, size_y = 8.0e-3, 8.0e-3

    input_intensity = np.ones((nx, ny), dtype=float)

    simulator = CartesianPSFSimulator(
        wavelength=780e-9,
        NA=1.4,
        theta_min=0.0,
        theta_samples=61,
        phi_samples=120,
    )

    use_gaussian_profile = False
    gaussian_waist_x = 1.6e-3
    gaussian_waist_y = 1.6e-3

    results = simulator.simulate_full_pipeline(
        input_intensity=input_intensity,
        size_x=size_x,
        size_y=size_y,
        entrance_pupil_diameter=5.6e-3,
        fresnel_distance=0.30,
        use_gaussian_profile=use_gaussian_profile,
        gaussian_waist_x=gaussian_waist_x,
        gaussian_waist_y=gaussian_waist_y,
        custom_mask=None,
        ring_inner_diameter=2.6e-3,
        ring_outer_diameter=4.6e-3,
        range_microns=2.0,
        num_points=100,
        verbose=True,
    )

    x_um = results["x_coords"] * 1e6
    y_um = results["y_coords"] * 1e6
    z_um = results["z_coords"] * 1e6

    i_xy = results["I_xy_norm"]
    i_xz = results["I_xz_norm"]
    i_yz = results["I_yz_norm"]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.2, 5.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.72]},
    )

    add_panel_labels(axes)

    im_xy = axes[0, 0].imshow(
        i_xy,
        origin="lower",
        cmap="magma",
        extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    style_image_axis(
        axes[0, 0],
        title="XY",
        xlabel="x (um)",
        ylabel="y (um)",
    )
    add_compact_colorbar(fig, im_xy, axes[0, 0], "Normalized intensity")

    im_xz = axes[0, 1].imshow(
        i_xz,
        origin="lower",
        cmap="magma",
        extent=[x_um[0], x_um[-1], z_um[0], z_um[-1]],
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    style_image_axis(axes[0, 1], title="XZ", xlabel="x (um)", ylabel="z (um)")
    add_compact_colorbar(fig, im_xz, axes[0, 1], "Normalized intensity")

    im_yz = axes[0, 2].imshow(
        i_yz,
        origin="lower",
        cmap="magma",
        extent=[y_um[0], y_um[-1], z_um[0], z_um[-1]],
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    style_image_axis(axes[0, 2], title="YZ", xlabel="y (um)", ylabel="z (um)")
    add_compact_colorbar(fig, im_yz, axes[0, 2], "Normalized intensity")

    ep_x_mm = results["entrance_pupil_x_coords"] * 1e3
    ep_y_mm = results["entrance_pupil_y_coords"] * 1e3
    ep_extent = [ep_y_mm[0], ep_y_mm[-1], ep_x_mm[0], ep_x_mm[-1]]

    im_ep_before = axes[1, 0].imshow(
        results["intensity_at_entrance_pupil_norm"],
        origin="lower",
        cmap="magma",
        extent=ep_extent,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    style_image_axis(axes[1, 0], title="Fresnel", xlabel="y (mm)", ylabel="x (mm)")
    add_compact_colorbar(fig, im_ep_before, axes[1, 0], "Normalized intensity")

    im_ep_after = axes[1, 1].imshow(
        results["intensity_after_pupil_mask_norm"],
        origin="lower",
        cmap="magma",
        extent=ep_extent,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    style_image_axis(axes[1, 1], title="Pupil", xlabel="y (mm)", ylabel="x (mm)")
    add_compact_colorbar(fig, im_ep_after, axes[1, 1], "Normalized intensity")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        1.0,
        "Summary",
        fontweight="bold",
        va="top",
        ha="left",
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].text(
        0.0,
        0.84,
        f"Focus z = {results['z_focus'] * 1e6:.3f} um",
        va="top",
        ha="left",
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].text(
        0.0,
        0.68,
        f"FWHM_x = {results['fwhm_x_um']:.3f} um",
        va="top",
        ha="left",
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].text(
        0.0,
        0.52,
        f"FWHM_y = {results['fwhm_y_um']:.3f} um",
        va="top",
        ha="left",
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].text(
        0.0,
        0.36,
        f"FWHM_z = {results['fwhm_z_um']:.3f} um",
        va="top",
        ha="left",
        transform=axes[1, 2].transAxes,
    )

    x_profile = results["x_profile_at_y0"]
    y_profile = results["y_profile_at_x0"]
    z_profile = results["z_profile_at_x0"]

    axes[2, 0].plot(x_um, x_profile, color="#1B3B6F", linewidth=1.35, zorder=3)
    axes[2, 0].axhline(0.5, color="#4D4D4D", linestyle="--", linewidth=0.9, zorder=2)
    style_profile_axis(
        axes[2, 0],
        title="X",
        xlabel="x (um)",
        ylabel="Normalized intensity",
    )
    axes[2, 0].set_ylim(0.0, 1.05)

    axes[2, 1].plot(y_um, y_profile, color="#D55E00", linewidth=1.35, zorder=3)
    axes[2, 1].axhline(0.5, color="#4D4D4D", linestyle="--", linewidth=0.9, zorder=2)
    style_profile_axis(
        axes[2, 1],
        title="Y",
        xlabel="y (um)",
        ylabel="Normalized intensity",
    )
    axes[2, 1].set_ylim(0.0, 1.05)

    axes[2, 2].plot(z_um, z_profile, color="#007C73", linewidth=1.35, zorder=3)
    axes[2, 2].axhline(0.5, color="#4D4D4D", linestyle="--", linewidth=0.9, zorder=2)
    style_profile_axis(
        axes[2, 2],
        title="Z",
        xlabel="z (um)",
        ylabel="Normalized intensity",
    )
    axes[2, 2].set_ylim(0.0, 1.05)

    fig.align_labels()
    if "agg" in matplotlib.get_backend().lower():
        plt.savefig("psf_demo.pdf")
        plt.savefig("psf_demo.png")
    else:
        plt.show()
