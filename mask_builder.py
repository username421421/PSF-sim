import numpy as np
from scipy.ndimage import zoom


class MaskBuilderModel:
    """Stateful mask builder using physical mm coordinates."""

    def __init__(self, nx, ny, size_x_mm, size_y_mm, init_value=1.0, max_history=50):
        self.max_history = int(max_history)
        if self.max_history < 1:
            self.max_history = 1
        self.history = []

        self.nx = 0
        self.ny = 0
        self.size_x_mm = 0.0
        self.size_y_mm = 0.0
        self.mask = None

        self.set_geometry(nx, ny, size_x_mm, size_y_mm, preserve=False)
        self.clear(value=init_value)

    @staticmethod
    def _validate_geometry(nx, ny, size_x_mm, size_y_mm):
        nx_i = int(nx)
        ny_i = int(ny)
        sx = float(size_x_mm)
        sy = float(size_y_mm)
        if nx_i <= 0 or ny_i <= 0:
            raise ValueError("Mask dimensions nx and ny must be positive.")
        if sx <= 0 or sy <= 0:
            raise ValueError("Mask physical sizes size_x_mm and size_y_mm must be positive.")
        return nx_i, ny_i, sx, sy

    @staticmethod
    def _as_2d_float(mask_2d):
        arr = np.asarray(mask_2d, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Mask must be a 2D array.")
        if arr.size == 0:
            raise ValueError("Mask cannot be empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Mask contains NaN or Inf.")
        return np.clip(arr, 0.0, 1.0)

    def set_geometry(self, nx, ny, size_x_mm, size_y_mm, preserve=True):
        nx_i, ny_i, sx, sy = self._validate_geometry(nx, ny, size_x_mm, size_y_mm)
        old = None if self.mask is None else np.asarray(self.mask, dtype=float)

        self.nx = nx_i
        self.ny = ny_i
        self.size_x_mm = sx
        self.size_y_mm = sy

        if old is None or (not bool(preserve)):
            self.mask = np.zeros((self.nx, self.ny), dtype=float)
            return

        if old.shape == (self.nx, self.ny):
            self.mask = np.clip(old, 0.0, 1.0)
            return

        factors = (self.nx / old.shape[0], self.ny / old.shape[1])
        resized = zoom(old, factors, order=1, mode="nearest", prefilter=False)
        out = np.zeros((self.nx, self.ny), dtype=float)
        nx_min = min(self.nx, resized.shape[0])
        ny_min = min(self.ny, resized.shape[1])
        out[:nx_min, :ny_min] = resized[:nx_min, :ny_min]
        self.mask = np.clip(out, 0.0, 1.0)

    def set_mask(self, mask_2d):
        arr = self._as_2d_float(mask_2d)
        self.nx, self.ny = int(arr.shape[0]), int(arr.shape[1])
        self.mask = arr.copy()

    def get_mask(self):
        return np.asarray(self.mask, dtype=float).copy()

    def clear(self, value=0.0):
        val = float(np.clip(value, 0.0, 1.0))
        self.mask = np.full((self.nx, self.ny), val, dtype=float)

    def push_history(self):
        self.history.append(self.get_mask())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def undo(self):
        if not self.history:
            return False
        self.mask = self.history.pop()
        self.nx, self.ny = self.mask.shape
        return True

    def _coord_grids_mm(self):
        x = np.linspace(-0.5 * self.size_x_mm, 0.5 * self.size_x_mm, self.nx, endpoint=False)
        y = np.linspace(-0.5 * self.size_y_mm, 0.5 * self.size_y_mm, self.ny, endpoint=False)
        return x[:, None], y[None, :]

    def _apply_region(self, region, mode, value=1.0):
        mode_norm = str(mode).strip().lower()
        if mode_norm not in {"add", "erase"}:
            raise ValueError("mode must be 'add' or 'erase'.")
        if mode_norm == "add":
            self.mask[region] = float(np.clip(value, 0.0, 1.0))
        else:
            self.mask[region] = 0.0

    def add_circle(self, center_x_mm, center_y_mm, radius_mm, value=1.0, mode="add"):
        r = float(radius_mm)
        if r <= 0:
            raise ValueError("Circle radius must be positive.")
        cx = float(center_x_mm)
        cy = float(center_y_mm)
        xg, yg = self._coord_grids_mm()
        region = (xg - cx) ** 2 + (yg - cy) ** 2 <= r * r
        self._apply_region(region, mode=mode, value=value)

    def add_ring(
        self,
        center_x_mm,
        center_y_mm,
        inner_radius_mm,
        outer_radius_mm,
        value=1.0,
        mode="add",
    ):
        rin = float(inner_radius_mm)
        rout = float(outer_radius_mm)
        if rin < 0:
            raise ValueError("Ring inner radius must be >= 0.")
        if rout <= rin:
            raise ValueError("Ring outer radius must be greater than inner radius.")
        cx = float(center_x_mm)
        cy = float(center_y_mm)
        xg, yg = self._coord_grids_mm()
        d2 = (xg - cx) ** 2 + (yg - cy) ** 2
        region = (d2 >= rin * rin) & (d2 <= rout * rout)
        self._apply_region(region, mode=mode, value=value)

    def add_rectangle(self, center_x_mm, center_y_mm, width_mm, height_mm, value=1.0, mode="add"):
        w = float(width_mm)
        h = float(height_mm)
        if w <= 0 or h <= 0:
            raise ValueError("Rectangle width and height must be positive.")
        cx = float(center_x_mm)
        cy = float(center_y_mm)
        xg, yg = self._coord_grids_mm()
        region = (np.abs(xg - cx) <= 0.5 * w) & (np.abs(yg - cy) <= 0.5 * h)
        self._apply_region(region, mode=mode, value=value)
