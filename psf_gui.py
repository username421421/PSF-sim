if __name__ == "__main__":
    import sys as _early_sys

    if len(_early_sys.argv) > 1 and str(_early_sys.argv[1]).lower() != "gui":
        from psf_cli import main as _early_cli_main

        raise SystemExit(_early_cli_main(_early_sys.argv[1:]))

import contextlib
import csv
import copy
import threading
import traceback
import tkinter as tk
import os
import sys
import json
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
import torch.nn.functional as F

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from matplotlib.transforms import Bbox

from mask_builder import MaskBuilderModel
from PSF_simulator import CartesianPSFSimulator, SimulationStoppedError
from psf_cli import (
    load_result as load_result_file,
    read_mask as read_mask_file,
    resize_mask as resize_mask_array,
    run_simulation,
    save_result as save_result_file,
)
from psf_models import (
    ABERRATION_MODELS,
    INTEGRATION_MODES,
    OBJECTIVE_SPECS,
    PARAM_SPECS,
    LEGACY_UI_KEYS,
    TORCH_DEVICES,
    objective_ui_values,
    upgrade_ui_values,
)
from psf_runtime import (
    create_simulator,
    enforce_resource_limits,
    estimate_resources,
)
from plot_style import (
    add_compact_colorbar,
    apply_publication_style,
    intensity_norm,
    make_figure,
    style_image_axis,
    style_profile_axis,
)

try:
    from psf_mask_optimizer import (
        OptimizationStoppedError,
        OptimizerConfig,
        PostFresnelMaskOptimizer,
        save_result_npz,
    )

    HAS_SCIPY_OPTIMIZER = True
    SCIPY_OPTIMIZER_IMPORT_ERROR = ""
except Exception as _opt_import_exc:
    OptimizationStoppedError = RuntimeError
    OptimizerConfig = None
    PostFresnelMaskOptimizer = None
    save_result_npz = None
    HAS_SCIPY_OPTIMIZER = False
    SCIPY_OPTIMIZER_IMPORT_ERROR = str(_opt_import_exc)

try:
    import tifffile
except Exception:
    tifffile = None

try:
    import pyvista as pv

    HAS_PYVISTA = True
    PYVISTA_IMPORT_ERROR = ""
except Exception as _pyvista_import_exc:
    HAS_PYVISTA = False
    PYVISTA_IMPORT_ERROR = str(_pyvista_import_exc)
    pv = None

try:
    from PyQt5 import QtWidgets
    from pyvistaqt import BackgroundPlotter

    HAS_PYVISTA_QT = True
    PYVISTA_QT_IMPORT_ERROR = ""
except Exception as _pyvista_qt_import_exc:
    HAS_PYVISTA_QT = False
    PYVISTA_QT_IMPORT_ERROR = str(_pyvista_qt_import_exc)
    QtWidgets = None
    BackgroundPlotter = None

HAS_PYVISTA_WINDOW = bool(HAS_PYVISTA and HAS_PYVISTA_QT)

apply_publication_style()


class _QueuedProgress:
    """Hands worker-thread progress updates to the Tk event loop."""

    def __init__(self, root, callback):
        self.root = root
        self.callback = callback

    def emit(self, percent, text, force=False):
        self.root.after(0, lambda: self.callback(percent, text))


class PSFSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PSF Simulator GUI (2D + 3D)")
        self.root.geometry("1720x980")

        self.mask_array = None
        self.post_fresnel_mask_array = None
        self.opt_input_mask = None
        self.config_extensions = {}
        self.mask_rotation_quadrants = 0  # +1 means +90 deg CCW.
        self.mask_path = None
        self.results_2d = None
        self.results_3d = None
        self.last_params = None
        self.last_simulator = None
        self.mask_builder_model = None
        self.mask_builder_action_buttons = []
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        # Keep simulation data beside the application.  This preserves the
        # established portable workflow and makes saved masks/results visible
        # without an implicit AppData migration.
        self.user_data_dir = self.project_dir
        self.mask_library_dir = os.path.join(self.user_data_dir, "mask library")
        self.mask_load_dir = self.project_dir
        self.saved_simulation_dir = os.path.join(self.user_data_dir, "saved simulation")
        self.objective_lens_dir = os.path.join(self.user_data_dir, "objective lens")
        self.optimized_mask_dir = os.path.join(self.user_data_dir, "optimized mask")
        os.makedirs(self.mask_library_dir, exist_ok=True)
        os.makedirs(self.saved_simulation_dir, exist_ok=True)
        os.makedirs(self.objective_lens_dir, exist_ok=True)
        os.makedirs(self.optimized_mask_dir, exist_ok=True)
        self.last_config_path = os.path.join(
            self.saved_simulation_dir, "last_used_simulation_config.json"
        )
        self.is_running = False
        self.stop_requested = False
        self._iso_refresh_job = None
        self._surface_cache = {}
        self._surface_cache_order = []
        self._surface_cache_max = 32
        self._surface_cache_max_bytes = 256 * 1024**2
        self._max_3d_display_dim = 96
        self._max_3d_faces = 30000
        self._qt_pump_interval_ms = 5
        self._qt_pump_passes = 2
        self._cursor_axes_map = {}
        self._cursor_cid = None
        self._cursor_press_cid = None
        self._cursor_release_cid = None
        self._profile_axes_map = {}
        self._profile_cursor_artists = {}
        self._profile_drag_ax = None
        self._axes_export_map = {}
        self._width_metrics_cache = {}
        self._width_plot_artists = {}
        self._width_level_step = 0.001
        self._active_width_level = 0.50
        self._width_level_source = "slider"
        self._width_entry_job = None
        self._mask_preview_job = None
        self.opt_is_running = False
        self.opt_stop_requested = False
        self.opt_result = None
        self.opt_last_params = None
        self.opt_last_summary = None
        self.opt_live_amp_preview = None
        self.opt_live_phase_preview = None
        self.opt_live_full_shape = None
        self.opt_live_objective = None
        self.opt_live_widths = None
        self._opt_live_refresh_job = None
        self._opt_live_preview_dirty = False
        self._pyvista_plotter = None
        self._pyvista_camera_position = None
        self._qt_app = None
        self._qt_pump_job = None

        self._setup_style()
        self._build_layout()
        self.root.bind("<Destroy>", self._on_root_destroy, add="+")
        loaded_last_config = self._load_last_used_config()
        self._init_empty_plots()
        if loaded_last_config:
            self.status_var.set("Loaded last used simulation settings.")

    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.root.configure(bg="#f3f5f8")
        style.configure("TFrame", background="#f3f5f8")
        style.configure("TLabel", background="#f3f5f8", font=("Segoe UI", 10))
        style.configure("TLabelframe", background="#f3f5f8", bordercolor="#c7d1df")
        style.configure(
            "TLabelframe.Label", background="#f3f5f8", foreground="#1e2a38", font=("Segoe UI", 10, "bold")
        )
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Small.TButton", font=("Segoe UI", 9), padding=(6, 3))
        style.configure("TEntry", padding=4)
        style.map(
            "LinkedMedium.TEntry",
            fieldbackground=[("disabled", "#e4e8ed")],
            foreground=[("disabled", "#737b87")],
        )
        style.configure("TRadiobutton", background="#f3f5f8", font=("Segoe UI", 10))
        style.configure("TCheckbutton", background="#f3f5f8", font=("Segoe UI", 10))
        style.configure("TNotebook", background="#f3f5f8", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(10, 6))
        style.configure(
            "Primary.TButton",
            foreground="#ffffff",
            background="#2a6fdb",
            bordercolor="#2a6fdb",
            focusthickness=3,
            focuscolor="#2a6fdb",
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#225db7"), ("disabled", "#8aa9de")],
            foreground=[("disabled", "#ecf1fb")],
        )

    def _build_layout(self):
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(pane, padding=6)
        plot_frame = ttk.Frame(pane, padding=6)
        pane.add(control_frame, weight=0)
        pane.add(plot_frame, weight=1)

        # Scrollable control panel so all parameters remain reachable.
        self.controls_canvas = tk.Canvas(
            control_frame, highlightthickness=0, bg="#f3f5f8", bd=0
        )
        controls_scrollbar = ttk.Scrollbar(
            control_frame, orient=tk.VERTICAL, command=self.controls_canvas.yview
        )
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls_inner = ttk.Frame(self.controls_canvas, padding=6)
        self._controls_window = self.controls_canvas.create_window(
            (0, 0), window=controls_inner, anchor="nw"
        )

        controls_inner.bind("<Configure>", self._on_controls_frame_configure)
        self.controls_canvas.bind("<Configure>", self._on_controls_canvas_configure)
        self.controls_canvas.bind("<Enter>", self._bind_controls_mousewheel)
        self.controls_canvas.bind("<Leave>", self._unbind_controls_mousewheel)

        self._build_controls(controls_inner)
        self._build_plots(plot_frame)

    def _build_controls(self, parent):
        self.status_var = tk.StringVar(value="Ready")

        self.torch_device_var = tk.StringVar(value="cpu")
        compute_group = ttk.LabelFrame(parent, text="Compute", padding=8)
        compute_group.pack(fill="x", pady=4)
        ttk.Label(compute_group, text="Torch device").pack(anchor="w")
        self.torch_device_combo = ttk.Combobox(
            compute_group,
            textvariable=self.torch_device_var,
            state="readonly",
            values=TORCH_DEVICES,
        )
        self.torch_device_combo.pack(fill="x", pady=(0, 4))

        mask_group = ttk.LabelFrame(parent, text="Mask Input (2D array in [0,1])", padding=8)
        mask_group.pack(fill="x", pady=4)

        self.mask_path_var = tk.StringVar(value="")
        ttk.Entry(mask_group, textvariable=self.mask_path_var).pack(fill="x")
        mask_btn_row = ttk.Frame(mask_group)
        mask_btn_row.pack(fill="x", pady=(6, 2))
        self.load_mask_btn = ttk.Button(
            mask_btn_row,
            text="Load Mask…",
            command=self._load_mask,
        )
        self.load_mask_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 3))
        self.remove_mask_btn = ttk.Button(
            mask_btn_row,
            text="Remove Mask",
            command=self._remove_loaded_mask,
        )
        self.remove_mask_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(3, 0))
        self.mask_info_var = tk.StringVar(value="No file loaded. Using all-pass mask.")
        self._wrap_label(ttk.Label(mask_group, textvariable=self.mask_info_var, wraplength=330)).pack(fill="x")
        self.nx_var = tk.IntVar(value=256)
        self.ny_var = tk.IntVar(value=256)
        self.size_x_mm_var = tk.DoubleVar(value=8.0)
        self.size_y_mm_var = tk.DoubleVar(value=8.0)
        self._entry(mask_group, "Grid Nx", self.nx_var)
        self._entry(mask_group, "Grid Ny", self.ny_var)
        self._entry(mask_group, "Mask size X (mm)", self.size_x_mm_var)
        self._entry(mask_group, "Mask size Y (mm)", self.size_y_mm_var)
        self.sampling_info_var = tk.StringVar(value="")
        ttk.Label(mask_group, text="Fresnel sampling").pack(anchor="w", pady=(4, 0))
        self._wrap_label(ttk.Label(mask_group, textvariable=self.sampling_info_var, wraplength=330)).pack(
            fill="x", pady=(0, 4)
        )
        self.mask_offset_x_mm_var = tk.DoubleVar(value=0.0)
        self.mask_offset_y_mm_var = tk.DoubleVar(value=0.0)
        self._entry(mask_group, "Mask X offset (mm)", self.mask_offset_x_mm_var)
        self._entry(mask_group, "Mask Y offset (mm)", self.mask_offset_y_mm_var)
        rot_row = ttk.Frame(mask_group)
        rot_row.pack(fill="x", pady=(2, 2))
        ttk.Label(rot_row, text="Rotate mask 90°").pack(side=tk.LEFT)
        self.rotate_cw_btn = ttk.Button(
            rot_row, text="↻ CW", width=6, style="Small.TButton", command=self._rotate_mask_cw_90
        )
        self.rotate_cw_btn.pack(side=tk.RIGHT, padx=(3, 0))
        self.rotate_ccw_btn = ttk.Button(
            rot_row, text="↺ CCW", width=6, style="Small.TButton", command=self._rotate_mask_ccw_90
        )
        self.rotate_ccw_btn.pack(side=tk.RIGHT)
        self.mask_orientation_var = tk.StringVar(value="Mask orientation: 0 deg")
        self._wrap_label(ttk.Label(mask_group, textvariable=self.mask_orientation_var, wraplength=330)).pack(
            fill="x", pady=(0, 2)
        )
        self._wrap_label(ttk.Label(
            mask_group,
            text="Shifted-out regions are treated as all-pass (transmission = 1).",
            wraplength=330,
        )).pack(fill="x", pady=(0, 2))
        preview_group = ttk.LabelFrame(
            mask_group, text="Mask Preview (orientation + offset)", padding=6
        )
        preview_group.pack(fill="x", pady=(4, 0))
        self.fig_mask_input = make_figure(figsize=(3.4, 2.8), dpi=150)
        self.canvas_mask_input = FigureCanvasTkAgg(self.fig_mask_input, master=preview_group)
        self.canvas_mask_input.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        opt_group = ttk.LabelFrame(parent, text="Optics", padding=8)
        opt_group.pack(fill="x", pady=4)
        self.wavelength_nm_var = tk.DoubleVar(value=780.0)
        self.na_var = tk.DoubleVar(value=1.4)
        self.pupil_d_mm_var = tk.DoubleVar(value=5.6)
        self.theta_samples_var = tk.IntVar(value=61)
        self.phi_samples_var = tk.IntVar(value=120)
        self.integration_mode_var = tk.StringVar(value="cartesian_jacobian")
        self.aberration_model_var = tk.StringVar(value="legacy")

        self.exp_immersion_ri_var = tk.DoubleVar(value=1.515)
        self.exp_coverslip_ri_var = tk.DoubleVar(value=1.515)
        self.exp_coverslip_thickness_um_var = tk.DoubleVar(value=170.0)
        self.exp_resin_ri_var = tk.DoubleVar(value=1.474)
        self.exp_resin_thickness_um_var = tk.DoubleVar(value=1.0)
        self.dip_in_enabled_var = tk.BooleanVar(value=False)
        self.aberration_model_description_var = tk.StringVar(value="")
        self._dip_in_previous_values = None

        self.design_immersion_ri_var = tk.DoubleVar(value=1.515)
        self.design_immersion_gap_um_var = tk.DoubleVar(value=90.0)
        self.design_coverslip_ri_var = tk.DoubleVar(value=1.515)
        self.design_coverslip_thickness_um_var = tk.DoubleVar(value=170.0)
        self.design_resin_ri_var = tk.DoubleVar(value=1.515)

        self._entry(opt_group, "Wavelength (nm)", self.wavelength_nm_var)
        self._entry(opt_group, "NA", self.na_var)
        self._entry(opt_group, "Entrance pupil diameter (mm)", self.pupil_d_mm_var)
        ttk.Label(opt_group, text="Integration mode").pack(anchor="w")
        self.integration_mode_combo = ttk.Combobox(
            opt_group,
            textvariable=self.integration_mode_var,
            state="readonly",
            values=INTEGRATION_MODES,
        )
        self.integration_mode_combo.pack(fill="x", pady=(0, 4))
        self.integration_mode_combo.bind("<<ComboboxSelected>>", self._on_integration_mode_change)
        self.theta_samples_entry = self._entry(opt_group, "Theta samples (polar mode)", self.theta_samples_var)
        self.phi_samples_entry = self._entry(opt_group, "Phi samples (polar mode)", self.phi_samples_var)
        self._on_integration_mode_change()
        ttk.Label(opt_group, text="Aberration model").pack(anchor="w")
        self.aberration_model_combo = ttk.Combobox(
            opt_group,
            textvariable=self.aberration_model_var,
            state="readonly",
            values=ABERRATION_MODELS,
        )
        self.aberration_model_combo.pack(fill="x", pady=(0, 2))
        self._wrap_label(ttk.Label(
            opt_group, textvariable=self.aberration_model_description_var, wraplength=330,
        )).pack(fill="x", pady=(0, 4))
        self.aberration_model_var.trace_add("write", self._update_model_description)
        self._update_model_description()
        design_media_group = ttk.LabelFrame(opt_group, text="Design Values", padding=6)
        design_media_group.pack(fill="x", pady=(0, 2))

        self._entry(design_media_group, "Immersion medium RI", self.design_immersion_ri_var)
        self._entry(design_media_group, "Immersion gap / free WD (um)", self.design_immersion_gap_um_var)
        self._entry(design_media_group, "Coverslip RI", self.design_coverslip_ri_var)
        self._entry(design_media_group, "Coverslip thickness (um)", self.design_coverslip_thickness_um_var)
        objective_lens_row = ttk.Frame(opt_group)
        objective_lens_row.pack(fill="x", pady=(4, 0))
        self.store_objective_lens_btn = ttk.Button(
            objective_lens_row, text="Store Objective Lens", command=self._save_objective_lens_dialog
        )
        self.store_objective_lens_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 3))
        self.load_objective_lens_btn = ttk.Button(
            objective_lens_row, text="Load Objective Lens", command=self._load_objective_lens_dialog
        )
        self.load_objective_lens_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(3, 0))

        exp_media_group = ttk.LabelFrame(parent, text="Experimental Values", padding=8)
        exp_media_group.pack(fill="x", pady=4)
        self.dip_in_check = ttk.Checkbutton(
            exp_media_group, text="Dip-in lens (directly in resin)",
            variable=self.dip_in_enabled_var, command=self._on_dip_in_toggle,
        )
        self.dip_in_check.pack(anchor="w", pady=(0, 4))
        self._dip_in_linked_entries = [
            self._entry(exp_media_group, "Immersion medium RI", self.exp_immersion_ri_var),
            self._entry(exp_media_group, "Coverslip RI", self.exp_coverslip_ri_var),
            self._entry(exp_media_group, "Coverslip thickness (um)", self.exp_coverslip_thickness_um_var),
        ]
        self._entry(exp_media_group, "Resin RI", self.exp_resin_ri_var)
        self._dip_in_linked_entries.append(self._entry(
            exp_media_group,
            "Resin thickness / PSF position (um)",
            self.exp_resin_thickness_um_var,
        ))
        for entry in self._dip_in_linked_entries:
            entry.configure(style="LinkedMedium.TEntry")
        self._wrap_label(ttk.Label(
            exp_media_group,
            text="Dip-in uses resin as the immersion medium (Gibson-Lanni model), with the coverslip beyond focus. Turning it off restores your previous settings.",
            wraplength=330,
        )).pack(fill="x", pady=(0, 4))
        self.exp_resin_ri_var.trace_add("write", self._sync_dip_in_media)

        propagation_group = ttk.LabelFrame(parent, text="Propagation", padding=8)
        propagation_group.pack(fill="x", pady=4)
        self.fresnel_z_m_var = tk.DoubleVar(value=0.30)
        self.use_gaussian_var = tk.BooleanVar(value=False)
        self.gauss_fwhm_x_mm_var = tk.DoubleVar(value=2.7)
        self.gauss_fwhm_y_mm_var = tk.DoubleVar(value=2.7)
        self._entry(propagation_group, "Fresnel propagation z (m)", self.fresnel_z_m_var)
        ttk.Checkbutton(
            propagation_group, text="Apply Gaussian envelope", variable=self.use_gaussian_var
        ).pack(anchor="w")
        self._entry(propagation_group, "Intensity FWHM x (mm)", self.gauss_fwhm_x_mm_var)
        self._entry(propagation_group, "Intensity FWHM y (mm)", self.gauss_fwhm_y_mm_var)

        sim_group = ttk.LabelFrame(parent, text="Simulation Settings", padding=8)
        sim_group.pack(fill="x", pady=4)
        self.range_x_um_var = tk.DoubleVar(value=2.0)
        self.range_y_um_var = tk.DoubleVar(value=2.0)
        self.range_z_um_var = tk.DoubleVar(value=2.0)
        self.points_2d_var = tk.IntVar(value=101)
        self.points_3d_var = tk.IntVar(value=61)
        self.z_chunk_var = tk.IntVar(value=8)
        self._entry(sim_group, "PSF range X +/- (um)", self.range_x_um_var)
        self._entry(sim_group, "PSF range Y +/- (um)", self.range_y_um_var)
        self._entry(sim_group, "PSF range Z +/- (um)", self.range_z_um_var)
        self._entry(sim_group, "2D points", self.points_2d_var)
        self._entry(sim_group, "3D points", self.points_3d_var)
        self._entry(sim_group, "3D z chunk", self.z_chunk_var)

        display_group = ttk.LabelFrame(parent, text="Display", padding=8)
        display_group.pack(fill="x", pady=4)

        self.plot_mode_var = tk.StringVar(value="I")
        self.normalize_var = tk.BooleanVar(value=True)
        self.profile_cursor_enabled_var = tk.BooleanVar(value=False)
        self.show_3d_grid_var = tk.BooleanVar(value=False)
        mode_row = ttk.Frame(display_group)
        mode_row.pack(fill="x")
        ttk.Label(mode_row, text="PSF mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            mode_row, text="I", value="I", variable=self.plot_mode_var, command=self._refresh_plots
        ).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Radiobutton(
            mode_row,
            text="I^2 (2-photon)",
            value="I^2",
            variable=self.plot_mode_var,
            command=self._refresh_plots,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(
            mode_row,
            text="Normalize",
            variable=self.normalize_var,
            command=self._refresh_plots,
        ).pack(side=tk.RIGHT)

        ttk.Checkbutton(
            display_group,
            text="Enable profile cursor (drag on X/Y/Z profiles)",
            variable=self.profile_cursor_enabled_var,
            command=self._on_profile_cursor_toggle,
        ).pack(anchor="w", pady=(6, 0))
        ttk.Checkbutton(
            display_group,
            text="Show 3D coordinate grid",
            variable=self.show_3d_grid_var,
            command=self._on_3d_grid_toggle,
        ).pack(anchor="w", pady=(4, 0))

        self.width_level_var = tk.StringVar(value="0.5")
        self.width_level_slider_var = tk.DoubleVar(value=self._active_width_level)
        self.iso_level_var = tk.DoubleVar(value=0.50)
        self.width_ratio_var = tk.StringVar(value="Width ratios: z/x = N/A, z/y = N/A")

        ttk.Label(display_group, text="Peak width level (0..1)").pack(anchor="w", pady=(8, 0))
        width_row = ttk.Frame(display_group)
        width_row.pack(fill="x")
        self.width_level_label = ttk.Label(width_row, text="0.5")
        self.width_level_label.pack(side=tk.LEFT)
        self.width_level_entry = ttk.Entry(width_row, width=8, textvariable=self.width_level_var)
        self.width_level_entry.pack(side=tk.RIGHT)
        self.width_level_entry.bind("<KeyRelease>", self._on_width_level_entry_key_release)
        self.width_level_entry.bind("<Return>", self._on_width_level_entry)
        self.width_level_entry.bind("<FocusOut>", self._on_width_level_entry)
        ttk.Scale(
            display_group,
            from_=0.01,
            to=0.99,
            variable=self.width_level_slider_var,
            command=self._on_width_level,
        ).pack(fill="x")
        ttk.Label(
            display_group,
            textvariable=self.width_ratio_var,
            justify=tk.LEFT,
            wraplength=300,
        ).pack(anchor="w", pady=(2, 0))

        ttk.Label(display_group, text="3D surface level (0..1)").pack(anchor="w", pady=(8, 0))
        iso_row = ttk.Frame(display_group)
        iso_row.pack(fill="x")
        self.iso_level_label = ttk.Label(iso_row, text="0.50")
        self.iso_level_label.pack(side=tk.LEFT)
        self.iso_level_entry = ttk.Entry(iso_row, width=8, textvariable=self.iso_level_var)
        self.iso_level_entry.pack(side=tk.RIGHT)
        self.iso_level_entry.bind("<Return>", self._on_iso_level_entry)
        self.iso_level_entry.bind("<FocusOut>", self._on_iso_level_entry)
        ttk.Scale(
            display_group,
            from_=0.01,
            to=0.99,
            variable=self.iso_level_var,
            command=self._on_iso_level,
        ).pack(fill="x")

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill="x", pady=(10, 4))
        self.run_2d_btn = ttk.Button(
            btn_row, text="Run 2D Quick", command=self._run_2d, style="Primary.TButton"
        )
        self.run_2d_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 3))
        self.run_3d_btn = ttk.Button(
            btn_row, text="Run 3D Full", command=self._run_3d, style="Primary.TButton"
        )
        self.run_3d_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(3, 0))
        self.stop_btn = ttk.Button(btn_row, text="Stop", command=self._stop_simulation)
        self.stop_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(3, 0))
        self.stop_btn.configure(state="disabled")

        save_row = ttk.Frame(parent)
        save_row.pack(fill="x", pady=(0, 4))
        self.save_btn = ttk.Button(save_row, text="Save Result", command=self._save_result_dialog)
        self.save_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 3))
        self.load_btn = ttk.Button(save_row, text="Load Result", command=self._load_result_dialog)
        self.load_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(3, 0))

        # Two buttons per row so no label is clipped at the sidebar width.
        export_row = ttk.Frame(parent)
        export_row.pack(fill="x", pady=(0, 4))
        self.export_csv_btn = ttk.Button(
            export_row,
            text="Export CSV",
            command=self._export_current_data_csv_dialog,
        )
        self.export_csv_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 3))
        self.show_3d_btn = ttk.Button(export_row, text="Show 3D Window", command=self._show_3d_plot_window)
        self.show_3d_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(3, 0))
        self.show_3d_btn.configure(state="disabled")

        config_row = ttk.Frame(parent)
        config_row.pack(fill="x", pady=(0, 4))
        self.save_config_btn = ttk.Button(
            config_row,
            text="Save Config",
            command=self._save_simulation_config_dialog,
        )
        self.save_config_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 3))
        self.load_config_btn = ttk.Button(
            config_row,
            text="Load Config",
            command=self._load_simulation_config_dialog,
        )
        self.load_config_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(3, 0))

        prog_row = ttk.Frame(parent)
        prog_row.pack(fill="x", pady=(6, 2))
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_pct_var = tk.StringVar(value="0.0%")
        ttk.Label(prog_row, text="Progress").pack(side=tk.LEFT)
        ttk.Label(prog_row, textvariable=self.progress_pct_var).pack(side=tk.RIGHT)
        self.progress_bar = ttk.Progressbar(
            parent,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100.0,
            variable=self.progress_var,
        )
        self.progress_bar.pack(fill="x", pady=(0, 4))

        self._wrap_label(ttk.Label(parent, textvariable=self.status_var, wraplength=330)).pack(fill="x", pady=(6, 0))
        self.cursor_info_var = tk.StringVar(
            value="Cursor (XY/XZ/YZ): move mouse over 2D plots after a run."
        )
        self._wrap_label(ttk.Label(parent, textvariable=self.cursor_info_var, wraplength=330)).pack(
            fill="x", pady=(2, 0)
        )

        for tracked in (self.nx_var, self.ny_var, self.size_x_mm_var, self.size_y_mm_var):
            tracked.trace_add("write", self._on_sampling_param_change)
            tracked.trace_add("write", self._on_mask_preview_param_change)
        for tracked in (self.mask_offset_x_mm_var, self.mask_offset_y_mm_var, self.pupil_d_mm_var):
            tracked.trace_add("write", self._on_mask_preview_param_change)
        self._update_mask_orientation_label()
        self._refresh_mask_input_preview()
        self._update_sampling_info()

    _UNITS = ("nm", "um", "mm", "m", "deg")

    @classmethod
    def _split_unit(cls, label):
        """'Mask size X (mm)' -> ('Mask size X', 'mm'); labels without a unit are unchanged."""
        if label.endswith(")") and " (" in label:
            text, unit = label[:-1].rsplit(" (", 1)
            if unit in cls._UNITS:
                return text, unit
        return label, ""

    @classmethod
    def _entry(cls, parent, label, var, unit=None):
        text, parsed_unit = cls._split_unit(label)
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 3))
        row.columnconfigure(0, weight=1)
        ttk.Label(row, text=text).grid(row=0, column=0, sticky="w", padx=(0, 6))
        entry = ttk.Entry(row, textvariable=var, width=10, justify="right")
        entry.grid(row=0, column=1, sticky="e")
        ttk.Label(row, text=parsed_unit if unit is None else unit, width=3).grid(
            row=0, column=2, sticky="w", padx=(4, 0)
        )
        return entry

    @staticmethod
    def _wrap_label(label):
        """Re-wrap a full-width label whenever the panel width changes."""
        label.bind("<Configure>", lambda event: event.widget.configure(wraplength=max(120, event.width - 4)))
        return label

    # GUI variable holding each PARAM_SPECS entry, keyed by its display-unit ui_key.
    _PARAM_VARIABLES = {
        "wavelength_nm": "wavelength_nm_var",
        "NA": "na_var",
        "theta_samples": "theta_samples_var",
        "phi_samples": "phi_samples_var",
        "integration_mode": "integration_mode_var",
        "aberration_model": "aberration_model_var",
        "torch_device": "torch_device_var",
        "nx": "nx_var",
        "ny": "ny_var",
        "size_x_mm": "size_x_mm_var",
        "size_y_mm": "size_y_mm_var",
        "pupil_d_mm": "pupil_d_mm_var",
        "fresnel_z_m": "fresnel_z_m_var",
        "range_x_um": "range_x_um_var",
        "range_y_um": "range_y_um_var",
        "range_z_um": "range_z_um_var",
        "points_2d": "points_2d_var",
        "points_3d": "points_3d_var",
        "z_chunk": "z_chunk_var",
        "n_i": "exp_immersion_ri_var",
        "n_g": "exp_coverslip_ri_var",
        "t_g_um": "exp_coverslip_thickness_um_var",
        "n_s": "exp_resin_ri_var",
        "t_s_um": "exp_resin_thickness_um_var",
        "n_id": "design_immersion_ri_var",
        "t_id_um": "design_immersion_gap_um_var",
        "n_gd": "design_coverslip_ri_var",
        "t_gd_um": "design_coverslip_thickness_um_var",
        "n_sd": "design_resin_ri_var",
        "use_gaussian": "use_gaussian_var",
        "gauss_fwhm_x_mm": "gauss_fwhm_x_mm_var",
        "gauss_fwhm_y_mm": "gauss_fwhm_y_mm_var",
        "mask_offset_x_mm": "mask_offset_x_mm_var",
        "mask_offset_y_mm": "mask_offset_y_mm_var",
    }
    _AXIS_RANGE_KEYS = ("range_x_um", "range_y_um", "range_z_um")

    def _param_var(self, spec):
        return getattr(self, self._PARAM_VARIABLES[spec.ui_key])

    def _dip_in_value_vars(self):
        return {
            "n_i": self.exp_immersion_ri_var,
            "n_g": self.exp_coverslip_ri_var,
            "t_g_um": self.exp_coverslip_thickness_um_var,
            "t_s_um": self.exp_resin_thickness_um_var,
            "aberration_model": self.aberration_model_var,
        }

    _DIP_IN_FIELD_LABELS = {
        "n_i": "Immersion medium RI",
        "n_g": "Coverslip RI",
        "t_g_um": "Coverslip thickness",
        "t_s_um": "Resin thickness / PSF position",
    }
    _MODEL_DESCRIPTIONS = {
        "legacy": "Nasse-Woehl: coverslip corrected to the design; no immersion-layer path.",
        "gibson_lanni": "Gibson-Lanni layers: adds the immersion-gap path (Torok-Varga Eq. 28).",
    }

    def _dip_in_snapshot(self):
        snapshot = {}
        for key, var in self._dip_in_value_vars().items():
            if key == "aberration_model":
                snapshot[key] = str(var.get()).strip().lower()
                continue
            try:
                value = float(var.get())
            except (tk.TclError, ValueError):
                value = float("nan")
            if not np.isfinite(value):
                label = self._DIP_IN_FIELD_LABELS[key]
                raise ValueError(f"Finish editing '{label}' before enabling dip-in.")
            snapshot[key] = value
        return snapshot

    def _on_dip_in_toggle(self):
        if self.dip_in_enabled_var.get():
            try:
                snapshot = self._dip_in_snapshot()
            except ValueError as exc:
                # Without a snapshot the layered settings could never be
                # restored, so stay in standard mode until the field is valid.
                self.dip_in_enabled_var.set(False)
                self._update_dip_in_controls()
                messagebox.showwarning("Dip-in lens", str(exc))
                return
            self._dip_in_previous_values = snapshot
        else:
            previous = getattr(self, "_dip_in_previous_values", None)
            if previous:
                for key, var in self._dip_in_value_vars().items():
                    var.set(previous[key])
            self._dip_in_previous_values = None
        self._sync_dip_in_media()
        self._update_dip_in_controls()

    def _update_model_description(self, *_args):
        model = str(self.aberration_model_var.get()).strip().lower()
        text = self._MODEL_DESCRIPTIONS.get(model, model)
        if bool(self.dip_in_enabled_var.get()):
            text = f"Dip-in (resin as immersion). {text}"
        self.aberration_model_description_var.set(text)

    def _sync_dip_in_media(self, *_args):
        dipping = bool(self.dip_in_enabled_var.get())
        if dipping:
            # Resin replaces the design immersion over the whole working
            # distance, which only the layered Gibson-Lanni phase describes.
            # Outside dip-in the user's (or loaded config's) model is kept.
            self.aberration_model_var.set("gibson_lanni")
        self._update_model_description()
        if not dipping:
            return
        try:
            resin_ri = float(self.exp_resin_ri_var.get())
        except (ValueError, tk.TclError):
            return  # Allow an incomplete entry while the user types.
        if not np.isfinite(resin_ri) or resin_ri <= 0:
            return
        self.exp_immersion_ri_var.set(resin_ri)
        self.exp_coverslip_ri_var.set(resin_ri)
        self.exp_coverslip_thickness_um_var.set(0.0)
        self.exp_resin_thickness_um_var.set(0.0)

    def _update_dip_in_controls(self):
        busy = bool(getattr(self, "is_running", False) or getattr(self, "opt_is_running", False))
        if hasattr(self, "dip_in_check"):
            self.dip_in_check.configure(state="disabled" if busy else "normal")
        dipping = bool(self.dip_in_enabled_var.get())
        state = "disabled" if dipping else "normal"
        for entry in getattr(self, "_dip_in_linked_entries", ()):
            entry.configure(state=state)
        if hasattr(self, "aberration_model_combo"):
            self.aberration_model_combo.configure(state="disabled" if dipping else "readonly")

    def _restore_dip_in_control_state(self, values, *, core_units=False):
        flag = values.get("dip_in_enabled")
        if flag is None:
            glass_key, sample_key = ("t_g", "t_s") if core_units else ("t_g_um", "t_s_um")
            try:
                flag = (
                    str(values.get("aberration_model", "")).strip().lower() == "gibson_lanni"
                    and float(values["n_i"]) == float(values["n_g"]) == float(values["n_s"])
                    and float(values[glass_key]) == 0.0
                    and float(values[sample_key]) == 0.0
                )
            except (KeyError, TypeError, ValueError):
                flag = False
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError("dip_in_enabled must be a boolean.")
        previous = values.get("dip_in_previous_values")
        if flag and previous is None:
            # Older/manual dip-in configs have no pre-toggle snapshot. Offer
            # layered settings based on their current objective's design.
            previous = {
                "n_i": float(self.design_immersion_ri_var.get()),
                "n_g": float(self.design_coverslip_ri_var.get()),
                "t_g_um": float(self.design_coverslip_thickness_um_var.get()),
                "t_s_um": 1.0, "aberration_model": "legacy",
            }
        if previous is not None:
            if not isinstance(previous, dict):
                raise ValueError("dip_in_previous_values must be an object.")
            previous = dict(previous)
            for key in ("n_i", "n_g", "t_g_um", "t_s_um"):
                value = float(previous[key])
                if not np.isfinite(value) or value < 0 or (key.startswith("n_") and value == 0):
                    raise ValueError("Invalid previous dip-in medium values.")
                previous[key] = value
            previous["aberration_model"] = str(previous.get("aberration_model", "")).strip().lower()
            if previous["aberration_model"] not in ABERRATION_MODELS:
                raise ValueError("Invalid previous dip-in aberration model.")
        self._dip_in_previous_values = previous if flag else None
        self.dip_in_enabled_var.set(bool(flag))
        self._sync_dip_in_media()
        self._update_dip_in_controls()

    def _on_sampling_param_change(self, *_args):
        self._update_sampling_info()

    def _on_integration_mode_change(self, *_args):
        mode = str(self.integration_mode_var.get()).strip().lower()
        state = "normal" if mode == "polar_interp" else "disabled"
        for entry_name in ("theta_samples_entry", "phi_samples_entry"):
            entry = getattr(self, entry_name, None)
            if entry is not None:
                entry.configure(state=state)

    # The preview only needs screen resolution; the simulation grid can be far larger.
    _PREVIEW_MAX_PIXELS = 512
    _PREVIEW_DELAY_MS = 150

    def _on_mask_preview_param_change(self, *_args):
        # Coalesce bursts of writes (typing, config loads) into one redraw.
        self._cancel_mask_preview_job()
        self._mask_preview_job = self.root.after(self._PREVIEW_DELAY_MS, self._refresh_mask_input_preview)

    def _cancel_mask_preview_job(self):
        job = getattr(self, "_mask_preview_job", None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
        self._mask_preview_job = None

    @classmethod
    def _preview_shape(cls, nx, ny):
        scale = min(1.0, cls._PREVIEW_MAX_PIXELS / max(nx, ny))
        return max(1, round(nx * scale)), max(1, round(ny * scale))

    def _mask_rotation_quadrants_mod(self):
        return int(self.mask_rotation_quadrants) % 4

    def _update_mask_orientation_label(self):
        q = self._mask_rotation_quadrants_mod()
        deg = (90 * q) % 360
        if deg == 0:
            text = "Mask orientation: 0 deg"
        else:
            text = f"Mask orientation: {deg} deg CCW"
        if hasattr(self, "mask_orientation_var"):
            self.mask_orientation_var.set(text)

    def _rotate_mask_cw_90(self):
        self.mask_rotation_quadrants = self._mask_rotation_quadrants_mod() - 1
        self._update_mask_orientation_label()
        self._refresh_mask_input_preview()

    def _rotate_mask_ccw_90(self):
        self.mask_rotation_quadrants = self._mask_rotation_quadrants_mod() + 1
        self._update_mask_orientation_label()
        self._refresh_mask_input_preview()

    def _mask_for_current_preview(self):
        nx = max(1, int(self.nx_var.get()))
        ny = max(1, int(self.ny_var.get()))
        size_x_m = float(self.size_x_mm_var.get()) * 1e-3
        size_y_m = float(self.size_y_mm_var.get()) * 1e-3
        if size_x_m <= 0 or size_y_m <= 0:
            raise ValueError("Mask size X/Y must be positive.")
        offset_x = float(self.mask_offset_x_mm_var.get()) * 1e-3
        offset_y = float(self.mask_offset_y_mm_var.get()) * 1e-3
        mask = self._get_input_mask(
            self._preview_shape(nx, ny),
            size_x=size_x_m,
            size_y=size_y_m,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        extent = [
            -0.5 * size_x_m * 1e3,
            0.5 * size_x_m * 1e3,
            -0.5 * size_y_m * 1e3,
            0.5 * size_y_m * 1e3,
        ]
        return np.asarray(mask, dtype=float), extent, (nx, ny)

    def _refresh_mask_input_preview(self):
        self._cancel_mask_preview_job()
        if not hasattr(self, "fig_mask_input") or not hasattr(self, "canvas_mask_input"):
            return

        self.fig_mask_input.clear()
        ax = self.fig_mask_input.add_subplot(111)
        try:
            mask, extent, (nx, ny) = self._mask_for_current_preview()
            im = ax.imshow(
                mask.T,
                origin="lower",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                extent=extent,
                aspect="equal",
                interpolation="nearest",
            )
            style_image_axis(
                ax,
                title=f"Effective mask ({nx} x {ny})",
                xlabel="x (mm)",
                ylabel="y (mm)",
            )
            pupil_d_mm = float(self.pupil_d_mm_var.get())
            if np.isfinite(pupil_d_mm) and pupil_d_mm > 0.0:
                pupil_circle = Circle(
                    (0.0, 0.0),
                    radius=0.5 * pupil_d_mm,
                    fill=False,
                    edgecolor="#1B3B6F",
                    linewidth=0.9,
                    linestyle=(0, (3, 2)),
                )
                ax.add_patch(pupil_circle)
            add_compact_colorbar(self.fig_mask_input, im, ax, "Mask value")
        except Exception as exc:
            ax.text(0.5, 0.5, f"Mask preview unavailable\n{exc}", ha="center", va="center")
            ax.axis("off")
        self.canvas_mask_input.draw_idle()

    def _update_sampling_info(self):
        try:
            nx = int(self.nx_var.get())
            ny = int(self.ny_var.get())
            size_x_m = float(self.size_x_mm_var.get()) * 1e-3
            size_y_m = float(self.size_y_mm_var.get()) * 1e-3
            if nx <= 0 or ny <= 0 or size_x_m <= 0 or size_y_m <= 0:
                raise ValueError
            dx_um = (size_x_m / nx) * 1e6
            dy_um = (size_y_m / ny) * 1e6
            self.sampling_info_var.set(
                f"Grid: {nx} x {ny} pixels, dx={dx_um:.3f} um, dy={dy_um:.3f} um"
            )
        except Exception:
            self.sampling_info_var.set("Grid: invalid parameters")

    def _on_controls_frame_configure(self, _event):
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _on_controls_canvas_configure(self, event):
        self.controls_canvas.itemconfigure(self._controls_window, width=event.width)

    def _bind_controls_mousewheel(self, _event):
        self.controls_canvas.bind_all("<MouseWheel>", self._on_controls_mousewheel)
        self.controls_canvas.bind_all("<Button-4>", self._on_controls_mousewheel_linux)
        self.controls_canvas.bind_all("<Button-5>", self._on_controls_mousewheel_linux)

    def _unbind_controls_mousewheel(self, _event):
        self.controls_canvas.unbind_all("<MouseWheel>")
        self.controls_canvas.unbind_all("<Button-4>")
        self.controls_canvas.unbind_all("<Button-5>")

    def _on_controls_mousewheel(self, event):
        step = int(-event.delta / 120) if event.delta != 0 else 0
        if step != 0:
            self.controls_canvas.yview_scroll(step, "units")

    def _on_controls_mousewheel_linux(self, event):
        if getattr(event, "num", None) == 4:
            self.controls_canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.controls_canvas.yview_scroll(1, "units")

    def _build_plots(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_2d = ttk.Frame(self.notebook)
        self.tab_mask = ttk.Frame(self.notebook)
        self.tab_opt = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_2d, text="2D Results")
        self.notebook.add(self.tab_mask, text="Mask Builder")
        self.notebook.add(self.tab_opt, text="Mask Optimization")

        self.fig_2d = make_figure(figsize=(10.6, 7.5), dpi=150)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=self.tab_2d)
        self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._cursor_cid = self.canvas_2d.mpl_connect("motion_notify_event", self._on_2d_cursor_move)
        self._cursor_press_cid = self.canvas_2d.mpl_connect(
            "button_press_event", self._on_2d_canvas_button_press
        )
        self._cursor_release_cid = self.canvas_2d.mpl_connect(
            "button_release_event", self._on_2d_canvas_button_release
        )

        self._build_mask_builder_tab(self.tab_mask)
        self._build_optimization_tab(self.tab_opt)

    def _build_mask_builder_tab(self, parent):
        container = ttk.Frame(parent, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        pane = ttk.PanedWindow(container, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(pane, padding=6)
        preview = ttk.Frame(pane, padding=6)
        pane.add(controls, weight=0)
        pane.add(preview, weight=1)

        geometry_group = ttk.LabelFrame(controls, text="Mask Geometry", padding=8)
        geometry_group.pack(fill="x", pady=(0, 6))
        self.builder_nx_var = tk.IntVar(value=max(1, int(self.nx_var.get())))
        self.builder_ny_var = tk.IntVar(value=max(1, int(self.ny_var.get())))
        self.builder_size_x_mm_var = tk.DoubleVar(value=float(self.size_x_mm_var.get()))
        self.builder_size_y_mm_var = tk.DoubleVar(value=float(self.size_y_mm_var.get()))
        self._entry(geometry_group, "Nx", self.builder_nx_var)
        self._entry(geometry_group, "Ny", self.builder_ny_var)
        self._entry(geometry_group, "Size X (mm)", self.builder_size_x_mm_var)
        self._entry(geometry_group, "Size Y (mm)", self.builder_size_y_mm_var)

        shape_group = ttk.LabelFrame(controls, text="Shape Tool (mm)", padding=8)
        shape_group.pack(fill="x", pady=(0, 6))
        self.builder_shape_var = tk.StringVar(value="circle")
        self.builder_mode_var = tk.StringVar(value="add")
        self.builder_center_x_mm_var = tk.DoubleVar(value=0.0)
        self.builder_center_y_mm_var = tk.DoubleVar(value=0.0)
        self.builder_circle_radius_mm_var = tk.DoubleVar(value=0.8)
        self.builder_ring_inner_mm_var = tk.DoubleVar(value=0.4)
        self.builder_ring_outer_mm_var = tk.DoubleVar(value=0.8)
        self.builder_rect_x_size_mm_var = tk.DoubleVar(value=1.2)
        self.builder_rect_y_size_mm_var = tk.DoubleVar(value=0.8)

        shape_row = ttk.Frame(shape_group)
        shape_row.pack(fill="x")
        ttk.Label(shape_row, text="Shape").pack(side=tk.LEFT)
        ttk.Combobox(
            shape_row,
            textvariable=self.builder_shape_var,
            values=("circle", "ring", "rectangle"),
            state="readonly",
            width=12,
        ).pack(side=tk.RIGHT, fill="x", expand=True)

        mode_row = ttk.Frame(shape_group)
        mode_row.pack(fill="x", pady=(4, 2))
        ttk.Label(mode_row, text="Mode").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Add", variable=self.builder_mode_var, value="add").pack(
            side=tk.LEFT, padx=(8, 4)
        )
        ttk.Radiobutton(mode_row, text="Erase", variable=self.builder_mode_var, value="erase").pack(
            side=tk.LEFT, padx=4
        )

        self._entry(shape_group, "Center x (mm)", self.builder_center_x_mm_var)
        self._entry(shape_group, "Center y (mm)", self.builder_center_y_mm_var)

        self.builder_shape_params_parent = ttk.Frame(shape_group)
        self.builder_shape_params_parent.pack(fill="x", pady=(2, 0))

        self.builder_circle_params = ttk.Frame(self.builder_shape_params_parent)
        self._entry(self.builder_circle_params, "Radius (mm)", self.builder_circle_radius_mm_var)

        self.builder_ring_params = ttk.Frame(self.builder_shape_params_parent)
        self._entry(self.builder_ring_params, "Inner radius (mm)", self.builder_ring_inner_mm_var)
        self._entry(self.builder_ring_params, "Outer radius (mm)", self.builder_ring_outer_mm_var)

        self.builder_rect_params = ttk.Frame(self.builder_shape_params_parent)
        self._entry(self.builder_rect_params, "X size (mm)", self.builder_rect_x_size_mm_var)
        self._entry(self.builder_rect_params, "Y size (mm)", self.builder_rect_y_size_mm_var)

        self.builder_shape_var.trace_add("write", self._on_builder_shape_change)
        self._update_builder_shape_controls()

        actions_group = ttk.LabelFrame(controls, text="Actions", padding=8)
        actions_group.pack(fill="x", pady=(0, 6))

        init_row = ttk.Frame(actions_group)
        init_row.pack(fill="x")
        self._add_mask_builder_button(
            init_row, "Initialize Blank Mask", self._initialize_mask_builder_blank, padx=(0, 3)
        )
        self._add_mask_builder_button(
            init_row,
            "Initialize From Current",
            self._initialize_mask_builder_from_current,
            padx=(3, 0),
        )

        edit_row = ttk.Frame(actions_group)
        edit_row.pack(fill="x", pady=(6, 0))
        self._add_mask_builder_button(edit_row, "Apply Shape", self._apply_builder_shape, padx=(0, 3))
        self._add_mask_builder_button(edit_row, "Undo", self._undo_builder_shape, padx=(3, 3))
        self._add_mask_builder_button(edit_row, "Clear", self._clear_builder_mask, padx=(3, 0))

        io_row = ttk.Frame(actions_group)
        io_row.pack(fill="x", pady=(6, 0))
        self._add_mask_builder_button(io_row, "Save Mask", self._save_builder_mask_dialog, padx=(0, 3))
        self._add_mask_builder_button(
            io_row, "Load Mask Into Builder", self._load_mask_into_builder_dialog, padx=(3, 0)
        )

        apply_row = ttk.Frame(actions_group)
        apply_row.pack(fill="x", pady=(6, 0))
        self._add_mask_builder_button(
            apply_row,
            "Use Builder Mask As Simulation Mask",
            self._apply_builder_mask_to_simulation,
            padx=(0, 0),
        )

        self.mask_builder_status_var = tk.StringVar(value="Mask Builder ready.")
        ttk.Label(controls, textvariable=self.mask_builder_status_var, wraplength=360).pack(
            fill="x", pady=(4, 0)
        )

        self.fig_mask = make_figure(figsize=(6.2, 5.3), dpi=150)
        self.canvas_mask = FigureCanvasTkAgg(self.fig_mask, master=preview)
        self.canvas_mask.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._refresh_mask_builder_plot()

    def _get_optimizer_category_map(self):
        fallback = {
            "Local (gradient-based)": ("L-BFGS-B", "SLSQP"),
            "Local (derivative-free)": ("Powell", "Nelder-Mead"),
            "Global": ("Differential Evolution", "SHGO"),
        }
        if HAS_SCIPY_OPTIMIZER and PostFresnelMaskOptimizer is not None:
            try:
                raw = PostFresnelMaskOptimizer.get_optimizer_category_map()
                if isinstance(raw, dict):
                    cleaned = {}
                    for category, names in raw.items():
                        cat = str(category).strip()
                        vals = tuple(
                            str(name).strip()
                            for name in tuple(names)
                            if str(name).strip() != ""
                        )
                        if cat != "" and len(vals) > 0:
                            cleaned[cat] = vals
                    if len(cleaned) > 0:
                        return cleaned
            except Exception:
                pass
        return dict(fallback)

    @staticmethod
    def _all_optimizer_names(category_map):
        names = []
        if isinstance(category_map, dict):
            for vals in category_map.values():
                for name in tuple(vals):
                    n = str(name).strip()
                    if n != "":
                        names.append(n)
        return tuple(names)

    @staticmethod
    def _optimizer_category_for_name(optimizer_name, category_map):
        name = str(optimizer_name).strip()
        if name == "" or (not isinstance(category_map, dict)):
            return None
        for category, names in category_map.items():
            if name in tuple(names):
                return str(category)
        return None

    def _sync_optimizer_combo_to_category(self, preferred_optimizer=None):
        if not hasattr(self, "opt_optimizer_combo") or not hasattr(self, "opt_optimizer_var"):
            return
        category_map = self._get_optimizer_category_map()
        self._opt_optimizer_categories = dict(category_map)
        if hasattr(self, "opt_optimizer_category_combo"):
            self.opt_optimizer_category_combo.configure(values=tuple(category_map.keys()))
        category = ""
        if hasattr(self, "opt_optimizer_category_var"):
            category = str(self.opt_optimizer_category_var.get()).strip()
        if category not in category_map:
            if len(category_map) > 0:
                category = next(iter(category_map.keys()))
                if hasattr(self, "opt_optimizer_category_var"):
                    self.opt_optimizer_category_var.set(category)
            else:
                category = ""
        opts = tuple(category_map.get(category, ()))
        if len(opts) == 0:
            opts = self._all_optimizer_names(category_map)
        self.opt_optimizer_combo.configure(values=opts)
        desired = str(preferred_optimizer).strip() if preferred_optimizer is not None else ""
        if desired not in opts:
            desired = str(self.opt_optimizer_var.get()).strip()
        if desired not in opts:
            desired = opts[0] if len(opts) > 0 else ""
        if desired != "":
            self.opt_optimizer_var.set(desired)

    def _on_opt_optimizer_category_change(self, *_args):
        self._sync_optimizer_combo_to_category(preferred_optimizer=self.opt_optimizer_var.get())

    def _build_optimization_tab(self, parent):
        container = ttk.Frame(parent, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        pane = ttk.PanedWindow(container, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        controls_host = ttk.Frame(pane, padding=0)
        preview = ttk.Frame(pane, padding=6)
        pane.add(controls_host, weight=0)
        pane.add(preview, weight=1)

        # Scrollable optimization controls panel so all solver options remain reachable.
        self.opt_controls_canvas = tk.Canvas(
            controls_host, highlightthickness=0, bg="#f3f5f8", bd=0
        )
        opt_controls_scrollbar = ttk.Scrollbar(
            controls_host, orient=tk.VERTICAL, command=self.opt_controls_canvas.yview
        )
        self.opt_controls_canvas.configure(yscrollcommand=opt_controls_scrollbar.set)

        opt_controls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.opt_controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls = ttk.Frame(self.opt_controls_canvas, padding=6)
        self._opt_controls_window = self.opt_controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )

        controls.bind("<Configure>", self._on_opt_controls_frame_configure)
        self.opt_controls_canvas.bind("<Configure>", self._on_opt_controls_canvas_configure)
        self.opt_controls_canvas.bind("<Enter>", self._bind_opt_controls_mousewheel)
        self.opt_controls_canvas.bind("<Leave>", self._unbind_opt_controls_mousewheel)

        self._opt_optimizer_categories = self._get_optimizer_category_map()
        _opt_names = self._all_optimizer_names(self._opt_optimizer_categories)
        _default_opt = "L-BFGS-B" if "L-BFGS-B" in _opt_names else (_opt_names[0] if len(_opt_names) > 0 else "")
        _default_category = self._optimizer_category_for_name(
            _default_opt, self._opt_optimizer_categories
        )
        if _default_category is None:
            _default_category = (
                next(iter(self._opt_optimizer_categories.keys()))
                if len(self._opt_optimizer_categories) > 0
                else "Local (gradient-based)"
            )
        self.opt_optimizer_category_var = tk.StringVar(value=str(_default_category))
        self.opt_optimizer_var = tk.StringVar(value=str(_default_opt))
        self.opt_mode_var = tk.StringVar(value="amplitude_phase")
        self.opt_width_level_var = tk.DoubleVar(value=0.50)
        self.opt_weight_x_var = tk.DoubleVar(value=1.0)
        self.opt_weight_y_var = tk.DoubleVar(value=1.0)
        self.opt_weight_z_var = tk.DoubleVar(value=1.0)
        self.opt_weight_abs_xy_var = tk.DoubleVar(value=0.0)
        self.opt_n_starts_var = tk.IntVar(value=2)
        self.opt_maxiter_var = tk.IntVar(value=30)
        self.opt_ftol_var = tk.DoubleVar(value=1e-6)
        self.opt_gtol_var = tk.DoubleVar(value=1e-5)
        self.opt_seed_var = tk.StringVar(value="")
        self.opt_axial_symmetry_var = tk.BooleanVar(value=True)
        self.opt_binary_amplitude_var = tk.BooleanVar(value=True)
        self.opt_binary_structure_var = tk.StringVar(value="pixel")
        self.opt_progress_var = tk.DoubleVar(value=0.0)
        self.opt_progress_pct_var = tk.StringVar(value="0.0%")
        self.opt_status_var = tk.StringVar(value="Mask optimization ready.")
        self.opt_summary_var = tk.StringVar(value="No optimization run yet.")
        if not HAS_SCIPY_OPTIMIZER:
            self.opt_status_var.set(
                "Mask optimization unavailable: scipy optimizer module could not load."
            )

        mode_group = ttk.LabelFrame(controls, text="Optimization Target", padding=8)
        mode_group.pack(fill="x", pady=(0, 6))
        ttk.Label(mode_group, text="Variables to optimize").pack(anchor="w")
        self.opt_mode_combo = ttk.Combobox(
            mode_group,
            textvariable=self.opt_mode_var,
            values=("amplitude", "phase", "amplitude_phase"),
            state="readonly",
        )
        self.opt_mode_combo.pack(fill="x", pady=(0, 4))
        self.opt_axial_symmetry_chk = ttk.Checkbutton(
            mode_group,
            text="Axial symmetry only (radial optimizer)",
            variable=self.opt_axial_symmetry_var,
        )
        self.opt_axial_symmetry_chk.pack(anchor="w", pady=(2, 2))
        self.opt_axial_symmetry_chk.configure(state="disabled")
        self.opt_binary_amplitude_chk = ttk.Checkbutton(
            mode_group,
            text="Amplitude-only binary mask (pass/block)",
            variable=self.opt_binary_amplitude_var,
        )
        self.opt_binary_amplitude_chk.pack(anchor="w", pady=(0, 2))
        ttk.Label(mode_group, text="Pass/block structure").pack(anchor="w")
        self.opt_binary_structure_combo = ttk.Combobox(
            mode_group,
            textvariable=self.opt_binary_structure_var,
            values=("pixel", "simple_ring_block", "simple_ellipse_block"),
            state="readonly",
        )
        self.opt_binary_structure_combo.pack(fill="x", pady=(0, 2))
        ttk.Label(
            mode_group,
            text=(
                "simple_ring_block: optimize inner/outer blocked radii only.\n"
                "simple_ellipse_block: optimize an elliptical ring with inner/outer x/y diameters."
            ),
            wraplength=340,
        ).pack(fill="x", pady=(0, 2))

        ttk.Label(mode_group, text="Algorithm category").pack(anchor="w")
        self.opt_optimizer_category_combo = ttk.Combobox(
            mode_group,
            textvariable=self.opt_optimizer_category_var,
            values=tuple(self._opt_optimizer_categories.keys()),
            state="readonly",
        )
        self.opt_optimizer_category_combo.pack(fill="x", pady=(0, 2))
        ttk.Label(mode_group, text="Optimizer").pack(anchor="w")
        self.opt_optimizer_combo = ttk.Combobox(
            mode_group,
            textvariable=self.opt_optimizer_var,
            values=self._opt_optimizer_categories.get(
                str(self.opt_optimizer_category_var.get()).strip(),
                self._all_optimizer_names(self._opt_optimizer_categories),
            ),
            state="readonly",
        )
        self.opt_optimizer_combo.pack(fill="x", pady=(0, 4))

        objective_group = ttk.LabelFrame(controls, text="Objective", padding=8)
        objective_group.pack(fill="x", pady=(0, 6))
        self._entry(objective_group, "Width level (0.01..0.99)", self.opt_width_level_var)
        self._entry(objective_group, "Weight X", self.opt_weight_x_var)
        self._entry(objective_group, "Weight Y", self.opt_weight_y_var)
        self._entry(objective_group, "Weight Z", self.opt_weight_z_var)
        self._entry(objective_group, "abs(x-y)", self.opt_weight_abs_xy_var)

        solver_group = ttk.LabelFrame(controls, text="Solver Budget", padding=8)
        solver_group.pack(fill="x", pady=(0, 6))
        self._entry(solver_group, "Multi-start count", self.opt_n_starts_var)
        self._entry(solver_group, "Max iterations", self.opt_maxiter_var)
        self._entry(solver_group, "ftol", self.opt_ftol_var)
        self._entry(solver_group, "gtol", self.opt_gtol_var)
        self._entry(solver_group, "Seed (blank = random)", self.opt_seed_var)

        actions_group = ttk.LabelFrame(controls, text="Actions", padding=8)
        actions_group.pack(fill="x", pady=(0, 6))
        run_row = ttk.Frame(actions_group)
        run_row.pack(fill="x")
        self.opt_run_btn = ttk.Button(
            run_row, text="Run Optimization", command=self._start_optimization_worker
        )
        self.opt_run_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 3))
        self.opt_stop_btn = ttk.Button(
            run_row, text="Stop", command=self._stop_optimization
        )
        self.opt_stop_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(3, 0))
        self.opt_stop_btn.configure(state="disabled")

        io_row = ttk.Frame(actions_group)
        io_row.pack(fill="x", pady=(6, 0))
        self.opt_save_btn = ttk.Button(
            io_row,
            text="Save Optimized Mask (NPZ)",
            command=self._save_optimized_mask_dialog,
        )
        self.opt_save_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 3))
        self.opt_apply_btn = ttk.Button(
            io_row,
            text="Use Optimized Amplitude As Input Mask",
            command=self._apply_optimized_mask_to_simulation,
        )
        self.opt_apply_btn.pack(side=tk.LEFT, fill="x", expand=True, padx=(3, 0))
        self.opt_save_btn.configure(state="disabled")
        self.opt_apply_btn.configure(state="disabled")

        if not HAS_SCIPY_OPTIMIZER:
            self.opt_run_btn.configure(state="disabled")
            self.opt_stop_btn.configure(state="disabled")

        prog_row = ttk.Frame(controls)
        prog_row.pack(fill="x", pady=(6, 2))
        ttk.Label(prog_row, text="Optimization progress").pack(side=tk.LEFT)
        ttk.Label(prog_row, textvariable=self.opt_progress_pct_var).pack(side=tk.RIGHT)
        self.opt_progress_bar = ttk.Progressbar(
            controls,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100.0,
            variable=self.opt_progress_var,
        )
        self.opt_progress_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(controls, textvariable=self.opt_status_var, wraplength=360).pack(
            fill="x", pady=(0, 2)
        )
        ttk.Label(controls, textvariable=self.opt_summary_var, wraplength=360).pack(
            fill="x"
        )

        self.fig_opt = make_figure(figsize=(6.8, 5.6), dpi=150)
        self.canvas_opt = FigureCanvasTkAgg(self.fig_opt, master=preview)
        self.canvas_opt.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.opt_optimizer_category_var.trace_add("write", self._on_opt_optimizer_category_change)
        self.opt_mode_var.trace_add("write", self._on_opt_mode_change)
        self.opt_binary_amplitude_var.trace_add("write", self._on_opt_binary_options_change)
        self.opt_binary_structure_var.trace_add("write", self._on_opt_binary_options_change)
        self._sync_optimizer_combo_to_category(preferred_optimizer=self.opt_optimizer_var.get())
        self._on_opt_mode_change()
        self._on_opt_binary_options_change()
        self._refresh_optimization_plot()

    def _on_opt_mode_change(self, *_args):
        if not hasattr(self, "opt_mode_var"):
            return
        mode = str(self.opt_mode_var.get()).strip().lower()
        state = "normal" if mode == "amplitude" else "disabled"
        if hasattr(self, "opt_binary_amplitude_chk"):
            self.opt_binary_amplitude_chk.configure(state=state)
        self._on_opt_binary_options_change()

    def _on_opt_binary_options_change(self, *_args):
        if not hasattr(self, "opt_mode_var"):
            return
        mode = str(self.opt_mode_var.get()).strip().lower()
        binary_enabled = bool(self.opt_binary_amplitude_var.get()) if hasattr(self, "opt_binary_amplitude_var") else False
        allow_structure = (mode == "amplitude") and bool(binary_enabled)
        structure = (
            str(self.opt_binary_structure_var.get()).strip().lower()
            if hasattr(self, "opt_binary_structure_var")
            else "pixel"
        )
        if hasattr(self, "opt_binary_structure_combo"):
            self.opt_binary_structure_combo.configure(state="readonly" if allow_structure else "disabled")
        if not allow_structure and hasattr(self, "opt_binary_structure_var"):
            self.opt_binary_structure_var.set("pixel")
            structure = "pixel"
        allow_axial_toggle = True
        if structure == "simple_ring_block":
            allow_axial_toggle = False
            if hasattr(self, "opt_axial_symmetry_var"):
                self.opt_axial_symmetry_var.set(True)
        elif structure == "simple_ellipse_block":
            allow_axial_toggle = False
            if hasattr(self, "opt_axial_symmetry_var"):
                self.opt_axial_symmetry_var.set(False)
        if hasattr(self, "opt_axial_symmetry_chk"):
            self.opt_axial_symmetry_chk.configure(state=("normal" if allow_axial_toggle else "disabled"))

    def _on_opt_controls_frame_configure(self, _event):
        if hasattr(self, "opt_controls_canvas"):
            self.opt_controls_canvas.configure(scrollregion=self.opt_controls_canvas.bbox("all"))

    def _on_opt_controls_canvas_configure(self, event):
        if hasattr(self, "opt_controls_canvas") and hasattr(self, "_opt_controls_window"):
            self.opt_controls_canvas.itemconfigure(self._opt_controls_window, width=event.width)

    def _bind_opt_controls_mousewheel(self, _event):
        if not hasattr(self, "opt_controls_canvas"):
            return
        self.opt_controls_canvas.bind_all("<MouseWheel>", self._on_opt_controls_mousewheel)
        self.opt_controls_canvas.bind_all("<Button-4>", self._on_opt_controls_mousewheel_linux)
        self.opt_controls_canvas.bind_all("<Button-5>", self._on_opt_controls_mousewheel_linux)

    def _unbind_opt_controls_mousewheel(self, _event):
        if not hasattr(self, "opt_controls_canvas"):
            return
        self.opt_controls_canvas.unbind_all("<MouseWheel>")
        self.opt_controls_canvas.unbind_all("<Button-4>")
        self.opt_controls_canvas.unbind_all("<Button-5>")

    def _on_opt_controls_mousewheel(self, event):
        if not hasattr(self, "opt_controls_canvas"):
            return
        step = int(-event.delta / 120) if event.delta != 0 else 0
        if step != 0:
            self.opt_controls_canvas.yview_scroll(step, "units")

    def _on_opt_controls_mousewheel_linux(self, event):
        if not hasattr(self, "opt_controls_canvas"):
            return
        if getattr(event, "num", None) == 4:
            self.opt_controls_canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.opt_controls_canvas.yview_scroll(1, "units")

    def _add_mask_builder_button(self, parent, text, command, padx=(0, 0)):
        btn = ttk.Button(parent, text=text, command=command)
        btn.pack(side=tk.LEFT, fill="x", expand=True, padx=padx)
        self.mask_builder_action_buttons.append(btn)
        return btn

    def _on_builder_shape_change(self, *_args):
        self._update_builder_shape_controls()

    def _update_builder_shape_controls(self):
        for frame in (self.builder_circle_params, self.builder_ring_params, self.builder_rect_params):
            frame.pack_forget()

        shape = str(self.builder_shape_var.get()).strip().lower()
        if shape == "ring":
            self.builder_ring_params.pack(fill="x")
        elif shape == "rectangle":
            self.builder_rect_params.pack(fill="x")
        else:
            self.builder_circle_params.pack(fill="x")

    def _builder_geometry_values(self):
        nx = int(self.builder_nx_var.get())
        ny = int(self.builder_ny_var.get())
        size_x_mm = float(self.builder_size_x_mm_var.get())
        size_y_mm = float(self.builder_size_y_mm_var.get())
        if nx <= 0 or ny <= 0:
            raise ValueError("Mask builder Nx and Ny must be positive.")
        if size_x_mm <= 0 or size_y_mm <= 0:
            raise ValueError("Mask builder Size X/Y must be positive.")
        return nx, ny, size_x_mm, size_y_mm

    def _ensure_builder_model(self):
        if self.mask_builder_model is None:
            nx, ny, size_x_mm, size_y_mm = self._builder_geometry_values()
            self.mask_builder_model = MaskBuilderModel(
                nx=nx,
                ny=ny,
                size_x_mm=size_x_mm,
                size_y_mm=size_y_mm,
                init_value=1.0,
            )

    def _sync_builder_geometry(self):
        self._ensure_builder_model()
        nx, ny, size_x_mm, size_y_mm = self._builder_geometry_values()
        model = self.mask_builder_model
        if (
            int(model.nx) != nx
            or int(model.ny) != ny
            or abs(float(model.size_x_mm) - size_x_mm) > 1e-12
            or abs(float(model.size_y_mm) - size_y_mm) > 1e-12
        ):
            model.set_geometry(nx=nx, ny=ny, size_x_mm=size_x_mm, size_y_mm=size_y_mm, preserve=True)

    def _initialize_mask_builder_blank(self):
        try:
            nx, ny, size_x_mm, size_y_mm = self._builder_geometry_values()
            self.mask_builder_model = MaskBuilderModel(
                nx=nx,
                ny=ny,
                size_x_mm=size_x_mm,
                size_y_mm=size_y_mm,
                init_value=1.0,
            )
            self.mask_builder_status_var.set(
                f"Initialized blank all-pass mask ({nx} x {ny}, intensity=1 everywhere)."
            )
            self._refresh_mask_builder_plot()
        except Exception as exc:
            messagebox.showerror("Mask builder", str(exc))

    def _initialize_mask_builder_from_current(self):
        try:
            if self.mask_array is None:
                arr = np.ones((int(self.nx_var.get()), int(self.ny_var.get())), dtype=float)
                source = "all-pass mask"
            else:
                arr = np.asarray(self.mask_array, dtype=float)
                source = "current input mask"

            size_x_mm = float(self.size_x_mm_var.get())
            size_y_mm = float(self.size_y_mm_var.get())
            self.mask_builder_model = MaskBuilderModel(
                nx=arr.shape[0],
                ny=arr.shape[1],
                size_x_mm=size_x_mm,
                size_y_mm=size_y_mm,
                init_value=0.0,
            )
            self.mask_builder_model.set_mask(arr)

            self.builder_nx_var.set(int(arr.shape[0]))
            self.builder_ny_var.set(int(arr.shape[1]))
            self.builder_size_x_mm_var.set(size_x_mm)
            self.builder_size_y_mm_var.set(size_y_mm)
            self.mask_builder_status_var.set(f"Initialized builder from {source}.")
            self._refresh_mask_builder_plot()
        except Exception as exc:
            messagebox.showerror("Mask builder", str(exc))

    def _apply_builder_shape(self):
        try:
            self._sync_builder_geometry()
            model = self.mask_builder_model
            mode = str(self.builder_mode_var.get()).strip().lower()
            center_x_mm = float(self.builder_center_x_mm_var.get())
            center_y_mm = float(self.builder_center_y_mm_var.get())
            shape = str(self.builder_shape_var.get()).strip().lower()

            previous = model.get_mask()
            if shape == "ring":
                model.add_ring(
                    center_x_mm=center_x_mm,
                    center_y_mm=center_y_mm,
                    inner_radius_mm=float(self.builder_ring_inner_mm_var.get()),
                    outer_radius_mm=float(self.builder_ring_outer_mm_var.get()),
                    mode=mode,
                )
            elif shape == "rectangle":
                model.add_rectangle(
                    center_x_mm=center_x_mm,
                    center_y_mm=center_y_mm,
                    width_mm=float(self.builder_rect_x_size_mm_var.get()),
                    height_mm=float(self.builder_rect_y_size_mm_var.get()),
                    mode=mode,
                )
            else:
                model.add_circle(
                    center_x_mm=center_x_mm,
                    center_y_mm=center_y_mm,
                    radius_mm=float(self.builder_circle_radius_mm_var.get()),
                    mode=mode,
                )
            model.history.append(previous)
            if len(model.history) > model.max_history:
                model.history.pop(0)
            self.mask_builder_status_var.set(f"Applied {mode} {shape}.")
            self._refresh_mask_builder_plot()
        except Exception as exc:
            messagebox.showerror("Mask builder", str(exc))

    def _undo_builder_shape(self):
        try:
            self._ensure_builder_model()
            if self.mask_builder_model.undo():
                self.builder_nx_var.set(int(self.mask_builder_model.nx))
                self.builder_ny_var.set(int(self.mask_builder_model.ny))
                self.mask_builder_status_var.set("Undo applied.")
                self._refresh_mask_builder_plot()
            else:
                self.mask_builder_status_var.set("Nothing to undo.")
        except Exception as exc:
            messagebox.showerror("Mask builder", str(exc))

    def _clear_builder_mask(self):
        try:
            self._sync_builder_geometry()
            self.mask_builder_model.push_history()
            self.mask_builder_model.clear(value=1.0)
            self.mask_builder_status_var.set("Reset builder mask to all-pass (intensity=1 everywhere).")
            self._refresh_mask_builder_plot()
        except Exception as exc:
            messagebox.showerror("Mask builder", str(exc))

    def _save_builder_mask_dialog(self):
        try:
            self._sync_builder_geometry()
        except Exception as exc:
            messagebox.showerror("Mask builder", str(exc))
            return

        path = filedialog.asksaveasfilename(
            title="Save mask from builder",
            initialdir=self.mask_library_dir,
            defaultextension=".tif",
            filetypes=[
                ("TIFF image", "*.tif *.tiff"),
                ("NumPy array", "*.npy"),
                ("PNG image", "*.png"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self._save_builder_mask(path)
            self.mask_builder_status_var.set(f"Saved builder mask: {path}")
        except Exception as exc:
            messagebox.showerror("Mask builder", str(exc))

    def _save_builder_mask(self, path):
        import matplotlib.pyplot as plt

        arr = np.clip(self.mask_builder_model.get_mask(), 0.0, 1.0)
        ext = os.path.splitext(path)[1].lower()
        out_path = path
        if ext == "":
            out_path = f"{path}.tif"
            ext = ".tif"

        if ext == ".npy":
            np.save(out_path, arr)
            return

        if ext in {".tif", ".tiff"}:
            if tifffile is not None:
                tifffile.imwrite(out_path, arr.astype(np.float32))
            else:
                plt.imsave(out_path, arr, cmap="gray", vmin=0.0, vmax=1.0)
            return

        if ext in {".png", ".jpg", ".jpeg", ".bmp"}:
            plt.imsave(out_path, arr, cmap="gray", vmin=0.0, vmax=1.0)
            return

        raise ValueError("Unsupported extension. Use .tif/.tiff, .npy, or common image formats.")

    def _load_mask_into_builder_dialog(self):
        path = filedialog.askopenfilename(
            title="Load mask into builder",
            initialdir=self.mask_load_dir,
            filetypes=[
                ("Mask files", "*.npy *.csv *.txt *.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
                ("NumPy array", "*.npy"),
                ("CSV/TXT", "*.csv *.txt"),
                ("Image", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ],
        )
        if not path:
            return
        try:
            arr = self._read_mask(path)
            size_x_mm = float(self.builder_size_x_mm_var.get())
            size_y_mm = float(self.builder_size_y_mm_var.get())
            self.mask_builder_model = MaskBuilderModel(
                nx=arr.shape[0],
                ny=arr.shape[1],
                size_x_mm=size_x_mm,
                size_y_mm=size_y_mm,
                init_value=0.0,
            )
            self.mask_builder_model.set_mask(arr)
            self.builder_nx_var.set(int(arr.shape[0]))
            self.builder_ny_var.set(int(arr.shape[1]))
            self.mask_builder_status_var.set(f"Loaded mask into builder: {path}")
            self._refresh_mask_builder_plot()
        except Exception as exc:
            messagebox.showerror("Mask builder", str(exc))

    def _apply_builder_mask_to_simulation(self):
        try:
            self._sync_builder_geometry()
            model = self.mask_builder_model
            arr = np.clip(model.get_mask(), 0.0, 1.0)
            self.mask_array = arr
            self.mask_rotation_quadrants = 0
            self._update_mask_orientation_label()
            self.mask_path = None
            self.mask_path_var.set("Generated in Mask Builder")
            self.nx_var.set(int(arr.shape[0]))
            self.ny_var.set(int(arr.shape[1]))
            self.size_x_mm_var.set(float(model.size_x_mm))
            self.size_y_mm_var.set(float(model.size_y_mm))
            self.mask_info_var.set(
                "Loaded + normalized mask: "
                f"shape={arr.shape}, min={float(arr.min()):.3f}, max={float(arr.max()):.3f} (source=builder)"
            )
            self.mask_builder_status_var.set("Applied builder mask to simulation input.")
            self._refresh_mask_input_preview()
        except Exception as exc:
            messagebox.showerror("Mask builder", str(exc))

    def _refresh_mask_builder_plot(self):
        self.fig_mask.clear()
        ax = self.fig_mask.add_subplot(111)

        if self.mask_builder_model is None:
            ax.text(0.5, 0.5, "Initialize or load a mask to start editing.", ha="center", va="center")
            ax.set_title("Mask Builder")
            ax.axis("off")
            self.canvas_mask.draw()
            return

        arr = np.clip(self.mask_builder_model.get_mask(), 0.0, 1.0)
        extent = [
            -0.5 * float(self.mask_builder_model.size_x_mm),
            0.5 * float(self.mask_builder_model.size_x_mm),
            -0.5 * float(self.mask_builder_model.size_y_mm),
            0.5 * float(self.mask_builder_model.size_y_mm),
        ]
        im = ax.imshow(
            arr.T,
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            aspect="equal",
            interpolation="nearest",
        )
        style_image_axis(
            ax,
            title=f"Mask preview ({arr.shape[0]} x {arr.shape[1]})",
            xlabel="x (mm)",
            ylabel="y (mm)",
        )
        add_compact_colorbar(self.fig_mask, im, ax, "Mask value")
        self.canvas_mask.draw()

    def _init_empty_plots(self):
        self._clear_2d_width_cache()
        self._cursor_axes_map = {}
        self._profile_axes_map = {}
        self._profile_cursor_artists = {}
        self._profile_drag_ax = None
        self._axes_export_map = {}
        if hasattr(self, "cursor_info_var"):
            self.cursor_info_var.set("Cursor (XY/XZ/YZ): move mouse over 2D plots after a run.")
        self.fig_2d.clear()
        ax = self.fig_2d.add_subplot(111)
        ax.text(0.5, 0.5, "Run 2D or 3D simulation.", ha="center", va="center")
        ax.axis("off")
        self.canvas_2d.draw()

        self._close_pyvista_window()
        self._update_3d_window_button_state()
        self._update_export_csv_button_state()
        self._refresh_mask_builder_plot()
        self._refresh_mask_input_preview()
        self._refresh_optimization_plot()

    def _set_running(self, running, status_text):
        self.is_running = bool(running)
        self._update_dip_in_controls()
        run_state = "disabled" if (self.is_running or self.opt_is_running) else "normal"
        stop_state = "normal" if self.is_running else "disabled"
        self.run_2d_btn.configure(state=run_state)
        self.run_3d_btn.configure(state=run_state)
        self.stop_btn.configure(state=stop_state)
        self.save_btn.configure(state=run_state)
        self.load_btn.configure(state=run_state)
        if hasattr(self, "save_config_btn"):
            self.save_config_btn.configure(state=run_state)
        if hasattr(self, "load_config_btn"):
            self.load_config_btn.configure(state=run_state)
        self.load_mask_btn.configure(state=run_state)
        if hasattr(self, "show_3d_btn"):
            self._update_3d_window_button_state()
        if hasattr(self, "export_csv_btn"):
            self._update_export_csv_button_state()
        if hasattr(self, "remove_mask_btn"):
            self.remove_mask_btn.configure(state=run_state)
        self.rotate_cw_btn.configure(state=run_state)
        self.rotate_ccw_btn.configure(state=run_state)
        if hasattr(self, "store_objective_lens_btn"):
            self.store_objective_lens_btn.configure(state=run_state)
        if hasattr(self, "load_objective_lens_btn"):
            self.load_objective_lens_btn.configure(state=run_state)
        for btn in self.mask_builder_action_buttons:
            btn.configure(state=run_state)

        if hasattr(self, "opt_run_btn"):
            opt_run_state = "disabled" if (self.is_running or self.opt_is_running) else "normal"
            self.opt_run_btn.configure(state=opt_run_state)
        if hasattr(self, "opt_stop_btn"):
            self.opt_stop_btn.configure(state=("normal" if self.opt_is_running else "disabled"))
        if hasattr(self, "opt_save_btn"):
            can_use_opt = (
                (not self.is_running)
                and (not self.opt_is_running)
                and (self.opt_result is not None)
            )
            self.opt_save_btn.configure(state=("normal" if can_use_opt else "disabled"))
        if hasattr(self, "opt_apply_btn"):
            can_use_opt = (
                (not self.is_running)
                and (not self.opt_is_running)
                and (self.opt_result is not None)
            )
            self.opt_apply_btn.configure(state=("normal" if can_use_opt else "disabled"))
        self.status_var.set(status_text)

    def _set_progress(self, percent, status_text=None, force=False):
        pct = float(min(100.0, max(0.0, percent)))
        if self.is_running and (not force):
            pct = max(float(self.progress_var.get()), pct)
        self.progress_var.set(pct)
        self.progress_pct_var.set(f"{pct:.1f}%")
        if status_text is not None:
            self.status_var.set(status_text)

    def _load_mask(self):
        path = filedialog.askopenfilename(
            title="Load mask",
            initialdir=self.mask_load_dir,
            filetypes=[
                ("Mask files", "*.npy *.csv *.txt *.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
                ("NumPy array", "*.npy"),
                ("CSV/TXT", "*.csv *.txt"),
                ("Image", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ],
        )
        if not path:
            return

        try:
            arr = self._read_mask(path)
        except Exception as exc:
            messagebox.showerror("Mask load error", str(exc))
            return

        self.mask_array = arr
        self.post_fresnel_mask_array = None
        self.mask_rotation_quadrants = 0
        self._update_mask_orientation_label()
        self.mask_path = path
        self.mask_path_var.set(path)
        self.nx_var.set(int(arr.shape[0]))
        self.ny_var.set(int(arr.shape[1]))
        self.mask_info_var.set(
            "Loaded + normalized mask: "
            f"shape={arr.shape}, min={float(arr.min()):.3f}, max={float(arr.max()):.3f}"
        )
        self._refresh_mask_input_preview()

    def _clear_loaded_mask(self, set_status=True):
        self.mask_array = None
        self.post_fresnel_mask_array = None
        self.mask_rotation_quadrants = 0
        self._update_mask_orientation_label()
        self.mask_path = None
        if hasattr(self, "mask_path_var"):
            self.mask_path_var.set("")
        if hasattr(self, "mask_info_var"):
            self.mask_info_var.set("No file loaded. Using all-pass mask.")
        self._refresh_mask_input_preview()
        if set_status and hasattr(self, "status_var"):
            self.status_var.set("Loaded mask removed. Using all-pass mask.")

    def _remove_loaded_mask(self):
        self._clear_loaded_mask(set_status=True)

    @staticmethod
    def _read_mask(path):
        return read_mask_file(path, normalization="physical")

    @staticmethod
    def _resize_mask(mask, shape):
        return resize_mask_array(mask, shape)

    @staticmethod
    def _clone_payload(data):
        return copy.deepcopy(data)

    @staticmethod
    def _encode_precise_params(params):
        """
        Encode numeric parameters with exact float round-trip semantics.

        Floats are serialized as hexadecimal strings via float.hex(), which
        preserves the exact IEEE-754 value with no precision loss.
        """
        def encode_value(value):
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, (np.bool_,)):
                return bool(value)
            if isinstance(value, (int, np.integer)) and (not isinstance(value, bool)):
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

    @staticmethod
    def _decode_precise_params(params):
        def decode_value(value):
            if (
                isinstance(value, dict)
                and value.get("type") == "float64_hex"
                and "value" in value
            ):
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

    def _collect_optimizer_last_run_values(self):
        if self.opt_result is None:
            return {}
        payload = {
            "summary_text": str(self.opt_summary_var.get()),
            "best_objective": float(self.opt_result.best_objective),
            "baseline_objective": float(self.opt_result.baseline_objective),
            "best_metrics_um": self._clone_payload(self.opt_result.best_metrics_um),
            "baseline_metrics_um": self._clone_payload(self.opt_result.baseline_metrics_um),
            "settings": self._clone_payload(self.opt_result.settings),
        }
        if isinstance(self.opt_last_params, dict):
            payload["params"] = self._clone_payload(self.opt_last_params)
        return payload

    def _apply_optimizer_last_run_values(self, values):
        if not isinstance(values, dict):
            return
        self.opt_last_summary = self._clone_payload(values)
        summary_text = str(values.get("summary_text", "")).strip()
        if summary_text != "":
            self.opt_summary_var.set(summary_text)
            return

        best = values.get("best_metrics_um")
        baseline = values.get("baseline_metrics_um")
        best_obj = values.get("best_objective")
        formatted = self._format_optimizer_summary(
            baseline,
            best,
            best_obj,
            settings=values.get("settings"),
            title="Last optimization (loaded)",
        )
        if formatted is not None:
            self.opt_summary_var.set(formatted)

    def _set_var_if_present(self, params, key, var, cast, scale=1.0):
        if key not in params:
            return
        try:
            raw = params[key]
            if cast is int:
                numeric = float(raw)
                if (not np.isfinite(numeric)) or (not numeric.is_integer()):
                    raise ValueError(f"{key} must be an integer.")
                value = int(round(numeric * scale))
            else:
                value = cast(raw) * scale
            if isinstance(value, float) and (not np.isfinite(value)):
                raise ValueError(f"{key} must be finite.")
            var.set(value)
        except Exception as exc:
            raise ValueError(f"Invalid config value for {key}: {params[key]!r}.") from exc

    @staticmethod
    def _finite_float_or_none(value):
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        return value if np.isfinite(value) else None

    def _set_bool_var_if_present(self, values, key, var):
        if key not in values:
            return
        if not isinstance(values[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be a JSON boolean.")
        var.set(bool(values[key]))

    def _set_choice_var_if_valid(self, values, key, var, valid_values, lowercase=False):
        if key not in values:
            return
        try:
            choice = str(values[key]).strip()
        except Exception:
            return
        if lowercase:
            choice = choice.lower()
        if choice in valid_values:
            var.set(choice)

    def _apply_value_specs(self, values, specs):
        for key, var, cast, scale in specs:
            self._set_var_if_present(values, key, var, cast, scale)

    # Non-variable state that loading a config may replace.
    _TRANSACTION_ATTRIBUTES = (
        "_dip_in_previous_values",
        "config_extensions",
        "mask_array",
        "post_fresnel_mask_array",
        "mask_path",
        "mask_rotation_quadrants",
    )

    @contextlib.contextmanager
    def _control_transaction(self):
        """Roll every control back if applying settings fails part-way."""
        variables = [
            value for value in vars(self).values()
            if all(hasattr(value, name) for name in ("get", "set", "trace_add"))
        ]
        # Raw Tcl values also capture half-typed text that get() rejects.
        saved_variables = [
            (var, var._tk.globalgetvar(var._name) if hasattr(var, "_tk") else var.get())
            for var in variables
        ]
        # Loading replaces these objects rather than mutating them, so keeping
        # references is enough (and avoids copying large mask arrays).
        saved_attributes = {
            name: getattr(self, name) for name in self._TRANSACTION_ATTRIBUTES if hasattr(self, name)
        }
        try:
            yield
        except BaseException:
            for name, value in saved_attributes.items():
                setattr(self, name, value)
            for var, raw in saved_variables:
                if hasattr(var, "_tk"):
                    var._tk.globalsetvar(var._name, raw)
                else:
                    var.set(raw)
            self._update_dip_in_controls()
            self._update_model_description()
            self._on_integration_mode_change()
            if hasattr(self, "mask_rotation_quadrants"):
                self._update_mask_orientation_label()
            self._update_sampling_info()
            self._refresh_mask_input_preview()
            raise

    def _apply_axis_range_values(self, values):
        loaded_axis_range = False
        for key, var in (
            ("range_x_um", self.range_x_um_var),
            ("range_y_um", self.range_y_um_var),
            ("range_z_um", self.range_z_um_var),
        ):
            if key in values:
                self._set_var_if_present(values, key, var, float, 1.0)
                loaded_axis_range = True
        if loaded_axis_range:
            return

        r_um = self._finite_float_or_none(values.get("range_um"))
        if r_um is None:
            return
        self.range_x_um_var.set(r_um)
        self.range_y_um_var.set(r_um)
        self.range_z_um_var.set(r_um)

    def _optimizer_structure_summary_line(self, settings, params=None):
        if not isinstance(settings, dict):
            return ""

        bstruct = str(settings.get("binary_structure", "")).strip().lower()
        if bstruct == "simple_ring_block":
            inner_mm = self._finite_float_or_none(settings.get("ring_inner_radius_mm"))
            outer_mm = self._finite_float_or_none(settings.get("ring_outer_radius_mm"))
            if inner_mm is None or outer_mm is None:
                inner_norm = self._finite_float_or_none(settings.get("ring_inner_radius_norm"))
                outer_norm = self._finite_float_or_none(settings.get("ring_outer_radius_norm"))
                pupil_d = None
                if isinstance(params, dict):
                    pupil_d = self._finite_float_or_none(params.get("pupil_d"))
                if inner_norm is None or outer_norm is None or pupil_d is None:
                    return ""
                pupil_radius_mm = 0.5 * pupil_d * 1e3
                inner_mm = inner_norm * pupil_radius_mm
                outer_mm = outer_norm * pupil_radius_mm
            return f"\nRing block radii (mm): inner={inner_mm:.4f}, outer={outer_mm:.4f}"

        if bstruct != "simple_ellipse_block":
            return ""

        ellipse_mm = tuple(
            self._finite_float_or_none(settings.get(key))
            for key in (
                "ellipse_inner_diameter_x_mm",
                "ellipse_inner_diameter_y_mm",
                "ellipse_outer_diameter_x_mm",
                "ellipse_outer_diameter_y_mm",
            )
        )
        if all(value is not None for value in ellipse_mm):
            inner_x_mm, inner_y_mm, outer_x_mm, outer_y_mm = ellipse_mm
            return (
                "\nEllipse ring diameters (mm): "
                f"inner=({inner_x_mm:.4f}, {inner_y_mm:.4f}), "
                f"outer=({outer_x_mm:.4f}, {outer_y_mm:.4f})"
            )

        radius_x_mm = self._finite_float_or_none(settings.get("ellipse_radius_x_mm"))
        radius_y_mm = self._finite_float_or_none(settings.get("ellipse_radius_y_mm"))
        if radius_x_mm is None or radius_y_mm is None:
            return ""
        return (
            "\nEllipse block radii (mm): "
            f"x={radius_x_mm:.4f}, y={radius_y_mm:.4f}"
        )

    def _format_optimizer_summary(
        self,
        before,
        after,
        objective,
        *,
        settings=None,
        params=None,
        title="",
    ):
        if not isinstance(before, dict) or not isinstance(after, dict):
            return None
        keys = ("width_x_um", "width_y_um", "width_z_um")
        if not all(key in before for key in keys) or not all(key in after for key in keys):
            return None

        try:
            abs_before = abs(float(before["width_x_um"]) - float(before["width_y_um"]))
            abs_after = abs(float(after["width_x_um"]) - float(after["width_y_um"]))
            title_prefix = "" if title == "" else f"{title}\n"
            return (
                f"{title_prefix}"
                "Before (um): "
                f"X={float(before['width_x_um']):.4f}, "
                f"Y={float(before['width_y_um']):.4f}, "
                f"Z={float(before['width_z_um']):.4f}\n"
                "After (um): "
                f"X={float(after['width_x_um']):.4f}, "
                f"Y={float(after['width_y_um']):.4f}, "
                f"Z={float(after['width_z_um']):.4f}\n"
                f"abs(x-y) (um): {abs_before:.4f} -> {abs_after:.4f}\n"
                f"Objective: {float(objective):.6f}"
                f"{self._optimizer_structure_summary_line(settings, params=params)}"
            )
        except Exception:
            return None

    @staticmethod
    def _pyvista_unavailable_message():
        backend_error = PYVISTA_IMPORT_ERROR if not HAS_PYVISTA else PYVISTA_QT_IMPORT_ERROR
        message = "PyVista window rendering is unavailable."
        if backend_error:
            message = f"{message} {backend_error}"
        return message

    def _collect_objective_lens_values(self):
        return {spec.ui_key: float(self._param_var(spec).get()) for spec in OBJECTIVE_SPECS}

    def _apply_objective_lens_values(self, values):
        validated = objective_ui_values(values)
        for spec in OBJECTIVE_SPECS:
            self._param_var(spec).set(validated[spec.ui_key])
        self._refresh_mask_input_preview()

    def _save_objective_lens_to_path(self, path):
        payload = {
            "format": "psf_sim_gui_objective_lens_preset",
            "version": 1,
            "values": self._encode_precise_params(self._collect_objective_lens_values()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_objective_lens_from_path(self, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Invalid objective lens file: expected dictionary payload.")
        values = payload.get("values", payload)
        if not isinstance(values, dict):
            raise ValueError("Invalid objective lens file: missing values dictionary.")
        decoded = self._decode_precise_params(values)
        self._apply_objective_lens_values(decoded)

    def _save_objective_lens_dialog(self):
        path = filedialog.asksaveasfilename(
            title="Store Objective Lens",
            initialdir=self.objective_lens_dir,
            defaultextension=".json",
            filetypes=[
                ("Objective lens preset", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self._save_objective_lens_to_path(path)
            self.status_var.set(f"Stored objective lens: {path}")
        except Exception as exc:
            messagebox.showerror("Store objective lens", str(exc))

    def _load_objective_lens_dialog(self):
        path = filedialog.askopenfilename(
            title="Load Objective Lens",
            initialdir=self.objective_lens_dir,
            filetypes=[
                ("Objective lens preset", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self._load_objective_lens_from_path(path)
            self.status_var.set(f"Loaded objective lens: {path}")
        except Exception as exc:
            messagebox.showerror("Load objective lens", str(exc))

    def _collect_control_values(self):
        self._sync_dip_in_media()
        values = {
            "dip_in_enabled": bool(self.dip_in_enabled_var.get()),
            "dip_in_previous_values": self._clone_payload(getattr(self, "_dip_in_previous_values", None)),
        }
        for spec in PARAM_SPECS:
            values[spec.ui_key] = spec.ui_value(self._param_var(spec).get())
        return values

    def _collect_optimizer_control_values(self):
        if not hasattr(self, "opt_optimizer_var"):
            return {}
        return {
            "optimizer_category": str(self.opt_optimizer_category_var.get()).strip(),
            "optimizer_name": str(self.opt_optimizer_var.get()).strip(),
            "mode": str(self.opt_mode_var.get()).strip().lower(),
            "axial_symmetry": bool(self.opt_axial_symmetry_var.get()),
            "binary_amplitude": bool(self.opt_binary_amplitude_var.get()),
            "binary_structure": str(self.opt_binary_structure_var.get()).strip().lower(),
            "width_level": float(self.opt_width_level_var.get()),
            "weight_x": float(self.opt_weight_x_var.get()),
            "weight_y": float(self.opt_weight_y_var.get()),
            "weight_z": float(self.opt_weight_z_var.get()),
            "weight_abs_xy": float(self.opt_weight_abs_xy_var.get()),
            "n_starts": int(self.opt_n_starts_var.get()),
            "maxiter": int(self.opt_maxiter_var.get()),
            "ftol": float(self.opt_ftol_var.get()),
            "gtol": float(self.opt_gtol_var.get()),
            "seed": str(self.opt_seed_var.get()).strip(),
        }

    def _apply_control_values(self, values):
        if not isinstance(values, dict):
            raise ValueError("values must be a dictionary.")
        with self._control_transaction():
            self.dip_in_enabled_var.set(False)
            self._apply_spec_values(values, si_units=False)
            self._restore_dip_in_control_state(values)

            self._update_sampling_info()
            self._refresh_mask_input_preview()

    def _apply_optimizer_control_values(self, values):
        if not isinstance(values, dict) or (not hasattr(self, "opt_optimizer_var")):
            return

        category_map = self._get_optimizer_category_map()
        self._opt_optimizer_categories = dict(category_map)
        all_opt_names = self._all_optimizer_names(category_map)
        opt_name = str(values.get("optimizer_name", self.opt_optimizer_var.get())).strip()
        if opt_name not in all_opt_names:
            opt_name = str(self.opt_optimizer_var.get()).strip()
        category = str(
            values.get(
                "optimizer_category",
                self._optimizer_category_for_name(opt_name, category_map) or self.opt_optimizer_category_var.get(),
            )
        ).strip()
        if category not in category_map:
            inferred = self._optimizer_category_for_name(opt_name, category_map)
            if inferred is not None:
                category = inferred
            elif len(category_map) > 0:
                category = next(iter(category_map.keys()))
        if category != "":
            self.opt_optimizer_category_var.set(category)
        self._sync_optimizer_combo_to_category(preferred_optimizer=opt_name)

        self._set_choice_var_if_valid(
            values,
            "mode",
            self.opt_mode_var,
            {"amplitude", "phase", "amplitude_phase"},
            lowercase=True,
        )
        self._set_bool_var_if_present(values, "axial_symmetry", self.opt_axial_symmetry_var)
        self._set_bool_var_if_present(values, "binary_amplitude", self.opt_binary_amplitude_var)
        self._set_choice_var_if_valid(
            values,
            "binary_structure",
            self.opt_binary_structure_var,
            {"pixel", "simple_ring_block", "simple_ellipse_block"},
            lowercase=True,
        )
        self._apply_value_specs(
            values,
            (
                ("width_level", self.opt_width_level_var, float, 1.0),
                ("weight_x", self.opt_weight_x_var, float, 1.0),
                ("weight_y", self.opt_weight_y_var, float, 1.0),
                ("weight_z", self.opt_weight_z_var, float, 1.0),
                ("weight_abs_xy", self.opt_weight_abs_xy_var, float, 1.0),
                ("n_starts", self.opt_n_starts_var, int, 1.0),
                ("maxiter", self.opt_maxiter_var, int, 1.0),
                ("ftol", self.opt_ftol_var, float, 1.0),
                ("gtol", self.opt_gtol_var, float, 1.0),
            ),
        )
        if "seed" in values:
            try:
                self.opt_seed_var.set(str(values["seed"]).strip())
            except Exception:
                pass

        self._on_opt_mode_change()

    def _apply_params_to_controls(self, params):
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary.")
        with self._control_transaction():
            self.dip_in_enabled_var.set(False)
            self._apply_spec_values(params, si_units=True)
            self._restore_dip_in_control_state(params, core_units=True)

            self._update_sampling_info()
            self._refresh_mask_input_preview()

    def _apply_spec_values(self, values, *, si_units):
        """Set each GUI parameter found in SI ``params`` or display-unit ``ui_values``."""
        converted = set()
        if not si_units:
            converted = {new for old, (new, _f) in LEGACY_UI_KEYS.items() if old in values and new not in values}
            values = upgrade_ui_values(values)
        for spec in PARAM_SPECS:
            if spec.reset_on_load:
                self._param_var(spec).set(spec.ui_default)
        for spec in PARAM_SPECS:
            if spec.ui_key in self._AXIS_RANGE_KEYS:
                continue  # also accepts the legacy single "range_um" key
            key = spec.key if si_units else spec.ui_key
            var = self._param_var(spec)
            if spec.choices:
                self._set_choice_var_if_valid(values, key, var, set(spec.choices), lowercase=True)
            elif spec.cast is bool:
                self._set_bool_var_if_present(values, key, var)
            else:
                self._set_var_if_present(values, key, var, spec.cast, spec.from_si if si_units else 1.0)
                if spec.display_digits and key in values and (si_units or key in converted):
                    # e.g. a stored 1/e radius shown as FWHM: 7.49299, not 7.492991632690848
                    var.set(float(f"{float(var.get()):.{spec.display_digits}g}"))
        self._on_integration_mode_change()
        self._apply_axis_range_values(values)

    def _save_last_used_config(self, params):
        if not isinstance(params, dict):
            return
        optimizer_last_run = self._collect_optimizer_last_run_values()
        payload = {
            "format": "psf_sim_gui_last_used_config",
            "version": 6,
            "params": self._encode_precise_params(self._clone_payload(params)),
            "ui_values": self._encode_precise_params(self._collect_control_values()),
            "optimizer_ui_values": self._encode_precise_params(
                self._collect_optimizer_control_values()
            ),
            "optimizer_last_run": self._encode_precise_params(optimizer_last_run),
        }
        try:
            sidecar = os.path.join(
                os.path.dirname(self.last_config_path), "last_used_simulation_masks.npz"
            )
            if self.mask_array is not None or self.post_fresnel_mask_array is not None:
                arrays = {}
                if self.mask_array is not None:
                    arrays["pre_fresnel_intensity"] = np.asarray(
                        self.mask_array, dtype=np.float64
                    )
                if self.post_fresnel_mask_array is not None:
                    value = np.asarray(self.post_fresnel_mask_array, dtype=np.complex128)
                    arrays["complex_mask_real"] = value.real
                    arrays["complex_mask_imag"] = value.imag
                np.savez_compressed(sidecar, **arrays)
                payload["mask"] = {
                    "source": (
                        "optimized_complex"
                        if self.post_fresnel_mask_array is not None
                        else "builder"
                    ),
                    "path": os.path.basename(sidecar),
                    "normalization": "physical",
                    "rotation_quadrants_ccw": int(self._mask_rotation_quadrants_mod()),
                    "loaded": True,
                }
            with open(self.last_config_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            return

    def _build_simulation_config_payload(self):
        params = self._collect_params()
        payload = {
            "format": "psf_sim_gui_config",
            "version": 2,
            "params": self._encode_precise_params(self._clone_payload(params)),
            "ui_values": self._encode_precise_params(self._collect_control_values()),
            "optimizer_ui_values": self._encode_precise_params(
                self._collect_optimizer_control_values()
            ),
            "mask": {
                "path": self.mask_path,
                "source": (
                    "optimized_complex"
                    if self.post_fresnel_mask_array is not None
                    else ("file" if self.mask_path else ("builder" if self.mask_array is not None else "all_pass"))
                ),
                "normalization": "physical",
                "plane": (
                    "post_fresnel_pre_pupil"
                    if self.post_fresnel_mask_array is not None
                    else "pre_fresnel_intensity"
                ),
                "rotation_quadrants_ccw": int(self._mask_rotation_quadrants_mod()),
                "loaded": bool(
                    self.mask_array is not None or self.post_fresnel_mask_array is not None
                ),
            },
        }
        if self.config_extensions:
            payload["extensions"] = self._clone_payload(self.config_extensions)
        if self.mask_array is not None:
            payload["mask"]["shape"] = [int(v) for v in np.asarray(self.mask_array).shape]
        return payload

    def _save_simulation_config_to_path(self, path):
        payload = self._build_simulation_config_payload()
        config_path = os.path.abspath(path)
        mask_info = payload["mask"]
        source = str(mask_info.get("source", "all_pass"))
        if source == "file" and self.mask_path:
            try:
                mask_info["path"] = os.path.relpath(
                    os.path.abspath(self.mask_path), os.path.dirname(config_path)
                )
            except Exception:
                mask_info["path"] = self.mask_path
        elif source in {"builder", "optimized_complex"}:
            stem = os.path.splitext(os.path.basename(config_path))[0]
            sidecar = os.path.join(os.path.dirname(config_path), f"{stem}.masks.npz")
            arrays = {}
            if self.mask_array is not None:
                arrays["pre_fresnel_intensity"] = np.asarray(self.mask_array, dtype=np.float64)
            if self.post_fresnel_mask_array is not None:
                value = np.asarray(self.post_fresnel_mask_array, dtype=np.complex128)
                arrays["complex_mask_real"] = value.real
                arrays["complex_mask_imag"] = value.imag
            np.savez_compressed(sidecar, **arrays)
            mask_info["path"] = os.path.basename(sidecar)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    def _load_simulation_config_from_path(self, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Simulation config must be a JSON object.")
        version = int(payload.get("version", 1))
        if version > 2 and payload.get("format") != "psf_sim_gui_last_used_config":
            raise ValueError(f"Unsupported future simulation config version: {version}.")
        with self._control_transaction():
            known = {
                "format",
                "version",
                "params",
                "ui_values",
                "optimizer_ui_values",
                "optimizer_settings",
                "optimizer_last_run",
                "mask",
                "extensions",
            }
            extensions = payload.get("extensions", {})
            self.config_extensions = dict(extensions) if isinstance(extensions, dict) else {}
            for key, value in payload.items():
                if key not in known:
                    self.config_extensions[str(key)] = self._clone_payload(value)
            params = payload.get("params")
            if isinstance(params, dict):
                self._apply_params_to_controls(self._decode_precise_params(params))
            ui_values = payload.get("ui_values")
            if isinstance(ui_values, dict):
                self._apply_control_values(self._decode_precise_params(ui_values))
            opt_values = payload.get("optimizer_ui_values") or payload.get("optimizer_settings")
            if isinstance(opt_values, dict):
                self._apply_optimizer_control_values(self._decode_precise_params(opt_values))

            info = payload.get("mask", {})
            if isinstance(info, dict) and bool(
                info.get("loaded", info.get("source", "all_pass") != "all_pass")
            ):
                raw_path = info.get("path")
                if raw_path:
                    mask_path = os.path.expanduser(str(raw_path))
                    if not os.path.isabs(mask_path):
                        mask_path = os.path.join(os.path.dirname(os.path.abspath(path)), mask_path)
                    source = str(info.get("source", "file"))
                    if source in {"builder", "optimized_complex"} or mask_path.lower().endswith(".npz"):
                        with np.load(mask_path, allow_pickle=False) as data:
                            if "pre_fresnel_intensity" in data.files:
                                self.mask_array = np.asarray(
                                    data["pre_fresnel_intensity"], dtype=np.float64
                                )
                            if {"complex_mask_real", "complex_mask_imag"}.issubset(data.files):
                                self.post_fresnel_mask_array = np.asarray(
                                    data["complex_mask_real"], dtype=np.float64
                                ) + 1j * np.asarray(data["complex_mask_imag"], dtype=np.float64)
                        self.mask_path = None
                        self.mask_path_var.set(f"From config sidecar: {mask_path}")
                    else:
                        self.mask_array = self._read_mask(mask_path)
                        self.post_fresnel_mask_array = None
                        self.mask_path = mask_path
                        self.mask_path_var.set(mask_path)
                    self.mask_rotation_quadrants = int(
                        info.get("rotation_quadrants_ccw", 0)
                    ) % 4
                    self._update_mask_orientation_label()
                    if self.mask_array is not None:
                        self.nx_var.set(int(self.mask_array.shape[0]))
                        self.ny_var.set(int(self.mask_array.shape[1]))
                    self.mask_info_var.set(f"Loaded config mask source: {source}.")
                    self._refresh_mask_input_preview()
            else:
                self._clear_loaded_mask(set_status=False)
        return payload

    def _load_simulation_config_dialog(self):
        path = filedialog.askopenfilename(
            title="Load simulation config",
            initialdir=self.saved_simulation_dir,
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self._load_simulation_config_from_path(path)
            self.status_var.set(f"Loaded simulation config: {path}")
        except Exception as exc:
            messagebox.showerror("Load simulation config", str(exc))

    def _save_simulation_config_dialog(self):
        try:
            self._build_simulation_config_payload()
        except Exception as exc:
            messagebox.showerror("Save simulation config", str(exc))
            return

        path = filedialog.asksaveasfilename(
            title="Save simulation config",
            initialdir=self.saved_simulation_dir,
            initialfile="simulation_config.json",
            defaultextension=".json",
            filetypes=[
                ("JSON config", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self._save_simulation_config_to_path(path)
            self.status_var.set(f"Saved simulation config: {path}")
        except Exception as exc:
            messagebox.showerror("Save simulation config", str(exc))

    def _load_last_used_config(self):
        if not os.path.isfile(self.last_config_path):
            return False
        try:
            with open(self.last_config_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False

        if not isinstance(payload, dict):
            return False
        loaded_any = False

        ui_values = payload.get("ui_values")
        if isinstance(ui_values, dict):
            try:
                self._apply_control_values(self._decode_precise_params(ui_values))
                loaded_any = True
            except Exception:
                pass

        if not loaded_any:
            params = payload.get("params")
            if params is None:
                # Backward compatibility if file stored bare params dict.
                params = payload
            if isinstance(params, dict):
                try:
                    params_decoded = self._decode_precise_params(params)
                    self._apply_params_to_controls(params_decoded)
                    loaded_any = True
                except Exception:
                    pass

        opt_values = payload.get("optimizer_ui_values")
        if isinstance(opt_values, dict):
            try:
                self._apply_optimizer_control_values(self._decode_precise_params(opt_values))
                loaded_any = True
            except Exception:
                pass

        opt_last_run = payload.get("optimizer_last_run")
        if isinstance(opt_last_run, dict):
            decoded_opt_last = self._decode_precise_params(opt_last_run)
            settings_from_last = decoded_opt_last.get("settings")
            if (not isinstance(opt_values, dict)) and isinstance(settings_from_last, dict):
                try:
                    self._apply_optimizer_control_values(settings_from_last)
                    loaded_any = True
                except Exception:
                    pass
            try:
                self._apply_optimizer_last_run_values(decoded_opt_last)
                loaded_any = True
            except Exception:
                pass

        if isinstance(payload.get("mask"), dict):
            try:
                self._load_simulation_config_from_path(self.last_config_path)
                loaded_any = True
            except Exception:
                pass

        return bool(loaded_any)

    def _save_result_dialog(self):
        if self.results_2d is None or self.last_params is None:
            messagebox.showwarning("No result", "Run a simulation first, then save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save simulation result",
            initialdir=self.saved_simulation_dir,
            defaultextension=".psfres",
            filetypes=[
                ("PSF Result", "*.psfres"),
                ("Pickle", "*.pkl"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._save_result_to_path(path)

    def _save_result_to_path(self, path):
        save_result_file(path, self.last_params, self.results_2d, self.results_3d)
        has_3d = "yes" if self.results_3d is not None else "no"
        self.status_var.set(f"Saved result. 3D included: {has_3d}.")

    def _collect_current_export_display_data(self):
        if self.results_2d is None or self.last_simulator is None:
            raise ValueError("Run or load a simulation result first.")

        r = self.results_2d
        x_um = np.asarray(r["x_coords"], dtype=float) * 1e6
        y_um = np.asarray(r["y_coords"], dtype=float) * 1e6
        z_um = np.asarray(r["z_coords"], dtype=float) * 1e6

        i_xy = np.asarray(
            self._display_intensity(self._choose_result_array(r, "I_xy", "I_xy_norm")),
            dtype=float,
        )
        i_xz = np.asarray(
            self._display_intensity(self._choose_result_array(r, "I_xz", "I_xz_norm")),
            dtype=float,
        )
        i_yz = np.asarray(
            self._display_intensity(self._choose_result_array(r, "I_yz", "I_yz_norm")),
            dtype=float,
        )

        width_metrics, x_prof, y_prof, z_prof = self._resolve_width_metrics_and_profiles(
            level=self._current_width_level(),
            exact=bool(self._width_level_source == "entry"),
        )
        if width_metrics is None:
            raise RuntimeError("Profile data are unavailable for the current 2D result.")

        focus_z_um = np.nan
        try:
            if "z_focus" in r:
                focus_z_um = float(r["z_focus"]) * 1e6
        except Exception:
            focus_z_um = np.nan

        return {
            "x_um": x_um,
            "y_um": y_um,
            "z_um": z_um,
            "i_xy": i_xy,
            "i_xz": i_xz,
            "i_yz": i_yz,
            "x_profile": np.asarray(x_prof, dtype=float),
            "y_profile": np.asarray(y_prof, dtype=float),
            "z_profile": np.asarray(z_prof, dtype=float),
            "width_metrics": dict(width_metrics),
            "plot_mode": str(self.plot_mode_var.get()),
            "normalized": bool(self.normalize_var.get()),
            "focus_z_um": float(focus_z_um),
        }

    @staticmethod
    def _write_csv_metadata_row(writer, dataset, key, value):
        writer.writerow(
            [
                "metadata",
                str(dataset),
                str(key),
                value,
                "",
                "",
                "",
                "",
                "",
            ]
        )

    @staticmethod
    def _write_csv_heatmap_rows(
        writer,
        dataset,
        axis_1_label,
        axis_1_coords,
        axis_2_label,
        axis_2_coords,
        values,
    ):
        x = np.asarray(axis_1_coords, dtype=float).ravel()
        y = np.asarray(axis_2_coords, dtype=float).ravel()
        arr = np.asarray(values, dtype=float)
        expected_shape = (int(y.size), int(x.size))
        if arr.shape != expected_shape:
            if arr.shape == (int(x.size), int(y.size)):
                arr = arr.T
            else:
                raise ValueError(
                    f"{dataset} heatmap shape mismatch: got {arr.shape}, expected {expected_shape}."
                )
        for iy, yv in enumerate(y):
            row = arr[iy]
            for ix, xv in enumerate(x):
                writer.writerow(
                    [
                        "heatmap",
                        str(dataset),
                        "",
                        "",
                        str(axis_1_label),
                        float(xv),
                        str(axis_2_label),
                        float(yv),
                        float(row[ix]),
                    ]
                )

    @staticmethod
    def _write_csv_profile_rows(writer, dataset, axis_label, axis_coords, values):
        x = np.asarray(axis_coords, dtype=float).ravel()
        y = np.asarray(values, dtype=float).ravel()
        if int(x.size) != int(y.size):
            raise ValueError(
                f"{dataset} profile length mismatch: got {int(y.size)} values for {int(x.size)} coordinates."
            )
        for xv, yv in zip(x, y):
            writer.writerow(
                [
                    "profile",
                    str(dataset),
                    "",
                    "",
                    str(axis_label),
                    float(xv),
                    "",
                    "",
                    float(yv),
                ]
            )

    def _write_current_data_csv(self, path):
        export = self._collect_current_export_display_data()
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
            self._write_csv_metadata_row(writer, "export", "format", "psf_sim_gui_csv_export_v1")
            self._write_csv_metadata_row(writer, "display", "plot_mode", export["plot_mode"])
            self._write_csv_metadata_row(writer, "display", "normalized", int(export["normalized"]))
            self._write_csv_metadata_row(
                writer,
                "display",
                "width_level",
                float(export["width_metrics"].get("level", self._current_width_level())),
            )
            self._write_csv_metadata_row(writer, "result", "focus_z_um", export["focus_z_um"])
            self._write_csv_metadata_row(
                writer, "result", "width_x_um", float(export["width_metrics"].get("width_x_um", np.nan))
            )
            self._write_csv_metadata_row(
                writer, "result", "width_y_um", float(export["width_metrics"].get("width_y_um", np.nan))
            )
            self._write_csv_metadata_row(
                writer, "result", "width_z_um", float(export["width_metrics"].get("width_z_um", np.nan))
            )
            self._write_csv_metadata_row(
                writer,
                "result",
                "width_ratio_zx",
                float(export["width_metrics"].get("width_ratio_zx", np.nan)),
            )
            self._write_csv_metadata_row(
                writer,
                "result",
                "width_ratio_zy",
                float(export["width_metrics"].get("width_ratio_zy", np.nan)),
            )

            self._write_csv_heatmap_rows(
                writer,
                "XY",
                "x_um",
                export["x_um"],
                "y_um",
                export["y_um"],
                export["i_xy"],
            )
            self._write_csv_heatmap_rows(
                writer,
                "XZ",
                "x_um",
                export["x_um"],
                "z_um",
                export["z_um"],
                export["i_xz"],
            )
            self._write_csv_heatmap_rows(
                writer,
                "YZ",
                "y_um",
                export["y_um"],
                "z_um",
                export["z_um"],
                export["i_yz"],
            )

            self._write_csv_profile_rows(writer, "X", "x_um", export["x_um"], export["x_profile"])
            self._write_csv_profile_rows(writer, "Y", "y_um", export["y_um"], export["y_profile"])
            self._write_csv_profile_rows(writer, "Z", "z_um", export["z_um"], export["z_profile"])

    def _export_current_data_csv_dialog(self):
        if self.results_2d is None or self.last_simulator is None:
            messagebox.showwarning("No result", "Run or load a simulation first, then export CSV.")
            return

        path = filedialog.asksaveasfilename(
            title="Export current simulation data as CSV",
            initialdir=self.saved_simulation_dir,
            initialfile="psf_data_export.csv",
            defaultextension=".csv",
            filetypes=[
                ("CSV", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        out_path = path if ext != "" else f"{path}.csv"
        try:
            self._write_current_data_csv(out_path)
            self.status_var.set(f"Exported simulation CSV: {out_path}")
        except Exception as exc:
            messagebox.showerror("Export CSV", str(exc))

    def _load_result_dialog(self):
        path = filedialog.askopenfilename(
            title="Load simulation result",
            initialdir=self.saved_simulation_dir,
            filetypes=[
                ("PSF Result", "*.psfres *.pkl"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self._load_result_from_path(path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _load_result_from_path(self, path):
        try:
            payload = load_result_file(path)
        except ValueError as exc:
            if "Legacy pickle result is unsafe" not in str(exc):
                raise
            trusted = messagebox.askyesno(
                "Unsafe legacy result",
                "This legacy pickle can execute code while loading. Load it only if you trust its source. Continue?",
            )
            if not trusted:
                raise ValueError("Legacy result loading was cancelled.") from exc
            payload = load_result_file(path, allow_unsafe_legacy_pickle=True)
        if not isinstance(payload, dict):
            raise ValueError("Invalid result file: expected dictionary payload.")
        if "params" not in payload or "results_2d" not in payload:
            raise ValueError("Invalid result file: missing params or 2D results.")

        params = payload["params"]
        results_2d = payload["results_2d"]
        results_3d = payload.get("results_3d")
        simulator = self._create_simulator_from_params(params)

        self.last_params = self._clone_payload(params)
        self.last_simulator = simulator
        self.results_2d = self._clone_payload(results_2d)
        self.results_3d = self._clone_payload(results_3d) if results_3d is not None else None
        self._clear_2d_width_cache()
        self._clear_3d_render_cache()
        self.stop_requested = False
        self._set_progress(100.0, force=True)
        self._refresh_plots()
        self._update_3d_window_button_state()
        self._update_export_csv_button_state()
        self.notebook.select(self.tab_2d)
        if self.results_3d is not None:
            self._show_3d_plot_window()
            self.status_var.set("Loaded saved result with 2D and 3D data. Opened the 3D PyVista window.")
        else:
            self.status_var.set("Loaded saved result with 2D data.")

    def _collect_params(self):
        self._sync_dip_in_media()
        p = {"dip_in_enabled": bool(self.dip_in_enabled_var.get())}
        for spec in PARAM_SPECS:
            p[spec.key] = spec.si_value(self._param_var(spec).get())

        if p["nx"] < 8 or p["ny"] < 8:
            raise ValueError("Nx and Ny must be >= 8.")
        if p["size_x"] <= 0 or p["size_y"] <= 0:
            raise ValueError("Mask size must be positive.")
        if p["pupil_d"] <= 0:
            raise ValueError("Entrance pupil diameter must be positive.")
        if (not np.isfinite(p["fresnel_z"])) or p["fresnel_z"] < 0.0:
            raise ValueError("Fresnel propagation distance must be finite and nonnegative.")
        if (
            (not np.isfinite(p["range_x_um"]))
            or (not np.isfinite(p["range_y_um"]))
            or (not np.isfinite(p["range_z_um"]))
            or p["range_x_um"] <= 0.0
            or p["range_y_um"] <= 0.0
            or p["range_z_um"] <= 0.0
        ):
            raise ValueError("PSF ranges X/Y/Z must be finite and > 0.")
        if p["points_2d"] < 9 or p["points_3d"] < 9:
            raise ValueError("2D/3D points must be >= 9.")
        if p["z_chunk"] < 1:
            raise ValueError("3D z chunk must be >= 1.")
        if p["n_i"] <= 0 or p["n_g"] <= 0 or p["n_s"] <= 0:
            raise ValueError("Experimental RI values must be positive.")
        if p["n_id"] <= 0 or p["n_gd"] <= 0 or p["n_sd"] <= 0:
            raise ValueError("Design RI values must be positive.")
        if p["t_g"] < 0 or p["t_s"] < 0 or p["t_gd"] < 0 or not np.isfinite(p["t_id"]) or p["t_id"] < 0:
            raise ValueError("Thickness values must be non-negative.")
        if p["torch_device"] not in TORCH_DEVICES:
            raise ValueError("Torch device must be one of: auto, cpu, cuda.")
        if p["integration_mode"] not in INTEGRATION_MODES:
            raise ValueError("Integration mode must be cartesian_jacobian or polar_interp.")
        if p["aberration_model"] not in ABERRATION_MODELS:
            raise ValueError("Aberration model must be legacy or gibson_lanni.")
        if p["use_gaussian"] and (p["gauss_wx"] <= 0 or p["gauss_wy"] <= 0):
            raise ValueError("Gaussian FWHM x/y must be positive when Gaussian is enabled.")
        if not np.isfinite(p["mask_offset_x"]) or not np.isfinite(p["mask_offset_y"]):
            raise ValueError("Mask x/y offsets must be finite.")

        return p

    @staticmethod
    def _create_simulator_from_params(p):
        return create_simulator(p)

    def _get_input_mask(self, shape, size_x, size_y, offset_x=0.0, offset_y=0.0):
        if self.mask_array is None:
            base = np.ones(shape, dtype=float)
        else:
            base = np.asarray(self.mask_array, dtype=float)
            k = self._mask_rotation_quadrants_mod()
            if k != 0:
                base = np.rot90(base, k=k)
        # Always interpolate onto the simulation grid shape.
        resized = self._resize_mask(base, shape)
        return CartesianPSFSimulator.apply_mask_offset(
            resized,
            size_x=float(size_x),
            size_y=float(size_y),
            offset_x=float(offset_x),
            offset_y=float(offset_y),
        )

    def _collect_optimizer_settings(self):
        settings = {
            "optimizer_category": str(self.opt_optimizer_category_var.get()).strip(),
            "optimizer_name": str(self.opt_optimizer_var.get()).strip(),
            "mode": str(self.opt_mode_var.get()).strip().lower(),
            "axial_symmetry": bool(self.opt_axial_symmetry_var.get()),
            "binary_amplitude": bool(self.opt_binary_amplitude_var.get()),
            "binary_structure": str(self.opt_binary_structure_var.get()).strip().lower(),
            "width_level": float(self.opt_width_level_var.get()),
            "weight_x": float(self.opt_weight_x_var.get()),
            "weight_y": float(self.opt_weight_y_var.get()),
            "weight_z": float(self.opt_weight_z_var.get()),
            "weight_abs_xy": float(self.opt_weight_abs_xy_var.get()),
            "n_starts": int(self.opt_n_starts_var.get()),
            "maxiter": int(self.opt_maxiter_var.get()),
            "ftol": float(self.opt_ftol_var.get()),
            "gtol": float(self.opt_gtol_var.get()),
            "seed_text": str(self.opt_seed_var.get()).strip(),
        }

        category_map = self._get_optimizer_category_map()
        all_opt_names = self._all_optimizer_names(category_map)
        if settings["optimizer_name"] not in all_opt_names:
            raise ValueError(f"Optimizer must be one of: {', '.join(all_opt_names)}.")
        if settings["optimizer_category"] not in category_map:
            inferred = self._optimizer_category_for_name(settings["optimizer_name"], category_map)
            settings["optimizer_category"] = (
                inferred
                if inferred is not None
                else (next(iter(category_map.keys())) if len(category_map) > 0 else "")
            )
        if settings["optimizer_name"] not in tuple(category_map.get(settings["optimizer_category"], ())):
            inferred = self._optimizer_category_for_name(settings["optimizer_name"], category_map)
            if inferred is not None:
                settings["optimizer_category"] = inferred
        if settings["mode"] not in {"amplitude", "phase", "amplitude_phase"}:
            raise ValueError("Mode must be one of: amplitude, phase, amplitude_phase.")
        if settings["binary_structure"] not in {"pixel", "simple_ring_block", "simple_ellipse_block"}:
            raise ValueError(
                "Pass/block structure must be one of: pixel, simple_ring_block, simple_ellipse_block."
            )
        if settings["mode"] != "amplitude":
            settings["binary_amplitude"] = False
            settings["binary_structure"] = "pixel"
        if not settings["binary_amplitude"]:
            settings["binary_structure"] = "pixel"
        if settings["binary_structure"] == "simple_ring_block":
            settings["axial_symmetry"] = True
        elif settings["binary_structure"] == "simple_ellipse_block":
            settings["axial_symmetry"] = False
        if not np.isfinite(settings["width_level"]) or not (0.01 <= settings["width_level"] <= 0.99):
            raise ValueError("Width level must be in [0.01, 0.99].")
        if (
            settings["weight_x"] < 0
            or settings["weight_y"] < 0
            or settings["weight_z"] < 0
            or settings["weight_abs_xy"] < 0
        ):
            raise ValueError("X/Y/Z/abs(x-y) weights must be >= 0.")
        if (
            settings["weight_x"]
            + settings["weight_y"]
            + settings["weight_z"]
            + settings["weight_abs_xy"]
        ) <= 0:
            raise ValueError("At least one of X/Y/Z/abs(x-y) weights must be > 0.")
        if settings["n_starts"] < 1:
            raise ValueError("Multi-start count must be >= 1.")
        if settings["maxiter"] < 1:
            raise ValueError("Max iterations must be >= 1.")
        if (not np.isfinite(settings["ftol"])) or settings["ftol"] < 0:
            raise ValueError("ftol must be finite and >= 0 (0 disables this criterion).")
        if (not np.isfinite(settings["gtol"])) or settings["gtol"] < 0:
            raise ValueError("gtol must be finite and >= 0 (0 disables this criterion).")

        seed_text = settings["seed_text"]
        if seed_text == "":
            settings["seed"] = None
        else:
            try:
                settings["seed"] = int(seed_text)
            except Exception as exc:
                raise ValueError("Seed must be an integer or blank.") from exc

        return settings

    def _set_opt_running(self, running, status_text):
        self.opt_is_running = bool(running)
        self._update_dip_in_controls()
        run_state = "disabled" if self.opt_is_running else "normal"
        stop_state = "normal" if self.opt_is_running else "disabled"

        if hasattr(self, "opt_run_btn"):
            # Keep optimization disabled when a simulation run is active.
            effective_run_state = "disabled" if self.is_running else run_state
            self.opt_run_btn.configure(state=effective_run_state)
        if hasattr(self, "opt_stop_btn"):
            self.opt_stop_btn.configure(state=stop_state)
        if hasattr(self, "opt_save_btn"):
            save_state = (
                "normal"
                if ((not self.opt_is_running) and (self.opt_result is not None) and (not self.is_running))
                else "disabled"
            )
            self.opt_save_btn.configure(state=save_state)
        if hasattr(self, "opt_apply_btn"):
            apply_state = (
                "normal"
                if ((not self.opt_is_running) and (self.opt_result is not None) and (not self.is_running))
                else "disabled"
            )
            self.opt_apply_btn.configure(state=apply_state)
        if hasattr(self, "export_csv_btn"):
            self._update_export_csv_button_state()

        # Freeze simulation controls while optimization is in progress.
        if self.opt_is_running:
            self.run_2d_btn.configure(state="disabled")
            self.run_3d_btn.configure(state="disabled")
            self.stop_btn.configure(state="disabled")
            self.save_btn.configure(state="disabled")
            self.load_btn.configure(state="disabled")
            if hasattr(self, "save_config_btn"):
                self.save_config_btn.configure(state="disabled")
            if hasattr(self, "export_csv_btn"):
                self.export_csv_btn.configure(state="disabled")
            self.load_mask_btn.configure(state="disabled")
            if hasattr(self, "remove_mask_btn"):
                self.remove_mask_btn.configure(state="disabled")
            self.rotate_cw_btn.configure(state="disabled")
            self.rotate_ccw_btn.configure(state="disabled")
            if hasattr(self, "store_objective_lens_btn"):
                self.store_objective_lens_btn.configure(state="disabled")
            if hasattr(self, "load_objective_lens_btn"):
                self.load_objective_lens_btn.configure(state="disabled")
            for btn in self.mask_builder_action_buttons:
                btn.configure(state="disabled")
        else:
            if not self.is_running:
                self.run_2d_btn.configure(state="normal")
                self.run_3d_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                self.save_btn.configure(state="normal")
                self.load_btn.configure(state="normal")
                if hasattr(self, "save_config_btn"):
                    self.save_config_btn.configure(state="normal")
                if hasattr(self, "export_csv_btn"):
                    self._update_export_csv_button_state()
                self.load_mask_btn.configure(state="normal")
                if hasattr(self, "remove_mask_btn"):
                    self.remove_mask_btn.configure(state="normal")
                self.rotate_cw_btn.configure(state="normal")
                self.rotate_ccw_btn.configure(state="normal")
                if hasattr(self, "store_objective_lens_btn"):
                    self.store_objective_lens_btn.configure(state="normal")
                if hasattr(self, "load_objective_lens_btn"):
                    self.load_objective_lens_btn.configure(state="normal")
                for btn in self.mask_builder_action_buttons:
                    btn.configure(state="normal")

        if status_text is not None:
            self.opt_status_var.set(str(status_text))

    def _set_opt_progress(self, percent, status_text=None, force=False):
        pct = float(min(100.0, max(0.0, percent)))
        if self.opt_is_running and (not force):
            pct = max(float(self.opt_progress_var.get()), pct)
        self.opt_progress_var.set(pct)
        self.opt_progress_pct_var.set(f"{pct:.1f}%")
        if status_text is not None:
            self.opt_status_var.set(str(status_text))

    def _clear_opt_live_preview(self):
        if self._opt_live_refresh_job is not None:
            try:
                self.root.after_cancel(self._opt_live_refresh_job)
            except Exception:
                pass
        self._opt_live_refresh_job = None
        self._opt_live_preview_dirty = False
        self.opt_live_amp_preview = None
        self.opt_live_phase_preview = None
        self.opt_live_full_shape = None
        self.opt_live_objective = None
        self.opt_live_widths = None

    def _schedule_opt_live_preview_refresh(self, delay_ms=120):
        if self._opt_live_refresh_job is not None:
            return
        self._opt_live_refresh_job = self.root.after(
            int(delay_ms), self._on_opt_live_preview_refresh_timer
        )

    def _on_opt_live_preview_refresh_timer(self):
        self._opt_live_refresh_job = None
        if not self._opt_live_preview_dirty:
            return
        self._opt_live_preview_dirty = False
        self._refresh_optimization_plot()

    def _on_optimization_progress_ui(self, percent, status_text, info):
        self._set_opt_progress(percent, status_text)
        if not isinstance(info, dict):
            return

        obj = info.get("objective")
        if obj is not None and np.isfinite(float(obj)):
            self.opt_live_objective = float(obj)

        width_x = info.get("width_x_um")
        width_y = info.get("width_y_um")
        width_z = info.get("width_z_um")
        finite_x = width_x is not None and np.isfinite(float(width_x))
        finite_y = width_y is not None and np.isfinite(float(width_y))
        finite_z = width_z is not None and np.isfinite(float(width_z))
        if finite_x and finite_y:
            self.opt_live_widths = (
                float(width_x),
                float(width_y),
                float(width_z) if finite_z else np.nan,
            )
            abs_xy = info.get("abs_x_minus_y_um")
            if abs_xy is None or (not np.isfinite(float(abs_xy))):
                abs_xy = abs(float(width_x) - float(width_y))
            width_parts = [
                f"X={float(width_x):.4f}",
                f"Y={float(width_y):.4f}",
            ]
            if finite_z:
                width_parts.append(f"Z={float(width_z):.4f}")
            self.opt_summary_var.set(
                "Current (um): "
                + ", ".join(width_parts)
                + f", abs(x-y)={float(abs_xy):.4f}\n"
                f"Current objective: {float(self.opt_live_objective) if self.opt_live_objective is not None else np.nan:.6f}"
            )

        amp_preview = info.get("mask_amp_preview")
        phase_preview = info.get("mask_phase_preview")
        if amp_preview is not None:
            try:
                amp_arr = np.asarray(amp_preview, dtype=float)
                self.opt_live_amp_preview = np.clip(amp_arr, 0.0, 1.0)
            except Exception:
                pass
        if phase_preview is not None:
            try:
                phase_arr = np.asarray(phase_preview, dtype=float)
                self.opt_live_phase_preview = np.clip(phase_arr, -np.pi, np.pi)
            except Exception:
                pass
        full_shape = info.get("preview_shape_full")
        if (
            isinstance(full_shape, (list, tuple))
            and len(full_shape) == 2
            and int(full_shape[0]) > 0
            and int(full_shape[1]) > 0
        ):
            self.opt_live_full_shape = (int(full_shape[0]), int(full_shape[1]))

        if self.opt_live_amp_preview is not None:
            self._opt_live_preview_dirty = True
            self._schedule_opt_live_preview_refresh(delay_ms=120)

    def _start_optimization_worker(self):
        if not HAS_SCIPY_OPTIMIZER:
            messagebox.showerror(
                "Mask optimization",
                "SciPy optimizer backend is unavailable.\n\n"
                f"Import error: {SCIPY_OPTIMIZER_IMPORT_ERROR}",
            )
            return
        if self.is_running or self.opt_is_running:
            return

        try:
            params = self._collect_params()
            settings = self._collect_optimizer_settings()
            mask = self._get_input_mask(
                (params["nx"], params["ny"]),
                size_x=params["size_x"],
                size_y=params["size_y"],
                offset_x=params["mask_offset_x"],
                offset_y=params["mask_offset_y"],
            )
            multiplier = 2 if settings["mode"] == "amplitude_phase" else 1
            variables = int(params["nx"] * params["ny"] * multiplier)
            predicted = int(
                np.ceil(
                    PostFresnelMaskOptimizer._estimate_evals_per_iteration(
                        settings["optimizer_name"], variables
                    )
                )
            )
            estimate = estimate_resources(
                params,
                optimizer_variables=variables,
                predicted_objective_evals_per_iteration=predicted,
            )
            try:
                enforce_resource_limits(estimate, allow_expensive=False)
            except ValueError as exc:
                if "Request is expensive" not in str(exc):
                    raise
                if not messagebox.askyesno(
                    "Expensive optimization",
                    f"{exc}\n\nEstimated peak RAM: {estimate.peak_ram_bytes / 1024**3:.2f} GiB. Continue?",
                ):
                    return
                enforce_resource_limits(estimate, allow_expensive=True)
        except Exception as exc:
            messagebox.showerror("Mask optimization", str(exc))
            return

        self.opt_stop_requested = False
        self.opt_result = None
        self.opt_input_mask = np.asarray(mask, dtype=np.float64).copy()
        self._clear_opt_live_preview()
        self.opt_live_amp_preview = np.clip(np.asarray(mask, dtype=float), 0.0, 1.0)
        self.opt_live_phase_preview = np.zeros_like(self.opt_live_amp_preview, dtype=float)
        self.opt_live_full_shape = tuple(int(v) for v in self.opt_live_amp_preview.shape)
        self.opt_live_objective = None
        self.opt_live_widths = None
        self.opt_summary_var.set("Running optimization. Live mask preview updates below.")
        self._refresh_optimization_plot()
        self._set_opt_running(True, "Running mask optimization...")
        self._set_opt_progress(0.0, "Preparing optimization...", force=True)
        self.notebook.select(self.tab_opt)
        thread = threading.Thread(
            target=self._optimization_worker_logic,
            args=(params, settings, mask),
            daemon=True,
        )
        thread.start()

    def _stop_optimization(self):
        if not self.opt_is_running:
            return
        self.opt_stop_requested = True
        self._set_opt_progress(float(self.opt_progress_var.get()), "Stopping requested...")

    def _optimization_worker_logic(self, params, settings, mask):
        try:
            simulator = self._create_simulator_from_params(params)
            optimizer = PostFresnelMaskOptimizer(simulator=simulator, params=params)

            cfg = OptimizerConfig(
                optimizer_name=settings["optimizer_name"],
                mode=settings["mode"],
                axial_symmetry=bool(settings["axial_symmetry"]),
                binary_amplitude=bool(settings["binary_amplitude"]),
                binary_structure=str(settings.get("binary_structure", "pixel")).strip().lower(),
                width_level=float(settings["width_level"]),
                weights_xyz=(
                    float(settings["weight_x"]),
                    float(settings["weight_y"]),
                    float(settings["weight_z"]),
                ),
                weight_abs_xy=float(settings.get("weight_abs_xy", 0.0)),
                n_starts=int(settings["n_starts"]),
                maxiter=int(settings["maxiter"]),
                ftol=float(settings["ftol"]),
                gtol=float(settings["gtol"]),
                seed=settings["seed"],
            )

            stop_cb = lambda: self.opt_stop_requested

            def queue_progress(stage, frac, info):
                pct = 100.0 * float(min(1.0, max(0.0, frac)))
                text = str(stage)
                if isinstance(info, dict):
                    parts = [text]
                    obj = info.get("objective")
                    if obj is not None and np.isfinite(obj):
                        parts.append(f"objective={float(obj):.4f}")
                    abs_xy = info.get("abs_x_minus_y_um")
                    if abs_xy is not None and np.isfinite(float(abs_xy)):
                        parts.append(f"abs(x-y)={float(abs_xy):.4f}um")
                    side_pen = info.get("sidelobe_penalty")
                    if side_pen is not None and np.isfinite(float(side_pen)):
                        parts.append(f"sl_pen={float(side_pen):.4f}")
                    iter_index = info.get("iter_index")
                    maxiter_hint = info.get("maxiter")
                    if iter_index is not None and np.isfinite(float(iter_index)):
                        if maxiter_hint is not None and np.isfinite(float(maxiter_hint)) and float(maxiter_hint) > 0:
                            parts.append(f"iter={int(float(iter_index))}/{int(float(maxiter_hint))}")
                        else:
                            parts.append(f"iter={int(float(iter_index))}")
                    eval_in_start = info.get("eval_in_start")
                    if eval_in_start is not None and np.isfinite(float(eval_in_start)):
                        parts.append(f"eval={int(float(eval_in_start))}")
                    n_free = info.get("n_free_variables")
                    n_total = info.get("n_variables")
                    if (
                        n_free is not None
                        and np.isfinite(float(n_free))
                        and n_total is not None
                        and np.isfinite(float(n_total))
                    ):
                        parts.append(f"vars={int(float(n_free))}/{int(float(n_total))}")
                    est_evals_per_iter = info.get("est_evals_per_iter")
                    if est_evals_per_iter is not None and np.isfinite(float(est_evals_per_iter)):
                        parts.append(f"~{int(round(float(est_evals_per_iter)))} eval/iter")
                    est_iter_s = info.get("est_iter_s")
                    if est_iter_s is not None and np.isfinite(float(est_iter_s)):
                        est_iter_s = float(est_iter_s)
                        if est_iter_s >= 60.0:
                            parts.append(f"~{est_iter_s / 60.0:.1f} min/iter")
                        elif est_iter_s >= 1.0:
                            parts.append(f"~{est_iter_s:.1f} s/iter")
                    ellipse_in_dx_mm = info.get("ellipse_inner_diameter_x_mm")
                    ellipse_in_dy_mm = info.get("ellipse_inner_diameter_y_mm")
                    ellipse_out_dx_mm = info.get("ellipse_outer_diameter_x_mm")
                    ellipse_out_dy_mm = info.get("ellipse_outer_diameter_y_mm")
                    if (
                        ellipse_in_dx_mm is not None
                        and ellipse_in_dy_mm is not None
                        and ellipse_out_dx_mm is not None
                        and ellipse_out_dy_mm is not None
                        and np.isfinite(float(ellipse_in_dx_mm))
                        and np.isfinite(float(ellipse_in_dy_mm))
                        and np.isfinite(float(ellipse_out_dx_mm))
                        and np.isfinite(float(ellipse_out_dy_mm))
                    ):
                        parts.append(
                            "ellipse(mm)="
                            f"in({float(ellipse_in_dx_mm):.4f},{float(ellipse_in_dy_mm):.4f}) "
                            f"out({float(ellipse_out_dx_mm):.4f},{float(ellipse_out_dy_mm):.4f})"
                        )
                    else:
                        ellipse_in_dx = info.get("ellipse_inner_diameter_x_norm")
                        ellipse_in_dy = info.get("ellipse_inner_diameter_y_norm")
                        ellipse_out_dx = info.get("ellipse_outer_diameter_x_norm")
                        ellipse_out_dy = info.get("ellipse_outer_diameter_y_norm")
                        if (
                            ellipse_in_dx is not None
                            and ellipse_in_dy is not None
                            and ellipse_out_dx is not None
                            and ellipse_out_dy is not None
                            and np.isfinite(float(ellipse_in_dx))
                            and np.isfinite(float(ellipse_in_dy))
                            and np.isfinite(float(ellipse_out_dx))
                            and np.isfinite(float(ellipse_out_dy))
                        ):
                            try:
                                pd_mm = float(params["pupil_d"]) * 1e3
                                parts.append(
                                    "ellipse(mm)="
                                    f"in({float(ellipse_in_dx)*pd_mm:.4f},{float(ellipse_in_dy)*pd_mm:.4f}) "
                                    f"out({float(ellipse_out_dx)*pd_mm:.4f},{float(ellipse_out_dy)*pd_mm:.4f})"
                                )
                            except Exception:
                                parts.append(
                                    "ellipse(norm)="
                                    f"in({float(ellipse_in_dx):.3f},{float(ellipse_in_dy):.3f}) "
                                    f"out({float(ellipse_out_dx):.3f},{float(ellipse_out_dy):.3f})"
                                )
                    ring_in_mm = info.get("ring_inner_radius_mm")
                    ring_out_mm = info.get("ring_outer_radius_mm")
                    if (
                        ring_in_mm is not None
                        and ring_out_mm is not None
                        and np.isfinite(float(ring_in_mm))
                        and np.isfinite(float(ring_out_mm))
                    ):
                        parts.append(
                            f"ring(mm)=[{float(ring_in_mm):.4f},{float(ring_out_mm):.4f}]"
                        )
                    else:
                        ring_in_m = info.get("ring_inner_radius_m")
                        ring_out_m = info.get("ring_outer_radius_m")
                        if (
                            ring_in_m is not None
                            and ring_out_m is not None
                            and np.isfinite(float(ring_in_m))
                            and np.isfinite(float(ring_out_m))
                        ):
                            parts.append(
                                f"ring(mm)=[{float(ring_in_m)*1e3:.4f},{float(ring_out_m)*1e3:.4f}]"
                            )
                        else:
                            ring_in = info.get("ring_inner_radius_norm")
                            ring_out = info.get("ring_outer_radius_norm")
                            if (
                                ring_in is not None
                                and ring_out is not None
                                and np.isfinite(float(ring_in))
                                and np.isfinite(float(ring_out))
                            ):
                                try:
                                    pr_mm = 0.5 * float(params["pupil_d"]) * 1e3
                                    parts.append(
                                        f"ring(mm)=[{float(ring_in)*pr_mm:.4f},{float(ring_out)*pr_mm:.4f}]"
                                    )
                                except Exception:
                                    parts.append(
                                        f"ring(norm)=[{float(ring_in):.3f},{float(ring_out):.3f}]"
                                    )
                    text = " | ".join(parts)
                self.root.after(
                    0,
                    lambda p=pct, t=text, i=info: self._on_optimization_progress_ui(
                        p, t, i
                    ),
                )

            result = optimizer.optimize(
                input_intensity=mask,
                config=cfg,
                progress_callback=queue_progress,
                stop_callback=stop_cb,
            )

            self.root.after(
                0, lambda p=params, r=result: self._on_optimization_success(p, r)
            )
        except OptimizationStoppedError:
            self.root.after(0, self._on_optimization_stopped)
        except Exception as exc:
            detail = f"{exc}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda d=detail: self._on_optimization_error(d))

    def _on_optimization_success(self, params, result):
        self.opt_stop_requested = False
        self._clear_opt_live_preview()
        self.opt_result = result
        self.opt_last_params = self._clone_payload(params)

        formatted = self._format_optimizer_summary(
            result.baseline_metrics_um,
            result.best_metrics_um,
            result.best_objective,
            settings=result.settings,
            params=params,
        )
        if formatted is not None:
            self.opt_summary_var.set(formatted)
        self._refresh_optimization_plot()
        converged = str(getattr(result, "status", "converged")) == "converged"
        final_text = "Optimization completed." if converged else "Optimization budget exhausted; best exact candidate retained."
        self._set_opt_progress(100.0, final_text, force=True)
        self._set_opt_running(False, final_text)
        self.status_var.set(final_text)
        self._save_last_used_config(params)
        self.notebook.select(self.tab_opt)

    def _on_optimization_error(self, detail):
        self.opt_stop_requested = False
        self._clear_opt_live_preview()
        self._set_opt_progress(0.0, "Optimization failed.", force=True)
        self._set_opt_running(False, "Optimization failed.")
        self._refresh_optimization_plot()
        messagebox.showerror("Mask optimization error", detail)

    def _on_optimization_stopped(self):
        self.opt_stop_requested = False
        self._clear_opt_live_preview()
        self._set_opt_progress(float(self.opt_progress_var.get()), "Optimization stopped.", force=True)
        self._set_opt_running(False, "Optimization stopped.")
        self.status_var.set("Mask optimization stopped by user.")
        self._refresh_optimization_plot()

    def _save_optimized_mask_dialog(self):
        if save_result_npz is None:
            messagebox.showerror(
                "Save optimized mask",
                "SciPy optimizer backend is unavailable. Cannot export optimized mask.",
            )
            return
        if self.opt_result is None:
            messagebox.showwarning("No optimization result", "Run mask optimization first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save optimized mask",
            initialdir=self.optimized_mask_dir,
            defaultextension=".npz",
            filetypes=[
                ("NumPy compressed", "*.npz"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        out_path = path if os.path.splitext(path)[1].lower() == ".npz" else f"{path}.npz"
        try:
            extra = {"simulation_params": self._clone_payload(self.opt_last_params)}
            save_result_npz(out_path, self.opt_result, extra_metadata=extra)
            self.opt_status_var.set(f"Saved optimized mask: {out_path}")
            self.status_var.set(f"Saved optimized mask: {out_path}")
        except Exception as exc:
            messagebox.showerror("Save optimized mask", str(exc))

    def _apply_optimized_mask_to_simulation(self):
        if self.opt_result is None:
            messagebox.showwarning("No optimization result", "Run mask optimization first.")
            return
        complex_mask = np.asarray(self.opt_result.complex_mask, dtype=np.complex128)
        if self.opt_input_mask is None:
            self.mask_array = np.ones(complex_mask.shape, dtype=np.float64)
        else:
            self.mask_array = np.asarray(self.opt_input_mask, dtype=np.float64).copy()
        self.post_fresnel_mask_array = complex_mask.copy()
        self.mask_rotation_quadrants = 0
        self._update_mask_orientation_label()
        self.mask_path = None
        self.mask_path_var.set("From Mask Optimization")
        self.nx_var.set(int(complex_mask.shape[0]))
        self.ny_var.set(int(complex_mask.shape[1]))
        self.mask_info_var.set(
            "Optimized post-Fresnel complex-amplitude mask: "
            f"shape={complex_mask.shape}, amplitude=[{float(np.abs(complex_mask).min()):.3f}, "
            f"{float(np.abs(complex_mask).max()):.3f}]"
        )
        self._refresh_mask_input_preview()
        self.opt_status_var.set("Applied optimized complex mask after Fresnel propagation.")
        self.status_var.set("Applied optimized complex mask after Fresnel propagation.")

    def _refresh_optimization_plot(self):
        if not hasattr(self, "fig_opt") or not hasattr(self, "canvas_opt"):
            return
        self.fig_opt.clear()

        if self.opt_result is None:
            if self.opt_live_amp_preview is not None:
                amp_live = np.clip(np.asarray(self.opt_live_amp_preview, dtype=float), 0.0, 1.0)
                if self.opt_live_phase_preview is None:
                    phase_live = np.zeros_like(amp_live, dtype=float)
                else:
                    phase_live = np.clip(
                        np.asarray(self.opt_live_phase_preview, dtype=float),
                        -np.pi,
                        np.pi,
                    )
                mag2_live = amp_live * amp_live

                axes = self.fig_opt.subplots(2, 2)
                im0 = axes[0, 0].imshow(
                    amp_live.T,
                    origin="lower",
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    aspect="equal",
                    interpolation="nearest",
                )
                style_image_axis(axes[0, 0], title="Amplitude", xlabel="x (px)", ylabel="y (px)")
                add_compact_colorbar(self.fig_opt, im0, axes[0, 0], "Amplitude")

                im1 = axes[0, 1].imshow(
                    phase_live.T,
                    origin="lower",
                    cmap="twilight",
                    vmin=-np.pi,
                    vmax=np.pi,
                    aspect="equal",
                    interpolation="nearest",
                )
                style_image_axis(axes[0, 1], title="Phase", xlabel="x (px)", ylabel="y (px)")
                add_compact_colorbar(
                    self.fig_opt,
                    im1,
                    axes[0, 1],
                    "Phase (rad)",
                    ticks=[-np.pi, 0.0, np.pi],
                )

                im2 = axes[1, 0].imshow(
                    mag2_live.T,
                    origin="lower",
                    cmap="magma",
                    norm=intensity_norm(mag2_live, vmin=0.0),
                    aspect="equal",
                    interpolation="nearest",
                )
                style_image_axis(
                    axes[1, 0],
                    title="|M|^2",
                    xlabel="x (px)",
                    ylabel="y (px)",
                )
                add_compact_colorbar(self.fig_opt, im2, axes[1, 0], "Magnitude squared")

                live_lines = ["Live optimization preview"]
                if self.opt_live_full_shape is not None:
                    full_nx, full_ny = self.opt_live_full_shape
                    live_lines.append(f"Full mask: {int(full_nx)} x {int(full_ny)} px")
                    live_lines.append(
                        f"Preview: {int(amp_live.shape[0])} x {int(amp_live.shape[1])} px"
                    )
                if self.opt_live_objective is not None and np.isfinite(self.opt_live_objective):
                    live_lines.append(f"Objective: {float(self.opt_live_objective):.6f}")
                if self.opt_live_widths is not None:
                    wx, wy, wz = self.opt_live_widths
                    width_line = f"Widths (um): X={float(wx):.4f}, Y={float(wy):.4f}"
                    if np.isfinite(float(wz)):
                        width_line += f", Z={float(wz):.4f}"
                    live_lines.append(width_line)
                    live_lines.append(f"|X-Y| = {abs(float(wx) - float(wy)):.4f} um")
                if self.opt_is_running:
                    live_lines.append("Status: optimizing...")
                self._summary_text_artist(axes[1, 1], "\n".join(live_lines))
                self.canvas_opt.draw()
                return

            ax = self.fig_opt.add_subplot(111)
            if not HAS_SCIPY_OPTIMIZER:
                msg = (
                    "Mask optimization backend unavailable.\n\n"
                    f"Import error:\n{SCIPY_OPTIMIZER_IMPORT_ERROR}"
                )
            else:
                msg = "Run mask optimization to preview amplitude/phase results."
            ax.text(
                0.5,
                0.5,
                msg,
                ha="center",
                va="center",
            )
            ax.set_title("Mask Optimization Preview")
            ax.axis("off")
            self.canvas_opt.draw()
            return

        amp = np.asarray(self.opt_result.amplitude_mask, dtype=float)
        phase = np.asarray(self.opt_result.phase_mask_rad, dtype=float)
        mag2 = np.abs(np.asarray(self.opt_result.complex_mask, dtype=np.complex128)) ** 2

        axes = self.fig_opt.subplots(2, 2)
        im0 = axes[0, 0].imshow(
            amp.T,
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="nearest",
        )
        style_image_axis(axes[0, 0], title="Amplitude", xlabel="x (px)", ylabel="y (px)")
        add_compact_colorbar(self.fig_opt, im0, axes[0, 0], "Amplitude")

        im1 = axes[0, 1].imshow(
            phase.T,
            origin="lower",
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            aspect="equal",
            interpolation="nearest",
        )
        style_image_axis(axes[0, 1], title="Phase", xlabel="x (px)", ylabel="y (px)")
        add_compact_colorbar(
            self.fig_opt,
            im1,
            axes[0, 1],
            "Phase (rad)",
            ticks=[-np.pi, 0.0, np.pi],
        )

        im2 = axes[1, 0].imshow(
            mag2.T,
            origin="lower",
            cmap="magma",
            norm=intensity_norm(mag2, vmin=0.0),
            aspect="equal",
            interpolation="nearest",
        )
        style_image_axis(axes[1, 0], title="|M|^2", xlabel="x (px)", ylabel="y (px)")
        add_compact_colorbar(self.fig_opt, im2, axes[1, 0], "Magnitude squared")

        before = self.opt_result.baseline_metrics_um
        after = self.opt_result.best_metrics_um
        summary_lines = [
            "Width summary (um)",
            f"X: {before['width_x_um']:.4f} -> {after['width_x_um']:.4f}",
            f"Y: {before['width_y_um']:.4f} -> {after['width_y_um']:.4f}",
            f"Z: {before['width_z_um']:.4f} -> {after['width_z_um']:.4f}",
            "",
            f"|X-Y|: {abs(float(before['width_x_um']) - float(before['width_y_um'])):.4f}"
            f" -> {abs(float(after['width_x_um']) - float(after['width_y_um'])):.4f}",
            "",
            f"Objective: {self.opt_result.best_objective:.6f}",
            f"Mode: {self.opt_result.settings.get('mode', '-')}",
            f"Optimizer: {self.opt_result.settings.get('optimizer_name', '-')}",
        ]
        try:
            bstruct = str(self.opt_result.settings.get("binary_structure", "")).strip().lower()
            rin_mm = self.opt_result.settings.get("ring_inner_radius_mm")
            rout_mm = self.opt_result.settings.get("ring_outer_radius_mm")
            if (
                bstruct == "simple_ring_block"
                and rin_mm is not None
                and rout_mm is not None
                and np.isfinite(float(rin_mm))
                and np.isfinite(float(rout_mm))
            ):
                summary_lines.append(f"Ring block (mm): {float(rin_mm):.4f} -> {float(rout_mm):.4f}")
            elif bstruct == "simple_ellipse_block":
                einx_mm = self.opt_result.settings.get("ellipse_inner_diameter_x_mm")
                einy_mm = self.opt_result.settings.get("ellipse_inner_diameter_y_mm")
                eoutx_mm = self.opt_result.settings.get("ellipse_outer_diameter_x_mm")
                eouty_mm = self.opt_result.settings.get("ellipse_outer_diameter_y_mm")
                if (
                    einx_mm is not None
                    and einy_mm is not None
                    and eoutx_mm is not None
                    and eouty_mm is not None
                    and np.isfinite(float(einx_mm))
                    and np.isfinite(float(einy_mm))
                    and np.isfinite(float(eoutx_mm))
                    and np.isfinite(float(eouty_mm))
                ):
                    summary_lines.append(
                        "Ellipse ring (mm): "
                        f"in=({float(einx_mm):.4f}, {float(einy_mm):.4f}) "
                        f"out=({float(eoutx_mm):.4f}, {float(eouty_mm):.4f})"
                    )
        except Exception:
            pass
        self._summary_text_artist(axes[1, 1], "\n".join(summary_lines))
        self.canvas_opt.draw()

    def _run_2d(self):
        self._start_worker(run_3d=False)

    def _run_3d(self):
        self._start_worker(run_3d=True)

    def _start_worker(self, run_3d):
        if self.is_running or self.opt_is_running:
            return
        try:
            params = self._collect_params()
            estimate = estimate_resources(params, run_3d=bool(run_3d))
            try:
                enforce_resource_limits(estimate, allow_expensive=False)
            except ValueError as exc:
                if "Request is expensive" not in str(exc):
                    raise
                if not messagebox.askyesno(
                    "Expensive simulation",
                    f"{exc}\n\nEstimated peak RAM: {estimate.peak_ram_bytes / 1024**3:.2f} GiB. Continue?",
                ):
                    return
                enforce_resource_limits(estimate, allow_expensive=True)
            # Snapshot everything the worker needs here: Tk variables and the
            # loaded masks must not be read or written from the worker thread.
            mask = self._get_input_mask(
                (params["nx"], params["ny"]),
                size_x=params["size_x"],
                size_y=params["size_y"],
                offset_x=params["mask_offset_x"],
                offset_y=params["mask_offset_y"],
            )
        except Exception as exc:
            messagebox.showerror("Resource preflight", str(exc))
            return
        self.stop_requested = False
        self._set_running(True, "Running simulation...")
        self._set_progress(0.0, "Initializing...", force=True)
        thread = threading.Thread(
            target=self._worker_logic,
            args=(run_3d, params, mask, self.post_fresnel_mask_array),
            daemon=True,
        )
        thread.start()

    def _stop_simulation(self):
        if not self.is_running:
            return
        self.stop_requested = True
        self._set_progress(float(self.progress_var.get()), "Stopping requested...")

    def _worker_logic(self, run_3d, p, mask, post_fresnel_mask):
        try:
            simulator = self._create_simulator_from_params(p)
            _simulator, res2d, res3d = run_simulation(
                p,
                mask,
                run_3d=run_3d,
                progress=_QueuedProgress(self.root, self._set_progress),
                post_fresnel_mask=post_fresnel_mask,
                simulator=simulator,
                stop_callback=lambda: self.stop_requested,
            )
            self.root.after(0, lambda: self._on_run_success(simulator, p, res2d, res3d, run_3d))
        except SimulationStoppedError:
            self.root.after(0, self._on_run_stopped)
        except Exception as exc:
            detail = f"{exc}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._on_run_error(detail))

    def _on_run_success(self, simulator, params, res2d, res3d, ran_3d):
        self.stop_requested = False
        self.last_simulator = simulator
        self.last_params = params
        self.results_2d = res2d
        self.results_3d = res3d if ran_3d else None
        self._save_last_used_config(params)
        self._clear_2d_width_cache()
        self._clear_3d_render_cache()
        self._refresh_plots()
        self._update_3d_window_button_state()
        self.notebook.select(self.tab_2d)
        if ran_3d:
            self._show_3d_plot_window()
        dx_um = (params["size_x"] / params["nx"]) * 1e6
        dy_um = (params["size_y"] / params["ny"]) * 1e6
        trim_note = ""
        try:
            if bool(res2d.get("pupil_trimmed_for_psf", False)):
                shp = np.asarray(res2d.get("field_at_pupil_for_psf")).shape
                trim_note = f", PSF grid trimmed to {int(shp[0])}x{int(shp[1])}"
        except Exception:
            trim_note = ""
        self._set_progress(100.0)
        self._set_running(
            False,
            (
                f"Done. Fresnel grid {params['nx']}x{params['ny']} "
                f"(dx={dx_um:.3f} um, dy={dy_um:.3f} um){trim_note}."
            ),
        )

    def _on_run_error(self, detail):
        self.stop_requested = False
        self._set_progress(0.0, force=True)
        self._set_running(False, "Simulation failed.")
        messagebox.showerror("Simulation error", detail)

    def _on_run_stopped(self):
        self.stop_requested = False
        self._set_progress(float(self.progress_var.get()), force=True)
        self._set_running(False, "Stopped by user.")

    def _power(self):
        return 2 if self.plot_mode_var.get() == "I^2" else 1

    def _display_intensity(self, arr, apply_power=True):
        arr = np.asarray(arr, dtype=float)
        if bool(apply_power):
            data = arr ** self._power()
        else:
            data = arr
        if bool(self.normalize_var.get()):
            m = float(np.max(data))
            return data / m if m > 0 else data
        return data

    def _display_intensity_label(self, apply_power=True):
        if bool(apply_power) and self._power() == 2:
            return "Normalized 2-photon signal" if bool(self.normalize_var.get()) else "2-photon signal (a.u.)"
        return "Normalized intensity" if bool(self.normalize_var.get()) else "Intensity (a.u.)"

    @staticmethod
    def _set_image_extent(ax, xlim, ylim):
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
        ax.set_aspect("equal", adjustable="box", anchor="C")

    @staticmethod
    def _summary_text_artist(ax, text):
        ax.axis("off")
        ax.text(
            0.0,
            1.0,
            text,
            ha="left",
            va="top",
            fontsize=6,
            linespacing=1.35,
            transform=ax.transAxes,
        )

    @staticmethod
    def _peak_scaled_level(profile, level):
        prof = np.asarray(profile, dtype=float)
        pmax = float(np.max(prof)) if prof.size else 0.0
        if pmax <= 0:
            return float(level)
        return float(level) * pmax

    @staticmethod
    def _format_width_ratio(value):
        return f"{float(value):.3f}" if np.isfinite(value) else "N/A"

    @staticmethod
    def _format_width_value(value):
        return f"{float(value):.3f}" if np.isfinite(value) else "N/A"

    @staticmethod
    def _format_width_level(level):
        text = f"{float(level):.6f}".rstrip("0").rstrip(".")
        return text if text != "" else "0"

    def _quantize_width_level(self, level):
        level = min(0.99, max(0.01, float(level)))
        step = float(self._width_level_step)
        return round(round(level / step) * step, 3)

    def _clear_2d_width_cache(self):
        self._width_metrics_cache = {}
        self._width_plot_artists = {}

    def _current_width_level(self):
        try:
            level = float(self._active_width_level)
        except Exception:
            level = 0.50
        return min(0.99, max(0.01, level))

    def _set_width_level_controls(self, active_level, *, slider_level=None, update_entry=True):
        active_level = min(0.99, max(0.01, float(active_level)))
        if slider_level is None:
            slider_level = self._quantize_width_level(active_level)
        self.width_level_slider_var.set(float(slider_level))
        if update_entry:
            self.width_level_var.set(self._format_width_level(active_level))
        self.width_level_label.configure(text=self._format_width_level(active_level))

    def _cancel_width_entry_job(self):
        if self._width_entry_job is None:
            return
        try:
            self.root.after_cancel(self._width_entry_job)
        except Exception:
            pass
        self._width_entry_job = None

    def _parse_width_level_text(self, *, clamp=False, require_in_range=False):
        text = str(self.width_level_var.get()).strip()
        if text == "":
            return None
        try:
            level = float(text)
        except Exception:
            return None
        if not np.isfinite(level):
            return None
        if require_in_range and not (0.01 <= level <= 0.99):
            return None
        if clamp:
            level = min(0.99, max(0.01, level))
        return float(level)

    def _apply_width_level(self, level, *, source, update_entry):
        clamped_level = min(0.99, max(0.01, float(level)))
        slider_level = self._quantize_width_level(clamped_level)
        self._width_level_source = str(source)
        self._active_width_level = slider_level if source == "slider" else clamped_level
        self._set_width_level_controls(
            self._active_width_level,
            slider_level=slider_level,
            update_entry=update_entry,
        )
        self._refresh_width_plots_only(
            level=self._active_width_level,
            exact=(self._width_level_source == "entry"),
        )

    def _current_width_metrics_cache_key(self):
        if self.results_2d is None:
            return None
        return (id(self.results_2d), self._power(), int(bool(self.normalize_var.get())))

    def _get_or_build_width_metrics_cache(self):
        key = self._current_width_metrics_cache_key()
        if key is None or self.last_simulator is None:
            return None
        cached = self._width_metrics_cache.get(key)
        if cached is not None:
            return cached

        r = self.results_2d
        i_xy = self._display_intensity(self._choose_result_array(r, "I_xy", "I_xy_norm"))
        i_xz = self._display_intensity(self._choose_result_array(r, "I_xz", "I_xz_norm"))
        i_yz = self._display_intensity(self._choose_result_array(r, "I_yz", "I_yz_norm"))
        cached = self.last_simulator.precompute_peak_width_metrics(
            x_coords=r["x_coords"],
            y_coords=r["y_coords"],
            z_coords=r["z_coords"],
            i_xy_norm=i_xy,
            i_xz_norm=i_xz,
            i_yz_norm=i_yz,
            level_min=0.01,
            level_max=0.99,
            level_step=self._width_level_step,
        )
        self._width_metrics_cache[key] = cached
        return cached

    def _resolve_width_metrics_and_profiles(self, level=None, exact=None):
        if self.results_2d is None or self.last_simulator is None:
            return None, None, None, None

        if level is None:
            level = self._current_width_level()
        else:
            level = min(0.99, max(0.01, float(level)))
        if exact is None:
            exact = bool(self._width_level_source == "entry")

        width_cache = self._get_or_build_width_metrics_cache()
        if width_cache is None:
            r = self.results_2d
            i_xy = self._display_intensity(self._choose_result_array(r, "I_xy", "I_xy_norm"))
            i_xz = self._display_intensity(self._choose_result_array(r, "I_xz", "I_xz_norm"))
            i_yz = self._display_intensity(self._choose_result_array(r, "I_yz", "I_yz_norm"))
            width_metrics = self.last_simulator.compute_peak_width_metrics(
                x_coords=r["x_coords"],
                y_coords=r["y_coords"],
                z_coords=r["z_coords"],
                i_xy_norm=i_xy,
                i_xz_norm=i_xz,
                i_yz_norm=i_yz,
                level=level,
            )
            return (
                width_metrics,
                np.asarray(width_metrics["x_profile_peak"], dtype=float),
                np.asarray(width_metrics["y_profile_peak"], dtype=float),
                np.asarray(width_metrics["z_profile_peak"], dtype=float),
            )

        if exact:
            width_metrics = self.last_simulator.peak_width_metrics_from_profiles(
                x_coords=self.results_2d["x_coords"],
                y_coords=self.results_2d["y_coords"],
                z_coords=self.results_2d["z_coords"],
                x_profile=width_cache["x_profile_peak"],
                y_profile=width_cache["y_profile_peak"],
                z_profile=width_cache["z_profile_peak"],
                level=level,
                peak_x_index=width_cache["peak_x_index"],
                peak_y_index=width_cache["peak_y_index"],
            )
        else:
            width_metrics = self.last_simulator.peak_width_metrics_from_precomputed(width_cache, level)

        return (
            width_metrics,
            np.asarray(width_cache["x_profile_peak"], dtype=float),
            np.asarray(width_cache["y_profile_peak"], dtype=float),
            np.asarray(width_cache["z_profile_peak"], dtype=float),
        )

    @staticmethod
    def _format_profile_axis_title(axis_name):
        return str(axis_name).strip().upper()

    def _update_width_ratio_display(self, width_metrics=None):
        mode = self.plot_mode_var.get()
        focus_z_um = np.nan
        if isinstance(self.results_2d, dict) and "z_focus" in self.results_2d:
            try:
                focus_z_um = float(self.results_2d["z_focus"]) * 1e6
            except Exception:
                focus_z_um = np.nan
        if not isinstance(width_metrics, dict):
            level_text = self._format_width_level(self._current_width_level())
            self.width_ratio_var.set(
                f"{mode} widths @{level_text}: X = N/A, Y = N/A, Z = N/A um | "
                f"z/x = N/A, z/y = N/A"
            )
            return
        level = float(width_metrics.get("level", self._current_width_level()))
        level_text = self._format_width_level(level)
        width_x = self._format_width_value(width_metrics.get("width_x_um", np.nan))
        width_y = self._format_width_value(width_metrics.get("width_y_um", np.nan))
        width_z = self._format_width_value(width_metrics.get("width_z_um", np.nan))
        ratio_zx = self._format_width_ratio(width_metrics.get("width_ratio_zx", np.nan))
        ratio_zy = self._format_width_ratio(width_metrics.get("width_ratio_zy", np.nan))
        focus_text = (
            f" | focus z = {focus_z_um:.3f} um" if np.isfinite(focus_z_um) else ""
        )
        self.width_ratio_var.set(
            f"{mode} widths @{level_text}: X = {width_x}, Y = {width_y}, Z = {width_z} um\n"
            f"z/x = {ratio_zx}, z/y = {ratio_zy}{focus_text}"
        )

    def _update_width_plot_artists(self, plot_state, width_metrics, x_prof, y_prof, z_prof):
        plot_state["x_level_line"].set_ydata(
            [self._peak_scaled_level(x_prof, width_metrics["level"])] * 2
        )
        plot_state["y_level_line"].set_ydata(
            [self._peak_scaled_level(y_prof, width_metrics["level"])] * 2
        )
        plot_state["z_level_line"].set_ydata(
            [self._peak_scaled_level(z_prof, width_metrics["level"])] * 2
        )
        plot_state["x_axis"].set_title(self._format_profile_axis_title("X"))
        plot_state["y_axis"].set_title(self._format_profile_axis_title("Y"))
        plot_state["z_axis"].set_title(self._format_profile_axis_title("Z"))

    def _refresh_width_plots_only(self, level=None, exact=None):
        if self.results_2d is None or self.last_simulator is None:
            self._update_width_ratio_display()
            return

        plot_state = self._width_plot_artists
        cache_key = self._current_width_metrics_cache_key()
        if level is None:
            level = self._current_width_level()
        else:
            level = min(0.99, max(0.01, float(level)))
        if exact is None:
            exact = bool(self._width_level_source == "entry")
        if (not plot_state) or plot_state.get("cache_key") != cache_key:
            self._refresh_2d_plots()
            return

        width_metrics, x_prof, y_prof, z_prof = self._resolve_width_metrics_and_profiles(
            level=level,
            exact=exact,
        )
        if width_metrics is None:
            self._update_width_ratio_display()
            return
        self._update_width_ratio_display(width_metrics)
        self._update_width_plot_artists(plot_state, width_metrics, x_prof, y_prof, z_prof)
        self.canvas_2d.draw_idle()

    @staticmethod
    def _choose_result_array(result_dict, raw_key, norm_key):
        if raw_key in result_dict and result_dict[raw_key] is not None:
            return np.asarray(result_dict[raw_key], dtype=float)
        return np.asarray(result_dict[norm_key], dtype=float)

    @staticmethod
    def _nearest_index(coords, value):
        arr = np.asarray(coords, dtype=float).ravel()
        if arr.size == 0:
            return 0
        return int(np.argmin(np.abs(arr - float(value))))

    @staticmethod
    def _centered_axis_limits(coords, default_half_span=1.0):
        arr = np.asarray(coords, dtype=float).ravel()
        if arr.size == 0:
            span = max(1e-12, float(default_half_span))
            return -span, span
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            span = max(1e-12, float(default_half_span))
            return -span, span
        half_span = max(abs(float(np.min(finite))), abs(float(np.max(finite))))
        if (not np.isfinite(half_span)) or half_span <= 0.0:
            half_span = max(1e-12, float(default_half_span))
        return -half_span, half_span

    def _on_profile_cursor_toggle(self):
        self._profile_drag_ax = None
        if not bool(self.profile_cursor_enabled_var.get()):
            self._clear_profile_cursor_artists()
            if self.results_2d is not None:
                self.cursor_info_var.set("Profile cursor disabled.")
            self.canvas_2d.draw_idle()
            return
        self._rebuild_profile_cursor_artists()
        self.canvas_2d.draw_idle()

    def _clear_profile_cursor_artists(self):
        for artists in list(self._profile_cursor_artists.values()):
            for key in ("marker", "vline", "text"):
                obj = artists.get(key)
                if obj is None:
                    continue
                try:
                    obj.remove()
                except Exception:
                    pass
        self._profile_cursor_artists = {}

    def _rebuild_profile_cursor_artists(self):
        self._clear_profile_cursor_artists()
        if not bool(self.profile_cursor_enabled_var.get()):
            return
        if not self._profile_axes_map:
            return

        for ax, info in self._profile_axes_map.items():
            x = np.asarray(info["x_coords"], dtype=float)
            y = np.asarray(info["profile"], dtype=float)
            if x.size == 0 or y.size == 0:
                continue
            idx = int(np.argmax(y))
            x0 = float(x[idx])
            y0 = float(y[idx])
            marker, = ax.plot(
                [x0], [y0], marker="o", markersize=6, color="cyan", markeredgecolor="black", zorder=8
            )
            vline = ax.axvline(x0, color="cyan", linestyle=":", linewidth=1.0, alpha=0.9)
            text = ax.annotate(
                "",
                xy=(x0, y0),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                color="cyan",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55),
            )
            self._profile_cursor_artists[ax] = {
                "idx": idx,
                "marker": marker,
                "vline": vline,
                "text": text,
            }
            self._set_profile_cursor_from_index(ax, idx, update_canvas=False)

    def _set_profile_cursor_from_x(self, ax, x_value, update_canvas=True):
        info = self._profile_axes_map.get(ax)
        if info is None:
            return
        idx = self._nearest_index(info["x_coords"], x_value)
        self._set_profile_cursor_from_index(ax, idx, update_canvas=update_canvas)

    def _set_profile_cursor_from_index(self, ax, idx, update_canvas=True):
        info = self._profile_axes_map.get(ax)
        artists = self._profile_cursor_artists.get(ax)
        if info is None or artists is None:
            return
        x = np.asarray(info["x_coords"], dtype=float)
        y = np.asarray(info["profile"], dtype=float)
        if x.size == 0 or y.size == 0:
            return
        idx_i = int(min(max(0, idx), x.size - 1))
        xv = float(x[idx_i])
        yv = float(y[idx_i])

        artists["idx"] = idx_i
        artists["marker"].set_data([xv], [yv])
        artists["vline"].set_xdata([xv, xv])
        artists["text"].xy = (xv, yv)
        artists["text"].set_text(f"{info['x_label']}={xv:.4f} um\nI={yv:.6g}")

        self.cursor_info_var.set(
            f"Profile cursor {info['name']}: {info['x_label']}={xv:.4f} um, I={yv:.6g}, idx={idx_i}"
        )
        if update_canvas:
            self.canvas_2d.draw_idle()

    def _on_2d_canvas_button_press(self, event):
        button = getattr(event, "button", None)
        if button in (2, 3):
            self._show_2d_axis_context_menu(event)
            return

        if not hasattr(self, "profile_cursor_enabled_var"):
            return
        if not bool(self.profile_cursor_enabled_var.get()):
            return
        if button not in (1, None):
            return
        ax = getattr(event, "inaxes", None)
        if ax is None or (ax not in self._profile_axes_map):
            return
        if event.xdata is None:
            return
        self._profile_drag_ax = ax
        self._set_profile_cursor_from_x(ax, event.xdata, update_canvas=True)

    def _on_2d_canvas_button_release(self, _event):
        self._profile_drag_ax = None

    def _on_2d_cursor_move(self, event):
        if hasattr(self, "profile_cursor_enabled_var") and bool(self.profile_cursor_enabled_var.get()):
            if self._profile_drag_ax is not None:
                ax = self._profile_drag_ax
                if event.inaxes is ax and event.xdata is not None:
                    self._set_profile_cursor_from_x(ax, event.xdata, update_canvas=True)
                return

        if self.results_2d is None:
            return
        ax = getattr(event, "inaxes", None)
        if ax is None or (ax not in self._cursor_axes_map):
            return
        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        if xdata is None or ydata is None:
            return

        info = self._cursor_axes_map[ax]
        x_coords = info["x_coords"]
        y_coords = info["y_coords"]
        arr = info["data"]
        ix = self._nearest_index(x_coords, xdata)
        iy = self._nearest_index(y_coords, ydata)
        val = float(arr[iy, ix])
        x_val = float(x_coords[ix])
        y_val = float(y_coords[iy])
        self.cursor_info_var.set(
            f"Cursor {info['name']}: {info['x_label']}={x_val:.4f} um, "
            f"{info['y_label']}={y_val:.4f} um, I={val:.6g} (ix={ix}, iy={iy})"
        )

    @staticmethod
    def _normalize_to_unit(arr):
        data = np.asarray(arr, dtype=float)
        if data.ndim != 2:
            raise ValueError("Only 2D arrays can be exported as TIFF data.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Array contains NaN or Inf and cannot be exported.")
        amin = float(np.min(data))
        amax = float(np.max(data))
        rng = amax - amin
        if rng > 0:
            out = (data - amin) / rng
        else:
            out = np.ones_like(data) if amax > 0 else np.zeros_like(data)
        return np.clip(out, 0.0, 1.0)

    def _save_array_as_tiff(self, path, arr):
        import matplotlib.pyplot as plt

        data = self._normalize_to_unit(arr)
        if tifffile is not None:
            tifffile.imwrite(path, data.astype(np.float32))
        else:
            plt.imsave(path, data, cmap="gray", vmin=0.0, vmax=1.0)

    def _save_axis_snapshot(self, ax, path, dpi=None, extra_axes=None):
        fig = self.fig_2d
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        bboxes = [ax.get_tightbbox(renderer)]
        if extra_axes:
            for extra in extra_axes:
                if extra is None:
                    continue
                bboxes.append(extra.get_tightbbox(renderer))
        bbox = Bbox.union(bboxes)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        save_kwargs = {"bbox_inches": bbox_inches}
        if dpi is not None:
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(path, **save_kwargs)

    def _show_2d_axis_context_menu(self, event):
        ax = getattr(event, "inaxes", None)
        if ax is None:
            return
        info = self._axes_export_map.get(ax)
        if info is None:
            return

        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label="Save plot as PDF (vector)",
            command=lambda ax=ax, info=info: self._save_plot_dialog(
                ax,
                info,
                title_suffix="PDF",
                defaultextension=".pdf",
                filetypes=[("PDF (vector)", "*.pdf"), ("All files", "*.*")],
            ),
        )
        menu.add_command(
            label="Save plot as SVG (vector text)",
            command=lambda ax=ax, info=info: self._save_plot_dialog(
                ax,
                info,
                title_suffix="SVG",
                defaultextension=".svg",
                filetypes=[("SVG (editable text)", "*.svg"), ("All files", "*.*")],
            ),
        )
        menu.add_command(
            label="Save plot as PNG (600 DPI)",
            command=lambda ax=ax, info=info: self._save_plot_dialog(
                ax,
                info,
                title_suffix="PNG",
                defaultextension=".png",
                filetypes=[("PNG image (600 DPI)", "*.png"), ("All files", "*.*")],
                dpi=600,
            ),
        )
        if info.get("array_data") is not None:
            menu.add_command(
                label="Save data as TIFF (normalized 0..1)",
                command=lambda info=info: self._save_data_tiff_dialog(info),
            )
        else:
            menu.add_command(label="Save data as TIFF (normalized 0..1)", state="disabled")
        menu.add_separator()
        menu.add_command(label="Cancel")

        gui_event = getattr(event, "guiEvent", None)
        if gui_event is not None and hasattr(gui_event, "x_root") and hasattr(gui_event, "y_root"):
            x_root = int(gui_event.x_root)
            y_root = int(gui_event.y_root)
        else:
            widget = self.canvas_2d.get_tk_widget()
            x_root = int(widget.winfo_pointerx())
            y_root = int(widget.winfo_pointery())

        try:
            menu.tk_popup(x_root, y_root)
        finally:
            menu.grab_release()

    def _save_plot_dialog(self, ax, info, title_suffix, defaultextension, filetypes, dpi=None):
        path = filedialog.asksaveasfilename(
            title=f"Save {info['name']} plot as {title_suffix}",
            initialdir=self.saved_simulation_dir,
            defaultextension=defaultextension,
            filetypes=filetypes,
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        out_path = path
        if ext == "":
            out_path = f"{path}{defaultextension}"

        try:
            self._save_axis_snapshot(
                ax=ax,
                path=out_path,
                dpi=dpi,
                extra_axes=info.get("extra_axes"),
            )
            self.status_var.set(f"Saved plot image for {info['name']}: {out_path}")
        except Exception as exc:
            messagebox.showerror("Save plot", str(exc))

    def _save_data_tiff_dialog(self, info):
        arr = info.get("array_data")
        if arr is None:
            messagebox.showwarning("Save data", "This plot does not have 2D array data to export.")
            return

        path = filedialog.asksaveasfilename(
            title=f"Save {info['name']} data as TIFF",
            initialdir=self.saved_simulation_dir,
            defaultextension=".tif",
            filetypes=[
                ("TIFF image", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        out_path = path
        if ext == "":
            out_path = f"{path}.tif"

        try:
            self._save_array_as_tiff(out_path, arr)
            self.status_var.set(f"Saved TIFF data for {info['name']}: {out_path}")
        except Exception as exc:
            messagebox.showerror("Save data", str(exc))

    def _clear_3d_render_cache(self):
        self._surface_cache.clear()
        self._surface_cache_order.clear()
        self._pyvista_camera_position = None
        if self._iso_refresh_job is not None:
            try:
                self.root.after_cancel(self._iso_refresh_job)
            except Exception:
                pass
            self._iso_refresh_job = None

    def _schedule_3d_refresh(self, delay_ms=0):
        if self._iso_refresh_job is not None:
            try:
                self.root.after_cancel(self._iso_refresh_job)
            except Exception:
                pass
            self._iso_refresh_job = None
        if int(delay_ms) <= 0:
            self._refresh_3d_plot()
            return

        def _run():
            self._iso_refresh_job = None
            self._refresh_3d_plot()

        self._iso_refresh_job = self.root.after(int(delay_ms), _run)

    def _downsample_3d_for_display(self, data, x_um, y_um, z_um):
        data = np.asarray(data, dtype=float)
        x_um = np.asarray(x_um, dtype=float)
        y_um = np.asarray(y_um, dtype=float)
        z_um = np.asarray(z_um, dtype=float)

        max_dim = int(max(1, self._max_3d_display_dim))
        max_n = int(max(data.shape))
        if max_n <= max_dim:
            return data, x_um, y_um, z_um

        step = int(np.ceil(max_n / max_dim))
        z_idx = np.arange(0, data.shape[0], step, dtype=int)
        y_idx = np.arange(0, data.shape[1], step, dtype=int)
        x_idx = np.arange(0, data.shape[2], step, dtype=int)
        if z_idx[-1] != data.shape[0] - 1:
            z_idx = np.append(z_idx, data.shape[0] - 1)
        if y_idx[-1] != data.shape[1] - 1:
            y_idx = np.append(y_idx, data.shape[1] - 1)
        if x_idx[-1] != data.shape[2] - 1:
            x_idx = np.append(x_idx, data.shape[2] - 1)

        data_ds = data[np.ix_(z_idx, y_idx, x_idx)]
        return data_ds, x_um[x_idx], y_um[y_idx], z_um[z_idx]

    def _get_surface_from_cache(self, key):
        cached = self._surface_cache.get(key)
        if cached is not None:
            try:
                self._surface_cache_order.remove(key)
            except ValueError:
                pass
            self._surface_cache_order.append(key)
        return cached

    def _put_surface_cache(self, key, value):
        self._surface_cache[key] = value
        try:
            self._surface_cache_order.remove(key)
        except ValueError:
            pass
        self._surface_cache_order.append(key)
        def cache_bytes():
            total = 0
            for item in self._surface_cache.values():
                try:
                    total += int(item.actual_memory_size) * 1024
                except Exception:
                    pass
            return total

        while (
            len(self._surface_cache_order) > int(self._surface_cache_max)
            or cache_bytes() > int(self._surface_cache_max_bytes)
        ):
            old = self._surface_cache_order.pop(0)
            self._surface_cache.pop(old, None)

    def _on_root_destroy(self, event=None):
        if event is not None and event.widget is not self.root:
            return
        self._cancel_width_entry_job()
        self._cancel_mask_preview_job()
        if self._qt_pump_job is not None:
            try:
                self.root.after_cancel(self._qt_pump_job)
            except Exception:
                pass
            self._qt_pump_job = None
        self._close_pyvista_window()

    def _update_3d_window_button_state(self):
        if not hasattr(self, "show_3d_btn"):
            return
        enabled = bool(
            HAS_PYVISTA_WINDOW
            and (self.results_3d is not None)
            and (not self.is_running)
            and (not self.opt_is_running)
        )
        self.show_3d_btn.configure(state=("normal" if enabled else "disabled"))

    def _update_export_csv_button_state(self):
        if not hasattr(self, "export_csv_btn"):
            return
        enabled = bool(
            (self.results_2d is not None)
            and (self.last_simulator is not None)
            and (not self.is_running)
            and (not self.opt_is_running)
        )
        self.export_csv_btn.configure(state=("normal" if enabled else "disabled"))

    def _ensure_qt_event_pump(self):
        if self._qt_app is None or self._qt_pump_job is not None:
            return
        interval_ms = max(1, int(self._qt_pump_interval_ms))
        self._qt_pump_job = self.root.after(interval_ms, self._pump_qt_events)

    def _pump_qt_events(self):
        self._qt_pump_job = None
        if self._qt_app is None:
            return
        try:
            self._qt_app.sendPostedEvents()
        except Exception:
            pass
        try:
            for _ in range(max(1, int(self._qt_pump_passes))):
                self._qt_app.processEvents()
        except Exception:
            return
        try:
            if self.root.winfo_exists():
                interval_ms = max(1, int(self._qt_pump_interval_ms))
                self._qt_pump_job = self.root.after(interval_ms, self._pump_qt_events)
        except Exception:
            self._qt_pump_job = None

    def _ensure_qt_app(self):
        if not HAS_PYVISTA_WINDOW:
            return None
        if self._qt_app is None:
            self._qt_app = QtWidgets.QApplication.instance()
            if self._qt_app is None:
                self._qt_app = QtWidgets.QApplication([sys.argv[0] or "psf_gui.py"])
        self._ensure_qt_event_pump()
        return self._qt_app

    def _on_pyvista_window_destroyed(self, *_args):
        self._pyvista_plotter = None

    def _get_live_pyvista_plotter(self):
        plotter = self._pyvista_plotter
        if plotter is None:
            return None
        try:
            plotter.windowTitle()
        except Exception:
            self._pyvista_plotter = None
            return None
        return plotter

    def _has_open_pyvista_window(self):
        plotter = self._get_live_pyvista_plotter()
        if plotter is None:
            return False
        try:
            return bool(plotter.isVisible())
        except Exception:
            self._pyvista_plotter = None
            return False

    def _close_pyvista_window(self):
        plotter = self._get_live_pyvista_plotter()
        self._pyvista_plotter = None
        if plotter is None:
            return
        try:
            plotter.close()
        except Exception:
            pass

    def _ensure_pyvista_plotter(self):
        if not HAS_PYVISTA_WINDOW:
            return None
        plotter = self._get_live_pyvista_plotter()
        if plotter is not None and (not self._has_open_pyvista_window()):
            self._close_pyvista_window()
            plotter = None
        if plotter is not None:
            return plotter

        self._ensure_qt_app()
        plotter = BackgroundPlotter(
            show=True,
            app=self._qt_app,
            window_size=(980, 760),
            toolbar=True,
            menu_bar=False,
            editor=False,
        )
        plotter.setWindowTitle("3D PSF Surface")
        try:
            plotter.app_window.destroyed.connect(self._on_pyvista_window_destroyed)
        except Exception:
            pass
        self._pyvista_plotter = plotter
        try:
            self._qt_app.processEvents()
        except Exception:
            pass
        return plotter

    def _raise_pyvista_window(self):
        plotter = self._get_live_pyvista_plotter()
        if plotter is None:
            return
        try:
            plotter.show()
            plotter.raise_()
            plotter.activateWindow()
        except Exception:
            pass

    @staticmethod
    def _axis_spacing(coords, default=1.0):
        coords = np.asarray(coords, dtype=float)
        if coords.size <= 1:
            return float(default)
        diffs = np.diff(coords)
        valid = diffs[np.isfinite(diffs) & (np.abs(diffs) > 0)]
        if valid.size == 0:
            return float(default)
        return float(np.mean(valid))

    def _capture_pyvista_camera(self):
        plotter = self._get_live_pyvista_plotter()
        if plotter is None:
            return None
        try:
            camera = plotter.camera
            return {
                "position": tuple(camera.GetPosition()),
                "focal_point": tuple(camera.GetFocalPoint()),
                "view_up": tuple(camera.GetViewUp()),
                "view_angle": float(camera.GetViewAngle()),
                "parallel_scale": float(camera.GetParallelScale()),
                "clipping_range": tuple(camera.GetClippingRange()),
            }
        except Exception:
            return None

    def _restore_pyvista_camera(self, camera_state):
        plotter = self._get_live_pyvista_plotter()
        if plotter is None or camera_state is None:
            return False
        try:
            camera = plotter.camera
            camera.SetPosition(*camera_state["position"])
            camera.SetFocalPoint(*camera_state["focal_point"])
            camera.SetViewUp(*camera_state["view_up"])
            camera.SetViewAngle(float(camera_state["view_angle"]))
            camera.SetParallelScale(float(camera_state["parallel_scale"]))
            camera.SetClippingRange(*camera_state["clipping_range"])
            camera.OrthogonalizeViewUp()
            return True
        except Exception:
            return False

    def _current_3d_display_state(self):
        if self.results_3d is None:
            return None

        r = self.results_3d
        data = self._display_intensity(self._choose_result_array(r, "I_xyz", "I_xyz_norm"))
        level = float(self.iso_level_var.get())
        mode = self.plot_mode_var.get()

        x_um = np.asarray(r["x_coords"], dtype=float) * 1e6
        y_um = np.asarray(r["y_coords"], dtype=float) * 1e6
        z_um = np.asarray(r["z_coords"], dtype=float) * 1e6
        raw_shape = tuple(int(v) for v in np.asarray(r["I_xyz"]).shape)
        data, x_um, y_um, z_um = self._downsample_3d_for_display(data, x_um, y_um, z_um)

        data_max = float(np.max(data)) if data.size else 0.0
        level_abs = level * data_max if data_max > 0 else level
        ds_note = ""
        if tuple(int(v) for v in data.shape) != raw_shape:
            ds_note = f", display={data.shape[2]}x{data.shape[1]}x{data.shape[0]}"

        return {
            "data": data,
            "x_um": x_um,
            "y_um": y_um,
            "z_um": z_um,
            "level": level,
            "level_abs": level_abs,
            "mode": mode,
            "ds_note": ds_note,
        }

    def _format_3d_title(self, display_state, plotted, backend):
        if display_state is None:
            return f"3D PSF Surface [{backend}]"

        if plotted:
            title = (
                f"3D surface ({display_state['mode']} level={display_state['level']:.2f}"
                f"{display_state['ds_note']})"
            )
        else:
            title = f"No surface at level={display_state['level']:.2f}{display_state['ds_note']}"

        if not bool(self.normalize_var.get()):
            title += f", abs={display_state['level_abs']:.3g}"
        return f"{title} [{backend}]"

    def _get_pyvista_surface(self, display_state):
        if display_state is None or not HAS_PYVISTA:
            return None

        cache_key = (
            "pyvista",
            int(id(self.results_3d)),
            int(self._power()),
            int(bool(self.normalize_var.get())),
            tuple(int(v) for v in display_state["data"].shape),
            round(float(display_state["level_abs"]), 6),
        )
        cached = self._get_surface_from_cache(cache_key)
        if cached is not None:
            return cached

        x_um = display_state["x_um"]
        y_um = display_state["y_um"]
        z_um = display_state["z_um"]
        data = np.asarray(display_state["data"], dtype=float)

        grid = pv.ImageData()
        grid.dimensions = (len(x_um), len(y_um), len(z_um))
        grid.origin = (float(x_um[0]), float(y_um[0]), float(z_um[0]))
        grid.spacing = (
            self._axis_spacing(x_um),
            self._axis_spacing(y_um),
            self._axis_spacing(z_um),
        )
        grid.point_data["psf"] = np.ascontiguousarray(np.transpose(data, (2, 1, 0)).ravel(order="F"))

        surface = grid.contour(isosurfaces=[float(display_state["level_abs"])], scalars="psf")
        if surface.n_points < 1 or surface.n_cells < 1:
            return None
        if int(surface.n_cells) > int(self._max_3d_faces):
            target_reduction = 1.0 - float(self._max_3d_faces) / float(surface.n_cells)
            try:
                surface = surface.decimate_pro(
                    max(0.0, min(0.99, target_reduction)),
                    preserve_topology=True,
                )
            except Exception:
                surface = surface.decimate(max(0.0, min(0.99, target_reduction)))

        self._put_surface_cache(cache_key, surface)
        return surface

    def _configure_pyvista_plotter(self, plotter, surface, title):
        plotter.clear()
        plotter.set_background("#f7f9fc")
        plotter.setWindowTitle(title)
        if surface is not None:
            plotter.add_mesh(
                surface,
                color="tomato",
                smooth_shading=True,
                opacity=0.82,
                specular=0.18,
                show_edges=False,
                render=False,
            )
            plotter.add_axes()
            if bool(self.show_3d_grid_var.get()):
                plotter.show_grid(
                    xtitle="X (um)",
                    ytitle="Y (um)",
                    ztitle="Z (um)",
                    color="#6b7280",
                    location="outer",
                    ticks="outside",
                )
            else:
                plotter.add_bounding_box(color="#6b7280", line_width=1.0)
            plotter.add_text(title, position="upper_edge", font_size=10, name="plot_title", render=False)
        else:
            plotter.add_text(title, position="upper_edge", font_size=10, name="plot_title", render=False)
            plotter.add_text(
                "No isosurface exists at the current level.\nTry lowering the 3D surface level.",
                position="upper_left",
                font_size=12,
                name="plot_notice",
                render=False,
            )

    def _show_3d_plot_window(self):
        if self.results_3d is None:
            self.status_var.set("No 3D result is available to display.")
            return
        if not HAS_PYVISTA_WINDOW:
            message = self._pyvista_unavailable_message()
            messagebox.showerror("PyVista unavailable", message)
            self.status_var.set(message)
            return
        self._refresh_3d_plot(show_window=True)

    def _refresh_3d_plot_pyvista(self, show_window=False):
        display_state = self._current_3d_display_state()
        if display_state is None:
            self._close_pyvista_window()
            return

        should_open = bool(show_window or self._has_open_pyvista_window())
        if not should_open:
            return

        surface = self._get_pyvista_surface(display_state)
        plotter = self._ensure_pyvista_plotter()
        if plotter is None:
            raise RuntimeError("PyVista window could not be created.")

        camera_state = self._capture_pyvista_camera() or self._pyvista_camera_position
        plotted = surface is not None
        title = self._format_3d_title(display_state, plotted=plotted, backend="PyVista")
        self._configure_pyvista_plotter(plotter, surface, title)

        if plotted:
            if camera_state and self._restore_pyvista_camera(camera_state):
                pass
            else:
                plotter.view_isometric(render=False)
                plotter.reset_camera(render=False)
                try:
                    plotter.camera.zoom(1.2)
                except Exception:
                    pass
            self._pyvista_camera_position = self._capture_pyvista_camera()
        plotter.render()
        if show_window:
            self._raise_pyvista_window()

    def _on_width_level(self, value):
        try:
            level = float(value)
        except Exception:
            level = self._current_width_level()
        self._cancel_width_entry_job()
        self._apply_width_level(level, source="slider", update_entry=True)

    def _on_width_level_entry_key_release(self, _event=None):
        self._cancel_width_entry_job()
        self._width_entry_job = self.root.after(120, self._apply_width_level_from_entry_live)

    def _apply_width_level_from_entry_live(self):
        self._width_entry_job = None
        level = self._parse_width_level_text(require_in_range=True)
        if level is None:
            return
        self._apply_width_level(level, source="entry", update_entry=False)

    def _on_width_level_entry(self, _event=None):
        self._cancel_width_entry_job()
        level = self._parse_width_level_text(clamp=True)
        if level is None:
            self._set_width_level_controls(
                self._current_width_level(),
                slider_level=self._quantize_width_level(self._current_width_level()),
                update_entry=True,
            )
            return
        self._apply_width_level(level, source="entry", update_entry=True)

    def _on_iso_level(self, value):
        try:
            level = float(value)
        except Exception:
            level = 0.50
        level = min(0.99, max(0.01, level))
        # Quantize to reduce expensive redraw churn while slider is dragged.
        level = round(level, 3)
        self.iso_level_var.set(level)
        self.iso_level_label.configure(text=f"{level:.2f}")
        self._schedule_3d_refresh(delay_ms=60)

    def _on_iso_level_entry(self, _event=None):
        try:
            level = float(self.iso_level_var.get())
        except Exception:
            level = 0.50
        self._on_iso_level(level)
        self._schedule_3d_refresh(delay_ms=0)

    def _on_3d_grid_toggle(self):
        self._refresh_3d_plot(show_window=False)

    @staticmethod
    def _side_view_xbins(lateral_lim, axial_lim):
        # Equal-aspect XZ/YZ panels are narrow when z spans more than x; five
        # tick labels then collide, so use fewer.
        lateral = abs(lateral_lim[1] - lateral_lim[0])
        axial = abs(axial_lim[1] - axial_lim[0])
        return 3 if lateral < 0.6 * axial else 5

    def _refresh_plots(self):
        self._refresh_2d_plots()
        self._refresh_3d_plot(show_window=False)

    def _refresh_2d_plots(self):
        if self.results_2d is None:
            self._cursor_axes_map = {}
            self._profile_axes_map = {}
            self._axes_export_map = {}
            self._width_plot_artists = {}
            self._clear_profile_cursor_artists()
            self._profile_drag_ax = None
            self._update_width_ratio_display()
            return

        r = self.results_2d
        p = self.last_params

        x_um = r["x_coords"] * 1e6
        y_um = r["y_coords"] * 1e6
        z_um = r["z_coords"] * 1e6
        x_lim = self._centered_axis_limits(x_um, default_half_span=1.0)
        y_lim = self._centered_axis_limits(y_um, default_half_span=1.0)
        z_lim = self._centered_axis_limits(z_um, default_half_span=1.0)

        i_xy = self._display_intensity(self._choose_result_array(r, "I_xy", "I_xy_norm"))
        i_xz = self._display_intensity(self._choose_result_array(r, "I_xz", "I_xz_norm"))
        i_yz = self._display_intensity(self._choose_result_array(r, "I_yz", "I_yz_norm"))

        width_metrics, x_prof, y_prof, z_prof = self._resolve_width_metrics_and_profiles(
            level=self._current_width_level(),
            exact=bool(self._width_level_source == "entry"),
        )
        if width_metrics is None:
            raise RuntimeError("Width metrics are unavailable for the current 2D result.")
        self._update_width_ratio_display(width_metrics)

        mask_extent = [
            -0.5 * p["size_x"] * 1e3,
            0.5 * p["size_x"] * 1e3,
            -0.5 * p["size_y"] * 1e3,
            0.5 * p["size_y"] * 1e3,
        ]

        ep_x_mm = r["entrance_pupil_x_coords"] * 1e3
        ep_y_mm = r["entrance_pupil_y_coords"] * 1e3
        ep_extent = [ep_x_mm[0], ep_x_mm[-1], ep_y_mm[0], ep_y_mm[-1]]

        pupil_before = self._display_intensity(
            self._choose_result_array(
                r,
                "intensity_at_entrance_pupil",
                "intensity_at_entrance_pupil_norm",
            ),
            apply_power=False,
        ).T
        pupil_after = self._display_intensity(
            self._choose_result_array(
                r,
                "intensity_after_pupil_mask",
                "intensity_after_pupil_mask_norm",
            ),
            apply_power=False,
        ).T
        optical_norm = intensity_norm(np.stack([pupil_before, pupil_after]), vmin=0.0)
        psf_vmax = max(
            float(np.nanmax(i_xy)) if np.size(i_xy) else 0.0,
            float(np.nanmax(i_xz)) if np.size(i_xz) else 0.0,
            float(np.nanmax(i_yz)) if np.size(i_yz) else 0.0,
            1.0,
        )

        self.fig_2d.clear()
        axes = self.fig_2d.subplots(3, 3, gridspec_kw={"height_ratios": [1.0, 1.0, 0.7]})

        im0 = axes[0, 0].imshow(
            np.asarray(r["input_intensity"], dtype=float).T,
            origin="lower",
            cmap="gray",
            extent=mask_extent,
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        style_image_axis(axes[0, 0], title="Mask", xlabel="x (mm)", ylabel="y (mm)")
        cbar0 = add_compact_colorbar(self.fig_2d, im0, axes[0, 0], "Mask value")

        im1 = axes[0, 1].imshow(
            pupil_before,
            origin="lower",
            cmap="magma",
            extent=ep_extent,
            aspect="equal",
            norm=optical_norm,
            interpolation="nearest",
        )
        style_image_axis(axes[0, 1], title="Fresnel", xlabel="x (mm)", ylabel="y (mm)")
        cbar1 = add_compact_colorbar(
            self.fig_2d,
            im1,
            axes[0, 1],
            self._display_intensity_label(apply_power=False),
        )

        im2 = axes[0, 2].imshow(
            pupil_after,
            origin="lower",
            cmap="magma",
            extent=ep_extent,
            aspect="equal",
            norm=optical_norm,
            interpolation="nearest",
        )
        style_image_axis(axes[0, 2], title="Pupil", xlabel="x (mm)", ylabel="y (mm)")
        cbar2 = add_compact_colorbar(
            self.fig_2d,
            im2,
            axes[0, 2],
            self._display_intensity_label(apply_power=False),
        )

        im3 = axes[1, 0].imshow(
            i_xy,
            origin="lower",
            cmap="magma",
            extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]],
            aspect="equal",
            vmin=0.0,
            vmax=psf_vmax,
            interpolation="nearest",
        )
        style_image_axis(
            axes[1, 0],
            title="XY",
            xlabel="x (um)",
            ylabel="y (um)",
        )
        self._set_image_extent(axes[1, 0], x_lim, y_lim)
        cbar3 = add_compact_colorbar(self.fig_2d, im3, axes[1, 0], self._display_intensity_label())

        im4 = axes[1, 1].imshow(
            i_xz,
            origin="lower",
            cmap="magma",
            extent=[x_lim[0], x_lim[1], z_lim[0], z_lim[1]],
            aspect="equal",
            vmin=0.0,
            vmax=psf_vmax,
            interpolation="nearest",
        )
        style_image_axis(
            axes[1, 1], title="XZ", xlabel="x (um)", ylabel="z (um)",
            xbins=self._side_view_xbins(x_lim, z_lim),
        )
        self._set_image_extent(axes[1, 1], x_lim, z_lim)
        cbar4 = add_compact_colorbar(self.fig_2d, im4, axes[1, 1], self._display_intensity_label())

        im5 = axes[1, 2].imshow(
            i_yz,
            origin="lower",
            cmap="magma",
            extent=[y_lim[0], y_lim[1], z_lim[0], z_lim[1]],
            aspect="equal",
            vmin=0.0,
            vmax=psf_vmax,
            interpolation="nearest",
        )
        style_image_axis(
            axes[1, 2], title="YZ", xlabel="y (um)", ylabel="z (um)",
            xbins=self._side_view_xbins(y_lim, z_lim),
        )
        self._set_image_extent(axes[1, 2], y_lim, z_lim)
        cbar5 = add_compact_colorbar(self.fig_2d, im5, axes[1, 2], self._display_intensity_label())

        norm_on = bool(self.normalize_var.get())
        y_label = "Normalized intensity" if norm_on else "Intensity (a.u.)"

        metric_level = float(width_metrics["level"])
        x_line = self._peak_scaled_level(x_prof, metric_level)
        y_line = self._peak_scaled_level(y_prof, metric_level)
        z_line = self._peak_scaled_level(z_prof, metric_level)

        axes[2, 0].plot(x_um, x_prof, color="#1B3B6F", linewidth=1.35, zorder=3)
        x_level_line = axes[2, 0].axhline(
            x_line, color="#4D4D4D", linestyle="--", linewidth=0.9, zorder=2
        )
        style_profile_axis(
            axes[2, 0],
            title=self._format_profile_axis_title("X"),
            xlabel="x (um)",
            ylabel=y_label,
        )

        axes[2, 1].plot(y_um, y_prof, color="#D55E00", linewidth=1.35, zorder=3)
        y_level_line = axes[2, 1].axhline(
            y_line, color="#4D4D4D", linestyle="--", linewidth=0.9, zorder=2
        )
        style_profile_axis(
            axes[2, 1],
            title=self._format_profile_axis_title("Y"),
            xlabel="y (um)",
            ylabel=y_label,
        )

        axes[2, 2].plot(z_um, z_prof, color="#007C73", linewidth=1.35, zorder=3)
        z_level_line = axes[2, 2].axhline(
            z_line, color="#4D4D4D", linestyle="--", linewidth=0.9, zorder=2
        )
        style_profile_axis(
            axes[2, 2],
            title=self._format_profile_axis_title("Z"),
            xlabel="z (um)",
            ylabel=y_label,
        )

        x_top = max(float(np.nanmax(x_prof)) if np.size(x_prof) else 0.0, float(x_line), 1e-6)
        y_top = max(float(np.nanmax(y_prof)) if np.size(y_prof) else 0.0, float(y_line), 1e-6)
        z_top = max(float(np.nanmax(z_prof)) if np.size(z_prof) else 0.0, float(z_line), 1e-6)
        axes[2, 0].set_ylim(0.0, 1.05 * x_top)
        axes[2, 1].set_ylim(0.0, 1.05 * y_top)
        axes[2, 2].set_ylim(0.0, 1.05 * z_top)

        self._cursor_axes_map = {
            axes[1, 0]: {
                "name": "XY",
                "x_coords": np.asarray(x_um, dtype=float),
                "y_coords": np.asarray(y_um, dtype=float),
                "data": np.asarray(i_xy, dtype=float),
                "x_label": "x",
                "y_label": "y",
            },
            axes[1, 1]: {
                "name": "XZ",
                "x_coords": np.asarray(x_um, dtype=float),
                "y_coords": np.asarray(z_um, dtype=float),
                "data": np.asarray(i_xz, dtype=float),
                "x_label": "x",
                "y_label": "z",
            },
            axes[1, 2]: {
                "name": "YZ",
                "x_coords": np.asarray(y_um, dtype=float),
                "y_coords": np.asarray(z_um, dtype=float),
                "data": np.asarray(i_yz, dtype=float),
                "x_label": "y",
                "y_label": "z",
            },
        }

        self._profile_axes_map = {
            axes[2, 0]: {
                "name": "X",
                "x_coords": np.asarray(x_um, dtype=float),
                "profile": np.asarray(x_prof, dtype=float),
                "x_label": "x",
            },
            axes[2, 1]: {
                "name": "Y",
                "x_coords": np.asarray(y_um, dtype=float),
                "profile": np.asarray(y_prof, dtype=float),
                "x_label": "y",
            },
            axes[2, 2]: {
                "name": "Z",
                "x_coords": np.asarray(z_um, dtype=float),
                "profile": np.asarray(z_prof, dtype=float),
                "x_label": "z",
            },
        }

        if bool(self.profile_cursor_enabled_var.get()):
            self._rebuild_profile_cursor_artists()
        else:
            self._clear_profile_cursor_artists()
            self._profile_drag_ax = None

        self._axes_export_map = {
            axes[0, 0]: {
                "name": "Input mask",
                "array_data": np.asarray(r["input_intensity"], dtype=float).T,
                "extra_axes": [cbar0.ax],
            },
            axes[0, 1]: {
                "name": "After Fresnel",
                "array_data": np.asarray(
                    self._display_intensity(
                        self._choose_result_array(
                            r,
                            "intensity_at_entrance_pupil",
                            "intensity_at_entrance_pupil_norm",
                        ),
                        apply_power=False,
                    ).T,
                    dtype=float,
                ),
                "extra_axes": [cbar1.ax],
            },
            axes[0, 2]: {
                "name": "After pupil mask",
                "array_data": np.asarray(
                    self._display_intensity(
                        self._choose_result_array(
                            r,
                            "intensity_after_pupil_mask",
                            "intensity_after_pupil_mask_norm",
                        ),
                        apply_power=False,
                    ).T,
                    dtype=float,
                ),
                "extra_axes": [cbar2.ax],
            },
            axes[1, 0]: {
                "name": "XY heatmap",
                "array_data": np.asarray(i_xy, dtype=float),
                "extra_axes": [cbar3.ax],
            },
            axes[1, 1]: {
                "name": "XZ heatmap",
                "array_data": np.asarray(i_xz, dtype=float),
                "extra_axes": [cbar4.ax],
            },
            axes[1, 2]: {
                "name": "YZ heatmap",
                "array_data": np.asarray(i_yz, dtype=float),
                "extra_axes": [cbar5.ax],
            },
            axes[2, 0]: {"name": "X intensity profile", "array_data": None, "extra_axes": []},
            axes[2, 1]: {"name": "Y intensity profile", "array_data": None, "extra_axes": []},
            axes[2, 2]: {"name": "Z intensity profile", "array_data": None, "extra_axes": []},
        }
        self._width_plot_artists = {
            "cache_key": self._current_width_metrics_cache_key(),
            "x_axis": axes[2, 0],
            "y_axis": axes[2, 1],
            "z_axis": axes[2, 2],
            "x_level_line": x_level_line,
            "y_level_line": y_level_line,
            "z_level_line": z_level_line,
        }

        self.fig_2d.align_labels()
        self.canvas_2d.draw()

    def _refresh_3d_plot(self, show_window=False):
        if self.results_3d is None:
            self._close_pyvista_window()
            return
        if not HAS_PYVISTA_WINDOW:
            if show_window:
                self.status_var.set(self._pyvista_unavailable_message())
            return
        try:
            self._refresh_3d_plot_pyvista(show_window=show_window)
        except Exception as exc:
            traceback.print_exc()
            self.status_var.set(f"PyVista window render failed: {exc}")


def launch_gui():
    root = tk.Tk()
    PSFSimulatorApp(root)
    root.mainloop()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) > 0:
        from psf_cli import main as cli_main

        return cli_main(argv, gui_launcher=launch_gui)
    launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
