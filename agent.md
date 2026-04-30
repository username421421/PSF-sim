# Agent Guide

This repository contains a Python desktop and CLI application for simulating microscope point spread functions (PSFs) from pupil-plane masks. The main app is a Tkinter GUI named "PSF Simulator GUI (2D + 3D)"; the same core simulation code is also exposed through `psf_cli.py` and through subcommands on `psf_gui.py`.

Use this file as the first stop for future agents working in this repo.

## Quick Start

Launch the GUI:

```powershell
python psf_gui.py
```

Launch the GUI through the batch wrapper:

```powershell
run_psf_gui.bat
```

Show CLI help:

```powershell
python psf_cli.py --help
python psf_gui.py simulate --help
run_psf_gui.bat simulate --help
```

Run a small headless smoke simulation:

```powershell
python psf_gui.py simulate --nx 16 --ny 16 --theta-samples 5 --phi-samples 12 --points-2d 9 --range-um 1 --fresnel-z-m 0 --quiet
```

Run a 2D simulation and write common outputs:

```powershell
python psf_gui.py simulate --mode 2d --output "saved simulation\run.psfres" --csv run.csv --npz run.npz --plot run.png
```

Run a 3D simulation:

```powershell
python psf_gui.py simulate --mode 3d --points-3d 41 --z-chunk 8 --output "saved simulation\run_3d.psfres"
```

## Repository Layout

Top-level Python files:

- `psf_gui.py`: Tkinter GUI entry point. Owns UI state, plotting, mask loading, mask builder, simulation worker threads, save/load dialogs, CSV export, and optional PyVista 3D visualization.
- `psf_cli.py`: Headless command-line layer. It reuses the same simulator and optimizer logic as the GUI, handles JSON configs, mask loading, result export, ring sweeps, and optimization commands.
- `PSF_simulator.py`: Core optical simulation engine. Contains Fresnel propagation, Gaussian envelope application, vectorial PSF integration, 2D plane simulation, full 3D volume simulation, profile width metrics, and an axial-symmetric simulator variant.
- `psf_mask_optimizer.py`: Post-Fresnel mask optimizer. Uses SciPy optimizers plus Torch-backed fast paths to optimize amplitude, phase, or amplitude+phase masks against PSF width objectives.
- `mask_builder.py`: Small stateful model for creating binary/gray masks from geometric shapes in physical millimeter coordinates.
- `plot_style.py`: Shared Matplotlib styling helpers, normalization helpers, colorbars, and figure constructors.
- `run_psf_gui.bat`: Windows launcher that picks `.venv`, `venv`, `env`, `py -3`, or `python`, then passes all arguments to `psf_gui.py`.
- `PSF_x_z_cartesian.m`: Older MATLAB reference implementation for Cartesian XZ PSF calculations.
- `CLI.md`: User-facing CLI examples and workflows.

Data and output folders:

- `mask library\`: Input mask images used by the GUI and CLI.
- `objective lens\`: Objective lens preset JSON files.
- `optimized mask\`: Saved optimizer outputs and optimized mask images.
- `saved simulation\`: `.psfres`, CSV, JSON config, and analysis outputs. Many files here are user data and can be large.
- `Literature\`: Reference material.

Do not delete or rewrite the user data directories unless the user explicitly asks.

## Runtime Dependencies

The code imports these main third-party packages:

- `numpy`
- `torch`
- `scipy`
- `matplotlib`
- `cycler`
- `tifffile` for TIFF input/output when available
- `pyvista`, `PyQt5`, and `pyvistaqt` for the optional live 3D surface window

Tkinter is used for the GUI and is normally included with Python on Windows. The CLI can run headless for most workflows. PyVista dependencies are optional; if they fail to import, the GUI still runs but the 3D window feature is disabled.

There is currently no `requirements.txt` or package metadata in the repo.

## App Purpose

The app simulates how an input mask or pupil intensity distribution affects the focused PSF of an objective. It supports:

- all-pass input when no mask is loaded
- loading mask arrays/images from `.npy`, `.csv`, `.txt`, `.tif`, `.tiff`, and common image formats
- rotating masks by 90 degree increments
- applying physical x/y mask offsets
- optional Gaussian envelope before Fresnel propagation
- scalar Fresnel propagation to the entrance pupil
- entrance pupil diameter clipping
- vectorial PSF calculation in `XY`, `XZ`, and `YZ` planes
- full 3D PSF volume calculation
- normalized or raw display intensity
- `I` and `I^2` display/export modes
- peak width / FWHM style metrics at configurable intensity levels
- profile cursors and plot/data export
- mask construction from simple shapes
- mask optimization for amplitude, phase, or both

## Main Execution Flow

GUI entry flow:

1. `python psf_gui.py` calls `main()` in `psf_gui.py`.
2. With no arguments, `main()` calls `launch_gui()`.
3. `launch_gui()` creates a Tk root and instantiates `PSFSimulatorApp`.
4. `PSFSimulatorApp.__init__()` builds the UI, loads last-used settings, and initializes empty plots.

CLI entry flow:

1. `python psf_cli.py ...` calls `main()` in `psf_cli.py`.
2. `python psf_gui.py ...` delegates to `psf_cli.main(...)` when command-line arguments are present.
3. `run_psf_gui.bat ...` also passes arguments to `psf_gui.py`, so every CLI command works through the batch file too.

Simulation flow:

1. Input mask is loaded or an all-pass mask is generated.
2. Mask is resized to the configured `nx` by `ny` simulation grid.
3. Optional 90 degree rotation is applied.
4. Physical x/y mask offset is applied; shifted-out regions become all-pass transmission.
5. Optional Gaussian envelope is applied.
6. Fresnel propagation converts the input intensity map into a complex field.
7. Entrance pupil clipping is applied.
8. The pupil field may be trimmed around the pupil to reduce computation.
9. Integration tables are prepared.
10. The simulator computes either 2D planes, a full 3D volume, or a 1D profile.
11. Results are normalized, width metrics are extracted, and the GUI/CLI exports display data.

## Core Simulation Details

`CartesianPSFSimulator` in `PSF_simulator.py` is the main engine. Its documented processing chain is:

1. input intensity -> optional Gaussian envelope
2. optional custom/ring mask
3. Fresnel propagation
4. entrance pupil diameter mask
5. vectorial objective PSF calculation on XZ, YZ, and XY planes

Important functions and methods:

- `propagate_fresnel_field(...)`: scalar free-space propagation. Input intensity is normalized in `[0, 1]`, converted to zero-phase field `sqrt(intensity)`, then propagated by FFT transfer kernel.
- `apply_gaussian_profile(...)`: multiplies an intensity map by `exp(-((x / wx)^2 + (y / wy)^2))`.
- `prepare_pupil_field_from_input(...)`: prepares intermediate Fresnel, pupil, and PSF input arrays.
- `simulate_full_pipeline(...)`: full 2D plane pipeline used by quick 2D runs.
- `simulate_psf_planes_from_pupil_field(...)`: computes selected `XY`, `XZ`, and `YZ` planes from a prepared pupil field.
- `simulate_psf_volume_from_pupil_field(...)`: computes full 3D intensity volume `I(z, y, x)`.
- `simulate_psf_profile_full_pipeline(...)`: computes a 1D `x`, `y`, or `z` profile without full 2D heatmaps.
- `compute_peak_width_metrics(...)`: computes widths at a configurable normalized intensity level.

The default integration mode is `cartesian_jacobian`. There is also a `polar_interp` path, but the GUI and CLI construct `CartesianPSFSimulator` without overriding the default.

`AxialSymmetricPSFSimulator` extends `CartesianPSFSimulator` for fast radial optimization paths when axial symmetry is valid.

## Important Parameters

Mask/input grid:

- `nx`, `ny`: pixel dimensions of the input mask grid.
- `size_x`, `size_y`: physical mask size in meters internally, entered as millimeters in the GUI/CLI.
- `mask_offset_x`, `mask_offset_y`: physical offset in meters internally, entered as millimeters.

Optics:

- `wavelength`: excitation wavelength in meters internally, entered as nanometers.
- `NA`: objective numerical aperture.
- `pupil_d`: entrance pupil diameter in meters internally, entered as millimeters.
- `theta_samples`, `phi_samples`: angular integration sample counts.
- `n_i`, `n_g`, `n_s`: experimental immersion, coverslip, and sample/resin refractive indices.
- `t_g`, `t_s`: experimental coverslip thickness and sample/resin PSF position in meters.
- `n_id`, `n_gd`, `n_sd`, `t_gd`: design-side refractive indices and coverslip thickness used in aberration terms.

Propagation:

- `fresnel_z`: propagation distance in meters.
- `use_gaussian`: whether to apply Gaussian envelope.
- `gauss_wx`, `gauss_wy`: Gaussian widths in meters internally, entered as millimeters.

PSF sampling:

- `range_x_um`, `range_y_um`, `range_z_um`: half-width of output PSF coordinates in microns. A value of `2` means coordinates span `-2 um` to `+2 um`.
- `points_2d`: number of sample points per axis for 2D plane runs. `101` produces `101 x 101` per plane.
- `points_3d`: number of sample points per axis for full 3D volume runs. `61` produces `61 x 61 x 61` voxels.
- `z_chunk`: number of z slices computed at a time for full 3D runs. This changes memory/performance, not physical resolution or accuracy.

Use odd point counts when possible so the coordinate grid includes exactly zero.

## GUI Structure

`PSFSimulatorApp` owns the full UI and current app state.

Main left-side control groups:

- Mask loading, mask grid, physical mask size, offset, rotation, and preview.
- Optics settings and objective lens preset save/load.
- Experimental refractive index/thickness settings.
- Propagation settings.
- Simulation settings.
- Display controls for `I`, `I^2`, normalization, profile cursor, 3D grid, width level, and 3D surface level.
- Run, stop, save/load, CSV export, 3D window, and config buttons.

Main notebook tabs:

- `2D Results`: Matplotlib figure for PSF planes, entrance pupil views, profiles, and summary.
- `Mask Builder`: shape-based mask construction UI backed by `MaskBuilderModel`.
- `Mask Optimization`: optimizer controls and live preview backed by `PostFresnelMaskOptimizer`.

Simulation runs are executed in worker threads so the GUI remains responsive. Stop buttons set `stop_requested` or `opt_stop_requested`; the simulator/optimizer poll callbacks and raise `SimulationStoppedError` or `OptimizationStoppedError`.

## 2D, 3D, and 3D Chunk Behavior

`Run 2D Quick` calls `simulate_full_pipeline(...)` with `points_2d`. It computes:

- `I_xz`
- `I_xy`
- `I_yz`
- normalized versions
- `fwhm_x`, `fwhm_y`, `fwhm_z`
- on-axis profile arrays

Cost scales roughly with `3 * points_2d^2 * integration_points`.

`Run 3D Full` prepares the pupil field once, then calls `simulate_psf_volume_from_pupil_field(...)` with `points_3d` and `z_chunk`. It computes:

- full `I_xyz` volume shaped `(z, y, x)`
- normalized `I_xyz_norm`
- peak-centered `XY`, `XZ`, and `YZ` slices extracted from the volume
- FWHM/profile metrics from those slices

Cost scales roughly with `points_3d^3 * integration_points`, so increasing `points_3d` is expensive.

`z_chunk` only controls batching over z during 3D calculation. A larger chunk can be faster but uses more memory; a smaller chunk reduces peak memory but adds overhead.

## CLI Commands

The parser in `psf_cli.py` defines these subcommands:

- `gui`: launch the Tk GUI.
- `defaults`: print default CLI config JSON.
- `write-config`: write a reusable config JSON.
- `mask-info`: load, normalize, rotate, resize, offset, and summarize a mask.
- `simulate`: run 2D or 3D PSF simulation.
- `profile`: run a compact 1D profile along `x`, `y`, or `z`.
- `ring-sweep-z`: generate annular blocking masks and sweep z profiles.
- `export`: export or plot a saved `.psfres`/pickle result.
- `inspect`: inspect `.psfres`, `.pkl`, or `.npz` result files.
- `optimizers`: list optimizer categories and names.
- `optimize`: run post-Fresnel mask optimization.

Common parameter options are added by `add_common_param_options(...)`. They use user-friendly units:

- `--wavelength-nm`
- `--size-x-mm`, `--size-y-mm`
- `--pupil-d-mm`
- `--fresnel-z-m`
- `--range-um`, `--range-x-um`, `--range-y-um`, `--range-z-um`
- `--points`, `--points-2d`, `--points-3d`
- `--t-g-um`, `--t-s-um`, `--t-gd-um`
- `--gauss-wx-mm`, `--gauss-wy-mm`
- `--mask-offset-x-mm`, `--mask-offset-y-mm`

Mask options:

- `--mask`
- `--mask-rotate`
- `--mask-rotate-quadrants`

Display/export options:

- `--plot-mode`
- `--normalized` / `--raw`
- `--width-level`
- `--json`
- `--summary-json`

## File Formats

Simulation result files:

- GUI and CLI `.psfres` files are pickle payloads.
- Payload format key: `psf_sim_gui_result`.
- Version: `1`.
- Required keys: `params`, `results_2d`.
- Optional key: `results_3d`.
- `.pkl` and `.pickle` are accepted by the CLI inspect/export paths.

GUI CSV exports:

- Format metadata row uses `psf_sim_gui_csv_export_v1`.
- CSV rows are long-form records with `metadata`, `heatmap`, and `profile` record types.
- Heatmaps include `XY`, `XZ`, and `YZ`.
- Profiles include `X`, `Y`, and `Z`.

NPZ array exports:

- CLI `simulate --npz` writes compressed arrays and metadata.
- Optimizer `save_result_npz(...)` writes `amplitude_mask`, `phase_mask_rad`, `complex_mask`, JSON metadata, history, settings, baseline metrics, and optimized metrics.

Config files:

- CLI config format key: `psf_sim_cli_config`.
- GUI saved config format key: `psf_sim_gui_config`.
- Last-used GUI settings are saved to `saved simulation\last_used_simulation_config.json` with format `psf_sim_gui_last_used_config`.
- Float values are often encoded with `float64_hex` wrappers to preserve exact UI values.

Objective lens presets:

- Stored in `objective lens\`.
- Format key: `psf_sim_gui_objective_lens_preset`.
- Values include `NA`, `pupil_d_mm`, `n_id`, `n_gd`, `t_gd_um`, and `n_sd`.

Mask inputs:

- `.npy` is loaded with `np.load`.
- `.csv` and `.txt` are loaded with `np.loadtxt`, first comma-delimited, then whitespace-delimited fallback.
- `.tif`/`.tiff` use `tifffile` when available, otherwise Matplotlib image loading.
- Common images use Matplotlib image loading.
- 3-channel and 4-channel images are converted to grayscale by averaging RGB channels.
- Values are min/max normalized to `[0, 1]` and clipped.

## Mask Builder

`MaskBuilderModel` in `mask_builder.py` keeps an editable mask with geometry in millimeters.

Supported shape operations:

- circle
- ring
- rectangle

Supported edit modes:

- `add`: set selected region to the given value.
- `erase`: set selected region to zero.

The GUI mask builder can initialize blank masks, initialize from the current simulation mask, apply shapes, undo, clear, save masks, load masks into the builder, and use the builder output as the simulation mask.

Saved builder masks can be TIFF, NPY, CSV, or PNG depending on the selected extension.

## Mask Optimization

`PostFresnelMaskOptimizer` in `psf_mask_optimizer.py` optimizes a complex mask applied after Fresnel propagation and before entrance pupil masking.

Optimizer modes:

- `amplitude`
- `phase`
- `amplitude_phase`

Optimizer categories:

- Local gradient-based: `L-BFGS-B`, `SLSQP`
- Local derivative-free: `Powell`, `Nelder-Mead`
- Global: `Differential Evolution`, `SHGO` when SciPy exposes them

Objective settings:

- `width_level`: normalized intensity level for width measurement.
- `weights_xyz`: objective weights for x, y, and z widths.
- `weight_abs_xy`: optional penalty for absolute x/y width mismatch.
- `sidelobe_penalty_weight`: penalty for side lobes in surrogate paths.

Binary amplitude options:

- `pixel`: general pixel-level binary latent parameterization.
- `simple_ring_block`: optimize only inner/outer blocked ring radii; forces axial symmetry.
- `simple_ellipse_block`: optimize an elliptical ring with inner/outer x/y diameters; forces full 2D.

Fast paths:

- `_LinearizedFull2DFastPSF` precomputes reusable field factors when optimizing a full 2D mask.
- `_AxialSymmetricRadialFastPSF` supports radial/axial optimization.
- Some gradient-based runs use a smooth surrogate width loss with analytic Torch gradients.

The optimizer returns an `OptimizerResult` containing objective values, baseline/optimized metrics, masks, history, settings, and baseline/optimized 2D results.

## Plotting and Export Behavior

The GUI uses Matplotlib embedded in Tk through `FigureCanvasTkAgg`.

Plot styling lives in `plot_style.py`. Keep new Matplotlib plots consistent by using:

- `make_figure(...)`
- `style_image_axis(...)`
- `style_profile_axis(...)`
- `add_compact_colorbar(...)`
- `intensity_norm(...)`

The GUI supports context-menu plot export:

- PDF vector
- SVG vector text
- PNG at 600 DPI
- TIFF data export for 2D array-backed plots

Full 3D display uses PyVista when `pyvista`, `PyQt5`, and `pyvistaqt` are importable. The GUI caches isosurfaces and limits display complexity with `_max_3d_display_dim` and `_max_3d_faces`.

## Development Notes

Be careful with the current worktree. This repo often contains user-generated simulations and may already have modified or untracked files. Do not revert or delete data or source changes you did not make.

Source files are plain scripts rather than a package. Imports assume the current working directory is the repo root.

Use `rg` for search. Useful searches:

```powershell
rg -n "points_2d|points_3d|z_chunk" .
rg -n "simulate_full_pipeline|simulate_psf_volume_from_pupil_field" .
rg -n "save_result|load_result|write_export_csv" .
rg -n "OptimizerConfig|PostFresnelMaskOptimizer" .
```

Use small simulations for verification. Full 3D and optimizer runs can be very expensive.

Good smoke checks:

```powershell
python psf_cli.py --help
python psf_gui.py simulate --nx 16 --ny 16 --theta-samples 5 --phi-samples 12 --points-2d 9 --range-um 1 --fresnel-z-m 0 --quiet
python psf_gui.py profile --axis z --nx 16 --ny 16 --theta-samples 5 --phi-samples 12 --points-3d 9 --range-z-um 1 --fresnel-z-m 0 --quiet
```

For GUI-specific changes, launch `python psf_gui.py` manually when possible. For headless logic changes, prefer CLI smoke checks.

## Common Change Locations

Add or rename GUI controls:

- Build controls in `PSFSimulatorApp._build_layout(...)` and nearby helper methods.
- Add collection/validation in `_collect_params(...)` or `_collect_optimizer_settings(...)`.
- Add save/load persistence in `_collect_control_values(...)`, `_apply_control_values(...)`, and config helpers.
- Add CLI equivalents in `psf_cli.py` if the parameter should be scriptable.

Change simulation behavior:

- Start in `PSF_simulator.py`.
- Keep GUI and CLI call sites aligned:
  - GUI worker: `PSFSimulatorApp._worker_logic(...)`
  - CLI runner: `run_simulation(...)`
  - Profile runner: `run_profile(...)`

Change result/export schema:

- GUI result save/load: `_save_result_to_path(...)`, `_load_result_from_path(...)`.
- CLI result save/load: `save_result(...)`, `load_result(...)`.
- CSV export: `_write_current_data_csv(...)` in GUI and `write_export_csv(...)` in CLI.
- NPZ export: `write_arrays_npz(...)` in CLI.

Change mask loading:

- GUI: `_read_mask(...)` in `psf_gui.py`.
- CLI: `read_mask(...)` in `psf_cli.py`.
- Keep normalization and multi-channel handling consistent.

Change objective lens preset fields:

- GUI save/load code around `_save_objective_lens_dialog(...)` and related objective lens helpers.
- CLI loading through `load_objective_lens(...)`.

Change optimizer behavior:

- UI controls in `_build_optimization_tab(...)`.
- Validation in `_collect_optimizer_settings(...)`.
- CLI options in `add_optimizer_options(...)`.
- Optimizer internals in `psf_mask_optimizer.py`.

## Performance Notes

The expensive part of PSF simulation is the output grid multiplied by the number of integration points.

For 2D:

```text
cost roughly scales as 3 * points_2d^2 * integration_points
```

For 3D:

```text
cost roughly scales as points_3d^3 * integration_points
```

For the default `cartesian_jacobian` integration mode, the integration point count is related to the number of input grid points inside the entrance pupil. Larger `nx`, `ny`, and pupil diameter increase cost.

For `polar_interp`, integration points are approximately `theta_samples * phi_samples`.

Useful preview settings:

```text
nx, ny: 64 to 128
theta_samples: 11 to 31
phi_samples: 24 to 60
points_2d: 31 to 81
points_3d: 21 to 41
z_chunk: 4 to 8
```

Useful higher-quality settings:

```text
nx, ny: 256 or higher
theta_samples: 61 or higher
phi_samples: 120 or higher
points_2d: 101 to 201
points_3d: 61 or higher
z_chunk: 8 to 16 if memory allows
```

## Conventions and Cautions

- Keep units explicit. GUI labels often use nm, um, or mm, but simulator internals mostly use meters.
- Keep `points_2d`, `points_3d`, and `z_chunk` separate. `z_chunk` must not change output coordinates or scientific results.
- Preserve compatibility with existing `.psfres` and JSON config files when possible.
- Large files in `saved simulation\` and `optimized mask\` are user artifacts, not fixtures to rewrite casually.
- Do not introduce broad refactors unless the user asks; most code is direct and stateful.
- Prefer focused CLI smoke tests over expensive GUI/3D/optimization runs.
