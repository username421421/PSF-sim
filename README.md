# PSF Simulator

Python/Tk application and headless CLI for simulating Cartesian point-spread functions (PSFs), applying pupil/mask effects, Fresnel propagation, Gaussian illumination envelopes, and mask sweeps.

This repository is intentionally source-only. Large simulation outputs, mask images, optimized masks, PDFs, and local notes are ignored by Git.

## Tracked Source

- `psf_gui.py` - Tk GUI entry point.
- `psf_cli.py` - headless CLI entry point and batch/sweep automation.
- `PSF_simulator.py` - optical simulation backend.
- `psf_mask_optimizer.py` - mask optimization backend.
- `psf_models.py` - typed request, mask, prepared-field, and resource models.
- `psf_io.py` - safe versioned serialization helpers.
- `psf_runtime.py` - shared GUI/CLI construction and resource preflight.
- `mask_builder.py` - mask-building helpers.
- `plot_style.py` - plotting style helpers.
- `run_psf_gui.bat` - Windows launcher.
- `CLI.md` - CLI examples and command notes.

## Setup

Use Python 3.10+ with the scientific Python stack:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install numpy scipy matplotlib torch tifffile cycler
```

Tkinter is required for the GUI and is included with most Windows Python installers.

The numerical core preserves float64/complex128. CUDA is optional; explicit
`--torch-device cuda` fails clearly when CUDA is unavailable.

## Launch GUI

```powershell
python psf_gui.py
```

or:

```powershell
run_psf_gui.bat
```

## CLI Basics

Show available commands:

```powershell
python psf_gui.py --help
```

Run a quick single-axis profile:

```powershell
python psf_gui.py profile --axis z --points-2d 300 --range-z-um 5 --csv z_profile.csv
```

Cartesian-Jacobian integration is the compatibility default. It samples the
input pupil grid directly; theta/phi counts are active only in explicit polar
mode:

```powershell
python psf_gui.py simulate --integration-mode polar_interp --theta-samples 61 --phi-samples 120 --fresnel-z-m 0
```

Run an annular ring-mask diameter sweep without saving mask images:

```powershell
python psf_gui.py ring-sweep-z `
  --axis z `
  --output-dir "saved simulation\mask dim vs z fwhm" `
  --max-radius-mm 2.8 `
  --radius-points 100 `
  --range-z-um 5 `
  --points-2d 300
```

The sweep writes:

- `ring_mask_<axis>_profiles.csv`
- `ring_mask_<axis>_profiles.npz`
- `ring_mask_<axis>_summary.csv`
- `ring_mask_<axis>_metadata.json`

The mask geometry is stored as inner/outer radii and diameters. Mask image files are not written.

## Numerical and file semantics

- A lower downstream refractive index truncates nominal pupil support; it does
  not stretch the accessible pupil to the rim. Summaries report requested and
  effective NA plus the excluded-pupil fraction.
- Numeric masks use physical transmission by default and must already be finite
  in `[0,1]`. `--mask-normalization minmax` is the explicit legacy image mode.
- Optimizer masks are complex-amplitude transmissions applied after Fresnel
  propagation and before the pupil. Saved optimizer NPZ files preserve phase.
- New `.psfres` files are safe version-2 ZIP/NPZ containers with JSON metadata
  and no object arrays. Trusted legacy pickle results require the explicit
  `--allow-unsafe-legacy-pickle` flag or a GUI confirmation.
- Requests estimated above 25% of available RAM/VRAM require
  `--allow-expensive`; requests above 70% are rejected. `--auto-z-chunk`
  selects a conservative 3D chunk.

## Audited performance

The final five-run medians on Windows 11, Python 3.13.2, Torch 2.6.0/CUDA
11.8, and an RTX 3080 Ti are recorded in
`benchmarks/final_windows_py313_rtx3080ti.json`: 298 ms CPU / 122 ms CUDA for
the default 2D workload, 822 ms CPU / 310 ms CUDA for 61 cubed, and 16.4 ms
CPU / 4.94 ms CUDA per production-sized profile mask. The CUDA 3D result uses
the resource-checked full-z chunk selected by `--auto-z-chunk` (3.17 GB peak
allocated VRAM in this environment). Explicit chunk choices are never
overridden.
