# PSF Simulator

Python/Tk application and headless CLI for simulating Cartesian point-spread functions (PSFs), applying pupil/mask effects, Fresnel propagation, Gaussian illumination envelopes, and mask sweeps.

This repository is intentionally source-only. Large simulation outputs, mask images, optimized masks, PDFs, and local notes are ignored by Git.

## Tracked Source

- `psf_gui.py` - Tk GUI entry point.
- `psf_cli.py` - headless CLI entry point and batch/sweep automation.
- `PSF_simulator.py` - optical simulation backend.
- `psf_mask_optimizer.py` - mask optimization backend.
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

## Git Hygiene

`.gitignore` is configured as an allowlist so Git tracks only simulator source/docs. Generated outputs and local data stay on disk but should not be committed.

