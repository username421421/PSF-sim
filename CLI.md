# PSF Simulator CLI

The GUI still launches with no arguments:

```powershell
python psf_gui.py
```

All CLI commands are available through either entry point:

```powershell
python psf_cli.py --help
python psf_gui.py simulate --help
run_psf_gui.bat simulate --help
```

## Common Workflows

Direct resin dipping for printing on the underside of a coverslip:

```powershell
python psf_gui.py profile --axis z --config examples/leica_63x_resin_dip_in.json --plot-mode I2 --csv dip_in_z.csv
```

Use `--aberration-model gibson_lanni` to include immersion-medium mismatch in
the full layer optical-path phase. `--t-id-um` supplies the design immersion
gap / free working distance. The example assumes a conventional coverslip
corrected Leica oil objective; confirm its design values for your lens.
The experimental glass thickness is zero because the substrate is beyond the
focus. See the direct-dip section in `README.md` for assumptions, beam-width
conventions, and model limits.
Profile summaries report the width of the selected `I` or `I2` signal in
`width_um`; `intensity_width_um` always retains the width of the unsquared
optical intensity.

Run a 2D simulation and save app-compatible, CSV, NPZ, and PNG outputs:

```powershell
python psf_gui.py simulate --mode 2d --output "saved simulation\run.psfres" --csv run.csv --npz run.npz --plot run.png
```

Run a small Cartesian smoke-test simulation:

```powershell
python psf_gui.py simulate --integration-mode cartesian_jacobian --nx 16 --ny 16 --points-2d 9 --range-um 1 --fresnel-z-m 0 --quiet
```

Run explicit polar interpolation (the theta/phi controls are used only here):

```powershell
python psf_gui.py simulate --integration-mode polar_interp --theta-samples 61 --phi-samples 120 --nx 128 --ny 128 --points-2d 101 --fresnel-z-m 0
```

Export a saved GUI result:

```powershell
python psf_gui.py export "saved simulation\run.psfres" --csv run_export.csv --plot run_export.png
```

Trusted legacy pickle results are never loaded automatically:

```powershell
python psf_gui.py inspect legacy.psfres --allow-unsafe-legacy-pickle --json
```

Inspect a saved result or optimized mask:

```powershell
python psf_gui.py inspect "saved simulation\run.psfres"
python psf_gui.py inspect "optimized mask\mask.npz"
```

Run a quick single-axis 1D profile without computing full 2D heatmaps:

```powershell
python psf_gui.py profile --axis z --config config.json --csv z_profile.csv
python psf_gui.py profile --axis x --config config.json --csv x_profile.csv
python psf_gui.py profile --axis y --config config.json --z-um 0 --csv y_profile_at_z0.csv
```

Sweep annular blocking masks and save aggregate 1D profiles, compact profile arrays, and FWHM summaries:

```powershell
python psf_gui.py ring-sweep-z `
  --axis z `
  --config "saved simulation\mask dim vs z fwhm\mask vs z.json" `
  --output-dir "saved simulation\mask dim vs z fwhm" `
  --max-radius-mm 2.3 `
  --radius-points 20 `
  --range-z-um 20 `
  --points-2d 300
```

Use `--axis x`, `--axis y`, or `--axis z` for the 1D scan axis. For x/y profiles, add `--z-um 0` to force a z plane, or omit it to use each mask's on-axis z peak.
Each sweep writes both an all-in-one CSV and a compact `.npz` profile bundle named like `ring_mask_z_profiles.npz`. Mask geometry is saved as inner/outer radii and diameters; mask image files are not written.

Write and reuse a parameter config:

```powershell
python psf_gui.py write-config config.json --wavelength-nm 780 --na 1.4 --include-optimizer
python psf_gui.py simulate --config config.json --output run.psfres
```

Create a self-contained physical-transmission mask and summarize it after
rotation, resizing, and offset:

```powershell
python -c "import numpy as np; np.save('example_mask.npy', np.ones((32,32), dtype=np.float64))"
python psf_gui.py mask-info --mask example_mask.npy --mask-normalization physical --nx 64 --ny 64 --json
```

List optimizer backends:

```powershell
python psf_gui.py optimizers
```

Run mask optimization:

```powershell
python psf_gui.py optimize --nx 32 --ny 32 --points-2d 21 --fresnel-z-m 0 --output optimized_mask.npz --maxiter 5 --allow-expensive
```

Optimizer summaries report `converged` or
`budget_exhausted_with_candidate`. Every returned candidate is evaluated by the
exact Cartesian solver and is no worse than the exact all-pass baseline within
`1e-10` objective tolerance. The NPZ mask is a post-Fresnel complex-amplitude
transmission and can be supplied through a version-2 config or `--mask`.

For large simulations, request an automatic chunk or acknowledge the advisory
threshold:

```powershell
python psf_gui.py simulate --mode 3d --points-3d 101 --auto-z-chunk --allow-expensive
```

## Notes

- Mask inputs support `.npy`, `.csv`, `.txt`, `.tif`, `.tiff`, and common image formats.
- Result files written by `simulate --output` and the GUI are safe `.psfres`
  version-2 containers. Default readers use `allow_pickle=False`.
- CSV exports use the GUI export schema: `psf_sim_gui_csv_export_v1`.
- `--objective` accepts the objective lens preset JSON files from `objective lens\`.
- Existing config v1 files and last-used v6 files migrate to Cartesian mode.
  The deprecated `n_sd` field remains readable but is inert in this optical model.
