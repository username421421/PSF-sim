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

Run a 2D simulation and save app-compatible, CSV, NPZ, and PNG outputs:

```powershell
python psf_gui.py simulate --mode 2d --output "saved simulation\run.psfres" --csv run.csv --npz run.npz --plot run.png
```

Run a small smoke-test simulation:

```powershell
python psf_gui.py simulate --nx 16 --ny 16 --theta-samples 5 --phi-samples 12 --points-2d 9 --range-um 1 --fresnel-z-m 0 --quiet
```

Export a saved GUI result:

```powershell
python psf_gui.py export "saved simulation\run.psfres" --csv run_export.csv --plot run_export.png
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

Summarize a mask after normalization, rotation, resizing, and offset:

```powershell
python psf_gui.py mask-info --mask "mask library\blank.tif" --nx 256 --ny 256 --json
```

List optimizer backends:

```powershell
python psf_gui.py optimizers
```

Run mask optimization:

```powershell
python psf_gui.py optimize --mask "mask library\blank.tif" --output optimized_mask.npz --maxiter 30
```

## Notes

- Mask inputs support `.npy`, `.csv`, `.txt`, `.tif`, `.tiff`, and common image formats.
- Result files written by `simulate --output` use the same `.psfres` pickle payload format as the GUI.
- CSV exports use the GUI export schema: `psf_sim_gui_csv_export_v1`.
- `--objective` accepts the objective lens preset JSON files from `objective lens\`.
