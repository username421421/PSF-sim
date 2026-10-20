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

### Printing below a coverslip with a resin-dipped objective

If the front lens touches resin and the focus is on the underside of a
coverslip, the incoming path is **objective → resin → focus**. The coverslip is
a substrate beyond focus, not a layer traversed by the incident beam.

In **Experimental Values**, turn on **Dip-in lens (directly in resin)**.
This links immersion and placeholder coverslip RI to the editable resin RI,
sets experimental coverslip thickness and resin position to zero, and locks the
**Aberration model** selector (Optics) to `gibson_lanni`. The linked fields are
disabled and greyed out. Turning the toggle off restores the previous medium
settings and the model selected before. Outside dip-in the selector chooses
`legacy` or `gibson_lanni` freely, and loaded configurations keep their model.
The toggle and restore values are saved with GUI configurations. Objective
design values, NA, pupil diameter, illumination, and sampling remain
independently configurable.

For the illustrative 63× configuration, load
`examples/leica_63x_resin_dip_in.json` with **Load Config**, or run:

```powershell
python psf_gui.py profile --axis z --config examples/leica_63x_resin_dip_in.json --plot-mode I2 --csv dip_in_z.csv
```

The example assumes the conventional Leica 63×/1.4 oil objective discussed
here, a nominal 8.89 mm pupil, 1.518 design oil/glass RI, 140 µm design oil gap,
and 170 µm design coverglass. These are modeling assumptions, not verified
specifications for a purpose-built dipping lens. Confirm the objective model,
design coverglass requirement, pupil diameter, and RI at the laser wavelength.
The [Leica objective finder](https://www.leica-microsystems.com/products/microscope-accessories/microscope-objective-lens/objectivefinder/objective/506349/)
lists a 63×/1.4 oil model with coverglass and 0.14 mm free working distance.

Use these controls for the direct-dip geometry:

| Control | Value / meaning |
| --- | --- |
| Aberration model | `gibson_lanni` |
| Experimental immersion RI / resin RI | Both `1.474` |
| Experimental coverslip thickness | `0` µm; the substrate is beyond focus |
| Experimental coverslip RI | `1.474` as a zero-layer placeholder |
| Resin thickness / PSF position | `0` µm as a homogeneous-medium reference |
| Design immersion RI | Objective's intended immersion medium, e.g. `1.518` |
| Design immersion gap / free WD | Intended front-lens-to-coverglass gap; `140` µm in this example |
| Design coverslip thickness | Keep the objective's design value; use `0` only for a lens designed without coverglass |
| Input field size X/Y | `12` mm to contain the estimated `8.89` mm pupil |
| Fresnel propagation z | `0` for an input field specified at the pupil |

The optional `gibson_lanni` mode adds the full layer optical-path difference
to the existing vectorial integral. With `q = NA * pupil_radius_fraction`, it
uses

```text
W(q) = t_s sqrt(n_s²-q²) + gap sqrt(n_i²-q²) - t_id sqrt(n_id²-q²)
       + t_g sqrt(n_g²-q²) - t_gd sqrt(n_gd²-q²)
gap  = n_i (t_id/n_id + t_gd/n_gd - t_g/n_g - t_s/n_s)
```

The gap sets the paraxial reference focus; the numerical intensity maximum
may shift away from `z = 0`. The glass phase is included once in the existing
transmission terms. This follows the design-versus-experimental layer OPD in
the [Gibson–Lanni model](https://www.aimicroscopy.org/microsc_psf/).
The default `legacy` mode retains existing simulations' phase convention.

For homogeneous resin (`n_i = n_s`, no incoming coverslip), changing `t_s`
just reallocates the same total lens-to-focus path; it does not introduce a
new refracting boundary. This distinguishes dipping from focusing through
oil and glass into a mismatched resin. The model assumes an ideal objective
with fixed nominal pupil-to-NA mapping and does not trace its internal lens
prescription. It also omits the reflected field and other substrate effects
at the resin–glass printing interface.

Start with an all-pass pupil. Use the actual beam footprint or enable the
Gaussian envelope to compare pupil filling. Enter the beam's **intensity FWHM**
(GUI **Intensity FWHM x/y**, CLI `--gauss-fwhm-x-mm/--gauss-fwhm-y-mm`). For a
beam specified by its 1/e² intensity diameter D, the FWHM is
`D*sqrt(ln(2)/2) ≈ 0.589 D`; a 5.6 mm beam gives `3.30` mm. Internally the
envelope is `exp(-(x/w)^2)` in intensity with `w = FWHM/(2*sqrt(ln 2))`. Configs
store `w` as `gauss_wx`/`gauss_wy` in metres, so older configs load as the same
beam, and the deprecated `--gauss-wx-mm` flag still takes `w`. A Gaussian
does not impose a hard effective-NA cutoff. For polymerization, `I^2` is the
two-photon excitation proxy, not a prediction of developed voxel size.

The example uses finer polar integration and a ±25 µm axial range because
immersion mismatch can create rapid pupil phase variation and a large focus
shift: about 21 waves across the pupil here. In polar mode too few θ samples
alias that phase: the default 61 θ samples give an axial FWHM of 0.72 µm
instead of the converged 3.04 µm, so keep about 400 or more. Cartesian mode at
512 px was within about 1% for this example. Also check that the peak and both
width crossings lie inside the axial range.

### Optical model and literature

The references are in `Literature/`.

- `legacy` is the Nasse & Woehl (2010) illumination PSF: aberration phase
  Ψ − Ψ*, coverslip phase β − β*, and the aplanatic apodization evaluated at the
  design angle n_id sin θ* = n_i sin θ. It omits the optical path through the
  immersion layer itself.
- `gibson_lanni` sums the layer optical paths from the objective (Török &
  Varga 1997, Eq. 28, with the objective as the first medium). It adds the
  immersion-gap path, which matters whenever the immersion differs from the
  design immersion, e.g. dip-in.
- Effective NA follows Nasse & Woehl Sec. 2: total internal reflection at the
  objective → immersion → coverslip → sample interfaces limits the aperture to
  `NA_eff = min(NA, n_i, n_g, n_s)`. An NA above the immersion RI (an oil
  objective dipped into a lower-index resin) truncates the pupil rather than
  raising an error. The NA must stay below the design immersion RI `n_id`.
- Multiple reflections in the coverslip use the Airy denominator of Török &
  Varga Eq. 32, `1 + r12 r23 exp(2iβ)`, with the actual coverslip phase only.
  Nasse & Woehl Eq. 8 (and the original MATLAB code) also put the design phase
  β* in this denominator, which adds reflected light in phase at every angle
  when the coverslip matches the design (about +7% rim amplitude for a dry
  NA 0.95 objective). Index-matched oil objectives are unaffected.

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
- Memory use is not limited or prompted for, so check the estimate yourself
  for large grids. Optimizer runs above 1000 objective evaluations per
  iteration still need `--allow-expensive`. `--auto-z-chunk` selects a
  conservative 3D chunk.

## Audited performance

The final five-run medians on Windows 11, Python 3.13.2, Torch 2.6.0/CUDA
11.8, and an RTX 3080 Ti are recorded in
`benchmarks/final_windows_py313_rtx3080ti.json`: 298 ms CPU / 122 ms CUDA for
the default 2D workload, 822 ms CPU / 310 ms CUDA for 61 cubed, and 16.4 ms
CPU / 4.94 ms CUDA per production-sized profile mask. The CUDA 3D result uses
the resource-checked full-z chunk selected by `--auto-z-chunk` (3.17 GB peak
allocated VRAM in this environment). Explicit chunk choices are never
overridden.
