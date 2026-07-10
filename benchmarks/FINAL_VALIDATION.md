# Final benchmark validation

Environment: Windows 11, Python 3.13.2, NumPy 2.2.3, Torch 2.6.0+cu118,
CUDA 11.8, NVIDIA GeForce RTX 3080 Ti. Each timing is the median of five runs
after one warm-up; peak memory is collected in a separate run so its watcher
does not perturb timing.

| Workload | Final median | Acceptance gate | Result |
| --- | ---: | ---: | --- |
| Default 2D CPU | 0.298 s | <= 0.313 s | pass |
| Default 2D CUDA | 0.122 s | <= 0.586 s | pass |
| 61 cubed CPU, z chunk 8 | 0.822 s | <= 0.853 s | pass |
| 61 cubed CUDA, auto z chunk 61 | 0.310 s | <= 0.356 s | pass |
| Profile sequence CPU | 0.0164 s/mask | <= 0.0446 s/mask | pass |
| Profile sequence CUDA | 0.00494 s/mask | informational | pass |

The original test-only JSON baseline contains only 2D timings and predates the
corrected optical path. CUDA 2D improved from 0.677 s to 0.122 s. CPU 2D is
10.5% slower than that 0.270 s pre-correction value, but remains within the
0.313 s audited corrected-physics gate; preserving corrected float64/complex128
physics takes priority over the obsolete result.

The final isolated CPU working-set deltas were 156 MiB for 2D and 94 MiB for
61 cubed, versus the audit's approximate 300 MiB and 565 MiB respectively.
The full-z CUDA 3D run allocated 2.96 GiB on the device. Resource preflight
estimates 3.26 GiB for this workload, 10.4% above the observed peak.

Canonical data and all individual samples are in
`final_windows_py313_rtx3080ti.json`; the embedded baseline is
`baseline_windows_py313_rtx3080ti.json`.
