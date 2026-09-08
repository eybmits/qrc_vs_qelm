# Figure 2 feature-density follow-up

All 252 planned rows completed: 12 fresh roots, three target sizes, six QELM
budgets plus the same-window quadratic control. There were no exclusions.
The locally frozen design and provenance are in configs/feature_density.json,
configs/feature_density_freeze.json and pra_extension/FEATURE_DENSITY_PROTOCOL.md.
Raw results and contrasts are in results_feature_density_v1/metrics.csv,
manifest.json and summary.json.

Median total common-suffix NRMSE (%), all 216 QELM rows retained:

| QRC qubits | 18 | 36 | 54 | 72 | 108 | 144 features |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 22.35 | 5.07 | 4.86 | 4.62 | 4.74 | 4.72 |
| 3 | 17.22 | 5.43 | 5.05 | 5.00 | 5.04 | 5.03 |
| 4 | 16.73 | 5.38 | 5.24 | 5.36 | 5.50 | 5.43 |

The main median change occurs between 18 and 36 features. The pre-specified
18->144 normalized-MSE contrasts improve in all 12 roots of each size;
all three Holm-adjusted sign p-values are 0.001465 (six comparisons total).
The 72->144 contrasts improve in 8, 9 and 7 roots, with adjusted p-values
0.388, 0.219 and 0.388. Their root-bootstrap intervals for median MSE
decrease include zero. This does not establish equivalence, a global optimal
feature budget, a pure approximation floor or quantum advantage.

The six-qubit emulator, input window, dynamics and total evolution time stay
fixed. Counts increase via uniformly spaced checkpoints, with nested banks
only where one count divides another. The old 1/2/4 schedules are unchanged.
All curves use exact expectations and historical fitting/panel normalization.
Different target sizes retain their different output dimensions and
target-specific normalization. Cohorts are not pooled.

Figure 2a adds the independent archived four-qubit cohort at the same five
positive loss rates and four windows as the original two-/three-qubit curve.
Figure 2b uses only the fresh cohort, not a splice of old/new points.
Figure 2c retains its existing data. The combined source CSV contains
912 rows and includes cohort/source hashes and raw metrics for a/b.
Medians and 10th-90th percentiles describe root variation, not confidence
intervals. All six observed feature points remain in the source without
smoothing. At the author's later display-only request, the main figure focuses
on 18, 36, 54 and 72; its caption discloses 108 and 144 in the full results.
The frozen protocol and the complete six-budget statistical analysis are
unchanged; this display crop does not establish a plateau or equivalence.

Validation: uniform-grid extraction matches direct evolution at all counts,
and the original schedules match exactly. Two development smokes have
identical scientific hashes; eleven dedicated tests passed before freezing.
Manual recomputation confirms six contrasts, 252 unique executed seeds,
complete coverage and the MSE identity (maximum discrepancy 1.63e-19).
Production took 20.62 seconds with single-threaded BLAS. Historical cohorts
and the separate training-data study were not rerun or amended.

Final integration checks: all 750 repository tests pass (120.42 seconds),
Ruff and whitespace checks pass, and both frozen study identities still match.
Both paper versions have 12 pages, with all pages rendered and visually
reviewed. The anonymous source archive rebuilds to the identical conference
PDF. This is not a claim of submission-policy approval.
Preflight and full-suite JUnit records are preserved under
results_feature_density_v1/validation. Its frozen_source.tar.gz contains all
46 frozen source files and the freeze record; all archived source hashes were
verified. Archive SHA256:
da6f8196b208f823dc9d34df7de3447eae1fc53b81df1528b5d0a55f37069519.

Reproduce with frozen sources and an unused output path:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=pra_extension/code:. .venv/bin/python \
  pra_extension/code/run_feature_density.py run \
  --output results_feature_density_reproduction
```

Re-render with pra_extension/code/render_information_interface_figures.py,
then paper/icaart2027/prepare_assets.py and paper/icaart2027/build_submission.py.
Use --feature-density-source to render a reproduction's metrics.csv instead
of the default original follow-up results.
