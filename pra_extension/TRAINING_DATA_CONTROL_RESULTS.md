# Training-data follow-up: results

More training data consistently helps the tested QELMs, but only moderately.
A substantial residual remains, and quadratic delay remains more accurate.
The additional control strengthens the interpretation of the existing paper;
it does not isolate a pure feature-approximation floor or establish quantum
advantage.

## Completed design

This separate, prior-results-informed follow-up used 12 fresh root blocks,
each containing both prespecified anchors. All 384 planned combinations
completed: two anchors, four nested training sizes, and four models per root.
No roots or settings were removed. Training sizes were 480, 960, 1,920 and
3,840, with 512 separate validation rows and 2,048 test rows. The final
readout refit uses the indicated training rows plus 512 validation rows;
normalization of features uses training rows alone. Features are exact
expectations, not finite-shot measurements.

The configuration, analysis, source hashes and seed ledger were frozen locally
at 2026-09-07 22:55:05 UTC, before any production root was opened. This is not
an independently witnessed public preregistration. Existing frozen studies,
paper text and figures were not changed. Both current paper PDFs still have
12 pages.

## Prespecified primary results

These comparisons use the largest QELM feature budget and error to the exact
conditional mean for the declared IID 128-input prefix distribution,
evaluated on 1,024 independent windows per root. They do not measure the
worst-case scalar quantity D_d/2 or an oracle-optimal QELM readout.

| Target / window / QELM features | Median NRMSE, 480 rows | Median NRMSE, 3,840 rows | Paired median relative MSE reduction (95% root-bootstrap interval) | Improving roots |
|---|---:|---:|---:|---:|
| 4 qubits / 6 / 72 | 5.104% | 4.992% | 6.33% (4.33-7.73%) | 12/12 |
| 5 qubits / 10 / 120 | 15.013% | 14.157% | 13.69% (11.77-14.62%) | 12/12 |

The Holm-adjusted one-sided sign-test p-value is 0.000488 for each of the two
primary comparisons. Relative MSE reductions are calculated within paired
roots and then summarized; they are not ratios of the tabulated median
NRMSE values. The small p-values indicate consistent directions, not large
practical improvements. Streaming test NRMSE also decreases: 5.173% to
5.024% and 15.703% to 14.691%, respectively.

At 3,840 rows, quadratic delay reaches median reference NRMSE of 2.779% and
6.220%, with 27 and 65 features. It outperforms the largest tested QELM in
every root of each anchor. The comparison uses the same input windows and
splits, but is not dimension- or qubit-matched. All lower QELM budgets and
intermediate training sizes are retained in the source table and figure.

## What the new reference clarifies

The four-prefix empirical mean differs from the exact finite-prefix mean by
median NRMSE of 0.128% (four-qubit anchor) and 1.062% (five-qubit anchor).
This is small relative to the remaining QELM residual in these two anchors,
but is not identically zero. The realized empirical-minus-exact residual
MSE difference is negative in five of twelve four-qubit roots, illustrating
why finite-branch residual inflation is only an expectation statement, not
a pointwise identity. These conclusions need not hold for shorter windows
or other reservoirs.

Changing the reference prefix from 128 to 256 steps changes normalized RMS
by at most 1.79e-13 and 5.00e-11 across the respective roots and sampled
windows. This supports numerical stability of this finite-prefix reference;
it is not a proof of stationarity or uniform fading memory.

At the largest training size, the median numerically resolved feature ranks
are 54 of 72 and 97.5 of 120. Median condition numbers within the resolved
subspace are approximately 1.12e5 and 8.82e5. The historical relative SVD
cutoff of 1e-6 and validation-selected ridge penalties remain in use and are
recorded per fit. These facts, finite training and the streaming/reference
distribution distinction prevent assigning all remaining error to pure
feature approximation. Retain "residual emulator error" as the general term.

## Figure and inspectable evidence

- [Learning curves (PDF)](../results_training_data_control_v1/analysis/training_data_control.pdf)
- [Learning curves (PNG)](../results_training_data_control_v1/analysis/training_data_control.png)
- [All root-level results (CSV)](../results_training_data_control_v1/analysis/training_data_control_source.csv)
- [Primary and secondary analysis (JSON)](../results_training_data_control_v1/analysis/summary.json)
- [Figure caption](../results_training_data_control_v1/analysis/CAPTION.md)
- [Protocol](TRAINING_DATA_CONTROL_PROTOCOL.md)
- [Configuration](../configs/training_data_control.json)
- [Local freeze record](../configs/training_data_control_freeze.json)
- [Frozen source snapshot](../results_training_data_control_v1/validation/frozen_source.tar.gz)

The figure shows medians and 10th-90th percentile between-root whiskers, not
confidence intervals. Lines connect only the four tested training sizes.
Its two panels use the same axes and formatting. The exported PDF was
rendered and visually checked for clipping, overlap and legibility.

Preflight: 118 tests passed, duplicate development smokes matched scientific
hashes, and 222 planned seeds were disjoint from 1,094 recorded historical
seeds. Full suite: 739 passed, with zero failures, errors or skips. Test
reports and a source-only archive of all 49 frozen material files plus the
freeze are retained in the run's validation directory. Production coverage,
source and result hashes, finite-state integrity and fixed normalization
checks passed. The maximum finite-panel MSE identity discrepancy was
1.63e-19.

The complete production run took 668 seconds locally with one process and
one BLAS thread. The separate actual-dimension development benchmark used
approximately 1.41 GB peak resident memory; this is not a measured peak for
the full production run. Raw arrays can be regenerated from the frozen
sources and seed ledger; their hashes are archived, not the large arrays.

## Reproduce without overwriting this run

From the repository root, using the recorded dependency versions:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=pra_extension/code:. .venv/bin/python \
  pra_extension/code/run_training_data_control.py \
  --config configs/training_data_control.json \
  --freeze configs/training_data_control_freeze.json \
  --profile production --output results_training_data_control_reproduction

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  MPLCONFIGDIR=/private/tmp/qrc-mpl-cache PYTHONPATH=pra_extension/code:. \
  .venv/bin/python pra_extension/code/analyze_training_data_control.py \
  --config configs/training_data_control.json \
  --freeze configs/training_data_control_freeze.json \
  --results results_training_data_control_reproduction \
  --output results_training_data_control_reproduction/analysis
```

Both commands refuse a nonempty output directory. The runner and analyzer
check the frozen configuration and source identity. Reusing these seeds is
a reproducibility run, not an additional independent cohort.
