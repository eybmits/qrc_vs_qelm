# Direct task prediction: fresh-root memory replication

The two previously observed directions replicate on fresh roots: the recurrent
QRC improves on a four-input QELM, and restricting that same QRC to four inputs
removes much of the benefit. This statement concerns NARMA10 and the fast-loss,
next-step laser setting. Slow loss is a retained counterexample, not omitted
evidence. Both models learn actual task targets, not another model's output.

## Current manuscript selection

The latest Figure 3 integrates NARMA as the vertically stacked middle panel
of `paper/figures/memory_prediction/fig04_memory_prediction.pdf`, between
the earlier laser emulation and the paired-gap/exact-reference panels. Its
values and excerpt selection are unchanged; only the display is consolidated.
The previous two-column Figure 3 and separate NARMA plot remain alternatives.
The following paragraph records the immediately preceding selection.

The earlier emulation Figure 3 is restored as the active manuscript figure,
including its red unavoidable-window-error reference. The fresh 24-root
NARMA results remain complementary evidence in the text and in an
[optional standalone plot](../paper/figures/narma_additional/narma_additional.pdf)
with its [source ledger](../paper/figures/narma_additional/source.json).
That plot is not an additional manuscript float. The fresh laser-forecasting
results and combined task-memory figure below are retained as archived
additional evidence. No data, simulation settings, predictions, or results
changed with this figure-selection revision.

## Frozen scope

[Protocol](TASK_MEMORY_REPLICATION_PROTOCOL.md) was fixed before these runs.
The new wrapper leaves both original runners and their original 12-root results
unchanged. The original and fresh cohorts are not pooled.

- NARMA: roots 921000--921023; independent iid driving inputs and couplings.
  The reduced input range is [0,0.2] inside the NARMA10 recurrence. Models see
  driving inputs through time t and predict y[t+1], never receiving earlier y.
  This is synthetic task data, not a measured time series. Each model selects
  loss from 0.1/0.3/1/4 and ridge strength using validation only.
- Laser: roots 922000--922023; new couplings on the same measured record and
  chronological split. Both loss rates (4 and 0.2), horizons (1 and 4), and all
  history/frozen-readout controls remain available. The fast-loss, one-step
  emphasis was informed by the original study, not chosen blind to that study.
- Both studies: four qubits, 48 exact expectation features, the same declared
  ridge grid and paired inputs/couplings. Independent readouts are fitted for
  every history restriction. Longer sequential replay costs more evolution
  steps; it is not an identical-runtime parallel QELM. Both quantum models see
  the same 33-level input representation; the actual targets are unquantized.

Recorded per-cell simulation times sum to 403.5 seconds for NARMA and 359.5
seconds for laser on this environment. These are recorded runtime sums, not
portable wall-time promises or a resource-efficiency comparison.

## Independently verified results

NRMSE is test RMSE divided by training-target standard deviation. Percentages
are error scales, not prediction accuracies. The table contains medians over
the 24 fresh roots, not the selected illustrative trace's error.

| Task/setting | QRC | QELM, four inputs | Same QRC, replay/refit four inputs |
| --- | ---: | ---: | ---: |
| NARMA10, validation-selected dynamics | 59.67% | 77.21% | 77.31% |
| Laser, fast loss, one step | 15.89% | 20.30% | 18.31% |
| Laser, slow loss, one step | 24.25% | 20.28% | 18.27% |
| Laser, fast loss, four steps | 36.45% | 37.62% | 37.78% |
| Laser, slow loss, four steps | 36.79% | 37.76% | 36.43% |

The four prospective direction checks all pass: each requires at least 20 of
24 positive paired differences and a wholly positive individual two-sided
95% paired-bootstrap mean interval. Actual agreement is 24/24 for each.

| Primary paired difference | Mean, NRMSE percentage points | Individual 95% interval |
| --- | ---: | ---: |
| NARMA QELM(4) minus QRC | 17.59 | [15.74, 19.47] |
| NARMA replay/refit(4) minus QRC | 17.63 | [15.78, 19.51] |
| Laser fast/one-step QELM(4) minus QRC | 4.65 | [4.13, 5.18] |
| Laser fast/one-step replay/refit(4) minus QRC | 2.65 | [2.21, 3.11] |

These are means of within-root differences, not differences between the table's
medians. Bootstrap intervals use 10,000 paired resamples, seeds 939991 (NARMA)
and 939992 (laser); they do not have simultaneous-coverage guarantees. Laser
intervals concern coupling variability conditional on one reused record, not
uncertainty across independent real-world datasets. NARMA roots vary both
couplings and input streams.

The slow-loss, one-step laser result reverses the primary direction:
QELM-minus-QRC is -3.55 points, interval [-4.85,-2.29], with 4/24 positive roots;
replay(4)-minus-QRC is -5.54 points, [-6.75,-4.34], with 1/24 positive roots.
Thus longer-lived state is not automatically more useful. At the secondary
four-step horizon, fast loss has a smaller positive paired QELM-minus-QRC effect
(1.17 points, [0.72,1.62], 21/24 positive); the slow-loss interval crosses zero.
All secondary results remain in the source ledger for manuscript reporting.

For NARMA, replay/refit with 12 inputs reaches a 53.67% median. A classical
quadratic 12-input control reaches 10.27%, substantially better than the tested
quantum models. The classical four-/twelve-input controls have 14/90 features,
so these are context checks, not matched quantum-advantage comparisons.

## Archived combined figure and interpretation

[Archived figure PDF](../paper/figures/task_memory/fig04_task_memory.pdf),
[PNG](../paper/figures/task_memory/fig04_task_memory.png), and
[source ledger](../paper/figures/task_memory/fig04_task_memory_source.json).

- a: Measured laser input and actual next-step predictions, both loss settings.
- b: Synthetic NARMA input and actual target/predictions.
- c: Refitted history restrictions: how changing the available past changes
  error while the reservoir circuit and feature count remain fixed.

Traces use the first fresh root and first 96 scored test rows, without smoothing
or cross-root trajectory averaging. NARMA targets differ between roots; an
averaged trajectory would not be one model's prediction. The first laser root
is an exception where slow QRC performs better than QELM; the all-root history
summary, not that illustrative trace, establishes the overall reversal.
History curves aggregate all 24 rootwise errors at every tested window. Bands
and bars show empirical 10th--90th percentiles, not confidence intervals.

The supported conclusion is that access to older inputs can improve direct
task prediction and that a small available window can miss useful history.
The refitted same-circuit intervention is not an exact additive decomposition
of the QRC/QELM gap or an architecture-independent information-floor estimate.
Neither task shows failure for every finite window: longer replay can work
well. The paper's asymptotic impossibility result remains a separate theorem
and controlled protected-memory example. No claim of necessary quantum
recurrence, quantum advantage, finite-shot robustness, hardware validation, or
practical resource savings follows from this exact-expectation replication.

## Reproduction

Run from the repository root with the frozen laser file and the original two
ignored results directories available: initialization checks their identities.
Use a new output directory; the runners refuse overwrites. No ignored results,
raw data, or NPZ files should be committed.

```sh
export PYTHONPATH=pra_extension/code:src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/qrc-mpl-cache
.venv/bin/python pra_extension/code/task_memory_replication.py initialize --output results_task_memory_reproduction
.venv/bin/python pra_extension/code/task_memory_replication.py narma --output results_task_memory_reproduction
.venv/bin/python pra_extension/code/task_memory_replication.py laser --output results_task_memory_reproduction
.venv/bin/python pra_extension/code/task_memory_replication.py finish --output results_task_memory_reproduction
.venv/bin/python pra_extension/code/verify_task_memory_replication.py --source results_task_memory_reproduction --output results_task_memory_reproduction/verification.json
.venv/bin/python pra_extension/code/render_task_memory.py --source results_task_memory_reproduction --output paper/figures/task_memory_reproduction
```

Verification recomputed all 1,896 stored test-score rows (456 NARMA, 1,440
laser), checked every true target and NARMA RNG stream, preserved all candidates
and secondary settings, and independently reconstructed 7,920 saved-readout
predictions at five fixed rows per model/cell. Replay reconstruction used the
original sequential channel updates, not the batched replay helper. All four
direction checks were independently recalculated and agreed with the report.
Nine focused verifier tests pass. This is not a claim that every prediction
was independently reconstructed or that every ridge fit was independently
retrained; every saved prediction does enter the recomputed score.

Recorded identities:

- Replication specification: `aeaa42a23610adc5e185665189234dd48f6be6bc0bc4b15493b7d8ac3ec4062f`.
- NARMA summary: `55e66aab8e9aa46f68267a8c7d09fad8efe4d7a61095b8b5abc0a96b384ebc20`.
- Laser summary: `051e84ae1b8b5b9906bce7bd0b4fe2b8bd5e75579c3c93a151e061943ffcf171`.

The independent report is `results_task_memory_replication_v1/verification.json`.
New reproduction summaries can have different hashes because their metadata
contains measured runtime; numerical arrays should agree for identical inputs
and environment. The manuscript's page count, full-suite tests, and final PDF
layout are checked by the separate integration workflow.
