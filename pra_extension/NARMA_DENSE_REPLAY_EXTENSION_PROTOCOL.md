# Every-window NARMA replay extension

This author-requested extension evaluates every integer window d=1,...,48 on
the existing 24-root NARMA cohort (921000--921023). It is a post hoc extension
of an observed cohort, not a new independent confirmation. It does not change
the frozen NARMA, task-memory, or emulation protocols or their runners.

The parent is `paper/figures/task_memory/fig04_task_memory_source.json`. Its
hash, all frozen dependency hashes, the new runner and this note are recorded
before extension results are computed. Original NPZ files are absent locally;
the archived configuration, root seeds, selected settings, row metrics, and
first displayed trace permit deterministic reconstruction and cross-checks.

For every root retain its already selected QRC loss rate, coupling seed,
four-qubit architecture, 48 final-step features, exact expectations, and all
original raw-input generation, 33-level input quantization and NARMA recursion.
Train/validation/test lengths remain 4096/1024/2048 with 128 warm-up inputs;
the scored rows remain 3967/895/1919. No target history is an input. Each
window readout is trained on the original training rows, with training-only
standardization and the original ridge grid selected on validation only.
No QRC loss rate, QELM model, or baseline is reselected. The original QELM
four-input point, full-QRC reference, and displayed task traces stay unchanged.

The optimized algorithm initializes an independent QRC state for every possible
input start. At length d, its state contains exactly the first d-1 full-duration
updates from that start. Original final-step checkpoint maps give the features
for length d; a full-duration update advances the same prefix for d+1. Scores
use exactly the original endpoint indices. This reuses chronological prefixes
without inverse channels, interpolating scores, smoothing, or changing physics.

Before the full run, compare the optimized feature generator with the original
independent replay function and direct sequential reservoir evolution on a
small deterministic system, including d=1 and the longest tested window.
Reconstruct the selected continuous QRC per root and all six archived replay
windows (1,4,8,12,24,48). Require identical selected ridge values, dimensions,
windows and loss rates, and metric agreement at rtol=1e-8, atol=1e-10. These
cross-platform tolerances are fixed before the extension, not selected from
new outcomes. Verify the first root's archived target and QRC trace as well.

## Numerical reproducibility addendum after the initial first-root check

The initial strict cross-runtime metric gate failed at root 921000 before any
extension result was published. Its specification, runner and protocol
snapshots and numerical diagnostic are preserved in
`results_narma_dense_replay_v1/failed_attempt/`. The unchanged original replay
helper and shared-prefix implementation produced bit-identical features and
fits at all six archived windows on this host. Original targets reproduced
exactly. Archived differences were concentrated in alpha=1e-8 normal-equation
ridge systems with condition numbers around 8--9 trillion. The largest
first-root archived replay NRMSE difference was 1.25e-7; continuous-QRC NRMSE
differed by 5.36e-7, normalized validation MSE by 3.00e-6, and the first 96
predictions by at most 2.87e-6 in target units.

After that diagnostic, and before running the other 23 roots, a fresh v2
specification uses an absolute 1e-5 bound (no relative allowance) for archived
validation/test metrics and first-trace prediction comparisons. For NRMSE this
is 0.001 percentage point, 100 times smaller than the reported 0.1 percentage
point precision. Alpha, gamma, dimensions and split identity must still match
exactly; current-engine feature equivalence and independent target/score
checks keep strict tolerances. This is a documented cross-runtime numerical
reproducibility amendment, not a change to fitting, data, model selection,
physics or reported performance. Every point is recomputed; original values
are never spliced into the denser curve. The original reference remains an
archived reference, and all per-root discrepancies are retained. v1 remains
failed and is not resumed; full execution uses `results_narma_dense_replay_v2`.

## Complete execution and publication

Run all 24 roots and all 48 windows; retain every result, including nonmonotone
or unfavorable values. Store each root's test predictions, fitted readouts,
inputs, targets, and coupling matrix in a resumable local NPZ checkpoint,
alongside metrics and hashes. Publish 1152 rows and 48 median/10th--90th
percentile summaries. Percentiles describe root variability, not confidence
intervals. The continuous-QRC reference remains empirical, not an error floor.
Longer replay entails more physical evolution even though simulation prefixes
are computationally reused. No new superiority, quantum-advantage, or
worst-case-error claim is inferred from the denser display.

An independent verifier checks stored test scores and reconstructs predictions
at predetermined test rows using direct reservoir evolution and saved readouts.
The portable source ledger keeps complete rows, aggregates, parent/dependency
identities and checkpoint hashes; loading a published figure must not require
the uncommitted raw experiment directory. Frozen source files stay unchanged.
