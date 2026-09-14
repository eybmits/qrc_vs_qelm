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
