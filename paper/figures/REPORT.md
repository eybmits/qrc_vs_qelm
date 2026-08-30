# QRC vs QELM Evidence Report

This report is generated only from the loaded stage score CSVs and manifests.

The original stage report below is followed by the separately frozen recurrence
fingerprint addendum. The addendum does not modify the original phase-map gates.

## Inputs
- Stage 1: `results_stage1_full` (6528 score rows); verdict `PASS`.
- Stage 2: `results_stage2_full` (9792 score rows); verdict `SHOT_INSENSITIVE`.
- Stage 3: `results_stage3_full` (10080 score rows); verdict `NO_TRADEOFF`.

## RQ1 Phase Diagram
Stage 1 verdict: `PASS`.
G1b with uninformative exclusion: `PASS` (QRC=0.47, QELM=0.33, tie=0.20, n=15).
G1b without uninformative exclusion: `SINGLE_REGIME` (QRC=0.37, QELM=0.19, tie=0.44, n=27).
G1c with uninformative exclusion: `PASS` (fraction=0.625).
G1c without uninformative exclusion: `PASS` (fraction=0.278).
Phase-cell labels from evaluated G1a records: QRC=10, QELM=5, tie=12.
QRC-favored significant regions: tau=2, p=3; tau=3, p=1; tau=3, p=2; tau=5, p=1; tau=5, p=2; tau=8, p=1; tau=10, p=1.
QELM-favored significant regions: tau=0, p=2; tau=0, p=3; tau=1, p=2; tau=1, p=3; tau=2, p=2.
Uninformative excluded cells: `tau3_p3`, `tau5_p3`, `tau8_p2`, `tau8_p3`, `tau10_p2`, `tau10_p3`, `tau12_p1`, `tau12_p2`, `tau12_p3`, `tau16_p1`, `tau16_p2`, `tau16_p3`.
Anchor-task placements: STM tau1 at tau=1, p=1; parity tau1 at tau=1, p=2; STM tau2 at tau=2, p=1; NARMA2 at tau=2, p=2; parity tau2 at tau=2, p=2; STM tau3 at tau=3, p=1; parity tau3 at tau=3, p=3; STM tau5 at tau=5, p=1; NARMA5 at tau=5, p=2; parity tau5 at tau=5, p=5; STM tau8 at tau=8, p=1; parity tau8 at tau=8, p=8; STM tau10 at tau=10, p=1; NARMA10 at tau=10, p=2; parity tau10 at tau=10, p=10; STM tau12 at tau=12, p=1; parity tau12 at tau=12, p=12; STM tau16 at tau=16, p=1; parity tau16 at tau=16, p=16; MG at tau=17, p=3.

## RQ2 Shot Scaling
Stage 2 verdict: `SHOT_INSENSITIVE`.
Mean favored-cell counts by shot tier: infinity: QRC=9.00, QELM=5.00; 1e5: QRC=8.67, QELM=4.33; 1e4: QRC=7.00, QELM=5.00; 1e3: QRC=5.00, QELM=4.67.

## RQ3 Gamma Tradeoff
Stage 3 verdict: `NO_TRADEOFF`.
G3a aggregate crossover gamma*: `0.005` with max crossover fraction `0.5185185185185185`.
collective_loss: first gamma with QELM on at least half the cells is 0.005.
local_loss: first gamma with QELM on at least half the cells is 0.1.
No `features_dead=True` collapses are present in the loaded Stage-3 data.

## Figures
- `phase_diagram.png`: phase diagram.
- `boundary_vs_shots.png`: boundary vs shots.
- `gamma_tradeoff.png`: gamma tradeoff.

## Limitations
- Finite scale: loaded manifests report `n_qubits=5`; the report should not be extrapolated beyond that scale.
- Degree grid: loaded phase-diagram degrees are `1, 2, 3`; degree 5 is excluded from the loaded phase-diagram gate grid.
- Reproduction budget: the first loaded stage contains 24 executed seed cells in its score rows, so the evidence is moderate rather than exhaustive.
- Input injection: the comparison keeps the intentional QRC/QELM input-injection asymmetry from the protocol.

## Recurrence Fingerprint Addendum

Source: `results_recurrence_fingerprint_heldout/recurrence_fingerprint_study.json`
and its sealed CSV/manifest set. Execution status is `COMPLETE`; the registered
scientific verdict is **`NO_GO`**.

### Primary 10,000-shot gate ledger

- Balanced accuracy: 0.832682, CI [0.729167, 0.916667] — **FAIL** against 0.85.
- Physics-cluster lower bound: 0.729167 — pass against 0.70.
- Recurrent sensitivity: 128/192 = 0.666667.
- Finite specificity: 767/768 = 0.998698.
- Correct: 895/960 = 0.932292.
- Abstentions: 65/960 = 0.067708.
- Observed decisive errors: 0/960.
- False recurrence: 0/768 — pass.
- Cutoff within one: 767/768 = 0.998698 — pass.
- Strict decisive exact agreement: 895/960 = 0.932292, CI
  [0.891667, 0.966667] — pass.

Only the balanced-accuracy point gate fails; because the gates are conjunctive,
the result must remain `NO_GO`.

### Failure anatomy and task link

- Exact precision: 112/120 correct, 8/120 low-signal abstentions, 0 observed
  decisive errors; exact balanced accuracy is 0.833333.
- All eight dephasing QRC blocks are below the registered baseline-signal floor;
  all 16 local/collective-loss QRC blocks are resolved. This partition is
  post-hoc descriptive and must not be presented as a causal dissipator gate.
- The only additional 10,000-shot abstention is replay QELM d=5 in H23 for one
  measurement seed.
- The registered task crossover passes: mean shift 0.536440, CI
  [0.433299, 0.644845], with 23/24 positive system shifts.
- The 1,000-shot stress tier has 638 correct, 322 abstentions, and 0 observed
  decisive errors; it supports conservative abstention, not low-shot robustness.

### Addendum integrity

Opening and closing provenance pass, all source hashes remain unchanged, and the
full Cartesian counts are 1,762,560 curve scores, 3,672 predictions, 216 key
rows, 240 task scores, and 24 task-system rows. The paper figures explicitly
display the `NO_GO` status; their source tables are
`fingerprint_visibility_by_system.csv`,
`fingerprint_classification_summary.csv`, and
`task_crossover_by_system_tau.csv`.
