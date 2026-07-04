# QRC vs QELM Gate Report

This report is generated only from the loaded stage score CSVs and manifests.

## Inputs
- Stage 1: `results_stage1_full` (6528 score rows); verdict `PASS`.
- Stage 2: `results_stage2_full` (9792 score rows); verdict `SHOT_INSENSITIVE`.
- Stage 3: `results_stage3_full` (10080 score rows); verdict `NO_TRADEOFF`.

## RQ1 Phase Diagram
Stage 1 verdict: `PASS`.
G1b with uninformative exclusion: `PASS` (QRC=0.47, QELM=0.33, tie=0.20, n=15).
G1b without uninformative exclusion: `SINGLE_REGIME` (QRC=0.37, QELM=0.19, tie=0.44, n=27).
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
- Reproduction budget: loaded manifests report 30 seed cells where that count is available, so the evidence is moderate rather than exhaustive.
- Input injection: the comparison keeps the intentional QRC/QELM input-injection asymmetry from the protocol.
