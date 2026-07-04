# Methods Protocol

This file defines how the frozen QRC-vs-QELM comparison is constructed and run.
Frozen commitments, gates, seed grids, row caps, shrink rules, and amendments live
in [PREREGISTRATION.md](PREREGISTRATION.md). The machine-readable frozen profile is
`data/frozen/frozen_v1_config.json`, mirrored by `configs/full.yaml`.

## Task Construction

The primary RQ1 phase diagram uses Legendre-delay targets
`P_p(2 u[t - tau] - 1)` on the frozen grid:

- `tau = {0, 1, 2, 3, 5, 8, 10, 12, 16}`
- `p = {1, 2, 3}` for gated fractions, giving 27 gated cells
- `p = {1, 2, 3, 5}` is still run and reported as a robustness appendix

The pre-registered uninformative-cell rule is applied before gate fractions are
formed: a target cell is excluded from the G1b/G1c denominators if both models'
test capacities are below `0.05`. The row remains in score files and robustness
tables. This rule is objective and is the reason the primary gated degree axis is
`{1, 2, 3}` while degree 5 remains a reported appendix band.

The secondary product family is not the headline gate family. It uses products of
delayed Legendre targets with `(p1, p2) in {(1, 1), (1, 2), (2, 2)}` and
`tau in {1, 2, 4, 8}`. Anchor tasks are diagnostic only: STM, NARMA-2/5/10,
Mackey-Glass, parity, and Santa Fe when a cache is present.

## Resource Matching

Every scored QRC/QELM pair uses the same frozen physical and readout budget:

- `n_qubits = 5`, `J_s = 1.0`, `h = 0.5`
- shared dissipator and `gamma` for the stage being run
- default weight-2, 3-axis Pauli observables, `D = 45` at `N = 5`
- `n_virtual = n_snapshots = 1`, so readout width is exactly one observable block
- same shot tier, alpha grid, input rows, split rows, and train-only preprocessing

The hard runtime gates are `matching.assert_bundles_matched` and
`matching.assert_features_live`. `assert_bundles_matched` checks the built feature
bundles, not only the intended config, so declared dimension parity cannot mask a
real feature-width mismatch. `assert_features_live` enforces the frozen liveness
floors (`min_col_var = 1e-8`; `min_multiplex_gain = 1.15` when `V > 1`) and records
effective-rank diagnostics. `matching.assert_matched` remains a cheap config-level
pre-flight, but the built-bundle checks are the operative gates.

The primary comparison intentionally fixes `V = 1`. Earlier calibration showed
that QRC virtual-node multiplexing and QELM trajectory snapshots do not create the
same effective-rank gain at these sizes, so matched declared `D` at `V > 1` is not
the clean headline comparison. Any `V > 1` result is exploratory and must be labeled
as such.

## Intended Asymmetry

One asymmetry is structural and deliberately not matched away. QRC injects the input
through an active recurrent Hamiltonian drive throughout the reservoir evolution.
QELM injects input only by state preparation through `encode_window`, then evolves
under a fixed input-independent channel. This isolates recurrent dynamic encoding
with intrinsic memory against static state-preparation encoding with a classical
delay window, under matched qubits, observables, readout dimension, shot budget, and
ridge pipeline.

## Shot-Noise Semantics

Stage 2 uses shot tiers `{infinity, 1e5, 1e4, 1e3}`. The `infinity` tier means exact
expectation values from the simulator. Finite tiers mean each observable expectation
is sampled with the same shots-per-expectation value for QRC and QELM. The physical
shot budget is reported as `S * T * D`, with identical `S`, `T`, and `D` for both
models at a matched cell.

Shot noise does not change target construction, split rows, alpha grids, or model
selection rules. The shot seed comes from `SEED_GRIDS["stage2_shots"]`, and the same
shot tier grid is retained under any runtime shrink.

## Hyperparameter Grids

Hyperparameters are selected on validation rows only, using equal-cardinality grids:

- QRC: `dt in {0.25, 0.5, 1.0}` by `h_in in {0.5, 1.0}` gives 6 candidates.
- QELM: `d in {1, 3, 5}` by two `T_evolve` values from the live window gives
  6 candidates.

The QELM `T_evolve` values are generated per matched channel from
`matching.recommend_T_evolve`: `0.75 * timescale` and `1.25 * timescale`. Since the
returned window is `[0.5 * timescale, 2.0 * timescale]`, both grid points are inside
`[t_lo, t_hi]` at every operating point where the recommendation is finite.

The alpha grid is shared:
`{0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 100}`. Gamma is not tuned in RQ1 or RQ2; it is
the RQ3 axis.

## Split Pipeline

For each input seed, the input stream is generated once and all models and controls
use targets derived from that same stream. Valid rows are the intersection of target
validity, washout, maximum delay, and model first-valid rows. `pipeline.make_splits`
then creates ordered contiguous train/validation/test splits with fractions
`0.6 / 0.2 / 0.2`.

All standardization and readout fitting state is fit on training rows only. Alpha
selection uses validation rows only. Final reported capacity and NRMSE use test rows
only. QRC, QELM, `linear_delay`, and `poly_delay` controls use the same split rows.
The `poly_delay` control is sized to the same readout width `D` as the quantum
bundles; at the frozen `N = 5`, `D = 45` exceeds the degree-2 delay count
`2d + C(d,2) = 20` for `d = 5`, so the degree-2 polynomial control is not truncated.
