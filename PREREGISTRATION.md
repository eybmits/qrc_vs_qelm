# Frozen Commitments

This document freezes the comparison commitments for the `stage1-frozen` tag. The
user applies the tag after this work package lands. Methods are defined in
[PROTOCOL.md](PROTOCOL.md); machine-readable configuration is in
`data/frozen/frozen_v1_config.json` and `configs/full.yaml`.

## Frozen Operating Point

- Primary: `N = 5`, `gamma = 0.3`, `dt = 0.5`, `h = 0.5`, `J_s = 1.0`,
  `dissipator = local_loss`, `V = 1`, `quantize_levels = 33`.
- Observables: default weight-2, 3-axis Pauli set, `D = 45`.
- Primary RQ1 grid: Legendre delay with
  `tau = {0, 1, 2, 3, 5, 8, 10, 12, 16}` and `p = {1, 2, 3}`.
- Robustness appendix: the full `p = {1, 2, 3, 5}` grid is run and reported.
- Primary gated set: the frozen degree-{1,2,3} family gives 27 `family_cell =
  (tau, p)` units for G1a/G1b/G1c.
- Uninformative-cell rule: as a secondary safeguard, exclude a confirmatory
  `family_cell` from G1b/G1c fractions iff both models' mean test capacities
  across that cell's seed replicates are below `0.05`. This is a family-cell
  mean-over-seeds rule, not a per-seed row rule; still score and report all rows,
  and report gate fractions with and without the exclusion.
- Secondary product family: `(p1, p2) in {(1,1), (1,2), (2,2)}` and
  `tau = {1, 2, 4, 8}`.
- Anchors are not gated: STM, NARMA-2/5/10, Mackey-Glass, parity, Santa Fe if cached.
- Shot tiers: `{infinity, 1e5, 1e4, 1e3}`.
- RQ3 grid: `gamma = {0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5}` by
  `dissipator = {local_loss, collective_loss}`.
- Robustness replication: `N = 4` coarse grid; `N = 6` optional sparse anchor only.

<!-- GATES:BEGIN -->
## Gates

| Gate | Stage | Kind | Description | Threshold | Kill | Required |
| --- | ---: | --- | --- | --- | --- | --- |
| G1a | 1 | per_cell_significance | Per-family-cell paired capacity delta has a CI excluding zero with BH-FDR q <= 0.10. | 0.0 | False | False |
| G1b | 1 | headline_two_regime | Both significant QRC-favored and significant QELM-favored family-cell regimes occupy at least 20% of cells. | {'min_qrc_fraction': 0.2, 'min_qelm_fraction': 0.2} | False | True |
| G1c | 1 | kill_qelm_trivial | QELM beats the linear-delay control on at least 25% of p>=2 family cells. | 0.25 | True | True |
| G2a | 2 | shot_sensitivity | The QRC-favored cell count changes between S=infinity and S=1e4 with a bootstrap CI excluding zero. | 0.0 | False | True |
| G3a | 3 | gamma_tradeoff | MC(gamma) is non-increasing, NC(gamma) is non-decreasing, and a QELM>=QRC crossover exists. | 0.0 | False | True |
<!-- GATES:END -->

## Seed Grids

Seed grids are frozen in `qrc_vs_qelm.seeds` and referenced by name in config:

| Stage | Grid | Count | Contents |
| --- | --- | ---: | --- |
| Stage 1 | `stage1_full` | 30 | input seeds `0..9` by coupling seeds `314159, 271828, 161803` |
| Stage 2 | `stage2_shots` | 9 | input seed `0` by the three stage-1 couplings and shot seeds `0, 1, 2` |
| Stage 3 | `stage3_gamma` | 10 | input seeds `0..4` by coupling seeds `314159, 271828` |

The design-pilot seed grid is disjoint from these confirmatory grids and is not a
confirmatory source.

## Row Caps

Full-profile row caps are `T = 3000`, `washout = 100`, split fractions
`0.6 / 0.2 / 0.2`. Smoke uses `T = 400`, `washout = 100`.

Runtime projection used `scripts.bench_configs.project_runtime` with
`results_freeze/timing_pilot.json` at `quantize_levels = 33`. Projection counts
multiply target-family cells, seed cells, and per-model HP candidates; anchors are
diagnostic rows and are not gate-budget cells.

| Stage | Count basis | Initial hours | Active hours | Shrink |
| --- | --- | ---: | ---: | --- |
| Stage 1 | 48 budgeted target cells x 30 seed cells x 6 HP candidates/model | 12.24 | 9.79 | Cut input seeds to `0..7`; kept all coupling seeds and the secondary product family |
| Stage 2 | 27 gated cells x 9 seed cells x 4 shot tiers | 1.38 | 1.38 | None |
| Stage 3 | 27 gated cells x 10 seed cells x 14 gamma/dissipator points | 5.36 | 5.36 | None |

The active stage-1 cap is therefore 24 seed cells: input seeds `0..7` by all three
stage-1 coupling seeds.

## Shrink Rules

Each stage has a 10 h projected-runtime budget. If a stage exceeds budget, shrink in
this order:

1. Cut input seeds first, never below 6 kept input seeds when a stage has enough
   input seeds to do so.
2. Cut the secondary product family.

Never cut gates, shot tiers, the gamma grid, or the `tau`/degree axes. WP7 projection
applied only the first shrink step to stage 1.

## Freeze Checklist

- [x] Gate thresholds are frozen before confirmatory runs: G1b 20%/20%, G1c 25%,
  `tie_band = 0.02`, BH-FDR `q <= 0.10`.
- [x] Threshold rationale: 20%/20% requires material two-regime occupancy; 25% G1c
  prevents a trivial QELM-vs-linear-delay explanation; `tie_band = 0.02` is the
  material capacity-delta tolerance; FDR `q <= 0.10` controls family-level discovery
  while retaining pilot-scale sensitivity.
- [x] G1c counts significant QELM-over-linear-delay wins only, requiring the paired
  bootstrap CI to exclude 0.
- [x] `poly_delay` is not starved: at `N = 5`, `D = 45`, while the full degree-2
  count for `d = 5` is `2d + C(d,2) = 20`.
- [x] QELM `T_evolve` grid uses `0.75 * timescale` and `1.25 * timescale`, both
  inside the `recommend_T_evolve` window `[0.5 * timescale, 2.0 * timescale]`.
- [x] Confirmatory seeds remain disjoint from design-pilot seeds.

## Amendment Log

- γ swept 0 → 0.1 → {0.2,0.3,0.5} → {0.25,0.3} on DISJOINT pilot seeds; γ=0.3 chosen
  as the QRC-memory optimum (QRC region maximal, QELM still present).
- τ axis extended 8 → 16 (diagnostic-driven; QRC's memory band sits at τ≥5).
- Pilot seeds expanded 6 → 15 to test power; result: ties are GENUINE not
  power-limited (ties_power_limited=0 at 15 seeds).
- Degree axis set to {1,2,3} via the uninformative-cell rule (degree-5 both-fail).
- Full-grid pilot result at γ=0.3: QRC 19.4% / QELM 19.4% (both-fail p=5 dilutes);
  informative-subgrid: QRC 25.9% / QELM 22.2% → GO. Confirmatory seeds UNTOUCHED.
- WP7 freeze checklist items from `WP7_prereg_notes` verified and checked off:
  poly_delay D floor, T_evolve grid inside `[t_lo,t_hi]`, G1c significance, and
  threshold rationale.
- WP8a clarification: the primary gated family remains the frozen 27
  degree-{1,2,3} cells. The capacity<0.05 uninformative-cell safeguard is defined
  at `family_cell = (tau, p)` granularity using each model's mean test capacity
  over seed replicates, and Stage 1 reports G1b/G1c fractions with and without
  the exclusion.
- WP8b amendment: RQ3 QELM `T_evolve` corrected PRE-EXECUTION; no Stage-3 data
  existed. Stage 3 now fixes QELM `T_evolve` at the gamma=0.3 operating-point
  `recommend_T_evolve` value across the gamma sweep. The prior per-gamma
  recommendation was unmatched to QRC's fixed `dt` and could make QELM
  gamma-insensitive. Stage-3 dead-feature points are recorded with
  `features_dead = True`, zero per-target capacity, and `MC = NC = 0`.
- POST-RUN G2a evaluation bug fix (verdict-neutral disclosure): the frozen code's
  `gates._is_infinite_shots` did not recognize the `∞` (U+221E) token that
  `run_stage2` wrote for the S=∞ tier, so G2a never saw the infinity rows and
  returned `SHOT_INSENSITIVE` via the missing-rows path. Fixed the matcher (and
  normalized future output to `infinity`) and RE-EVALUATED G2a on the SAME saved
  Stage-2 scores — no re-simulation. This is a fix to gate *evaluation*, not the
  gate *definition/threshold* (unchanged: compare S=∞ vs S=1e4 QRC-favored count,
  bootstrap CI must exclude 0). Corrected verdict is STILL `SHOT_INSENSITIVE`, now
  legitimately: the QRC-favored count shifts 9.0(∞)→7.0(1e4)→5.0(1e3) but the
  bootstrap CI on the ∞-vs-1e4 shift is [0.0, 3.78], which does not exclude 0 at
  the 24-seed budget (suggestive but underpowered).
