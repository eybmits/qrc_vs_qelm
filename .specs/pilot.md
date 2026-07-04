# Design pilot — confirm both regimes exist BEFORE the pre-registration freeze

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Depends on WP1–WP6.
This is the go/no-go gate for the whole study: the headline deliverable is a phase
diagram with BOTH a QRC-winning region and a QELM-winning region. If one model
dominates everywhere at pilot scale, we must learn that NOW (and retune τ/p ranges or
reframe) — not after freezing and burning the confirmatory compute. Uses the DISJOINT
`pilot` seed grid so confirmatory seeds stay untouched (pre-registration integrity).

## Deliverable

`scripts/design_pilot.py` — writes `results_pilot/pilot.json` + a printed summary.
CLI: `--out results_pilot`, `--profile smoke|full` (pilot is always coarse/cheap).

Procedure:
1. Fix the primary operating point: N=5, V=1, default weight-2 3-axis observables,
   γ=0 (add a γ=0.1 slice if cheap). QRC hyperparameters and QELM `T_evolve` chosen
   via the existing selection path; QELM `T_evolve` MUST come from
   `matching.recommend_T_evolve` (live regime), and `assert_bundles_matched` +
   `assert_features_live` MUST pass for every pilot config (assert this — a pilot that
   silently ran on collapsed/ mismatched features tells us nothing).
2. Coarse phase-diagram subgrid: Legendre delay family τ ∈ {0,1,2,3,5,8} × degree
   p ∈ {1,2,3,5} (24 cells — spans below/at/above the QELM hard-memory cutoff d and
   low→high nonlinearity). Targets from `task_families.build_family_targets`.
3. For each `pilot` seed cell: build QRC + QELM feature bundles once (cache), evaluate
   ALL 24 targets via `pipeline.evaluate_feature_matrix`, collect per-cell test
   capacity for both models. Include the `linear_delay` and `poly_delay` controls
   (needed to sanity-check the G1c "QELM must beat linear delay" kill-rule at pilot
   scale — if QELM already can't beat linear_delay here, that is a red flag worth
   knowing before freeze).
4. Aggregate across pilot seeds: per cell compute mean Δ = cap_QRC − cap_QELM and the
   `stats.label_cell` label (QRC / QELM / tie).

## Go/no-go report (printed + in pilot.json)

- `frac_qrc`, `frac_qelm`, `frac_tie` over the 24 cells.
- **GO** iff `frac_qrc ≥ 0.20` AND `frac_qelm ≥ 0.20` (mirrors gate G1b: both regimes
  materially present). Print the (τ,p) coordinates of each region so the boundary shape
  is visible.
- **Diagnostics on NO-GO**: if QRC dominates everywhere → QELM likely under-powered
  (check: does QELM beat linear_delay anywhere? is T_evolve mid-window? try larger d).
  If QELM dominates everywhere → task family may be too memory-shallow (extend τ range).
  If mostly ties → capacity ceiling/floor saturation (adjust p range / sequence length).
  Emit a concrete suggested τ/p-range adjustment so the freeze decision is informed.
- The QRC-vs-QELM boundary should track intuition: QELM strong for τ < d and low p;
  QRC strong for τ ≥ d (beyond the delay window) and where cross-time nonlinearity
  matters. Flag if the observed boundary contradicts this (possible bug, not physics).

## Rules

Pilot results NEVER enter the confirmatory analysis — they inform τ/p ranges and the
freeze decision only, and any range change is logged in the PREREGISTRATION amendment
log as design-stage. `results_pilot/` is gitignored. Explicit rng. No commits.
Don't touch `.specs/`. This is a Codex-built script that I (orchestrator) then RUN and
interpret to make the GO/NO-GO freeze decision.
