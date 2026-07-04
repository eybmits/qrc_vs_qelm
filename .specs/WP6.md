# WP6 — Validation gate + timing pilot (HARD GATE before any main run)

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Depends on WP1–WP5.
You may READ `/Users/superposition/Coding/qrc_dissipation_engineering/results/`
(stored reference numbers). This WP is the reproduction gate: NO stage runs happen
until `scripts/run_validation.py` exits 0.

## Deliverables

1. `scripts/run_validation.py` — writes `results_validation/validation.json` with a
   top-level `"passed": bool` and a per-check breakdown. Exit code 0 iff all pass.
   CLI: `--profile smoke|full` (smoke uses shorter sequences / fewer seeds for CI).
   Three checks:

   **V1 — QRC physics reproduction** (the port must reproduce source physics through
   OUR pipeline). At the documented operating point N=5, h=0.5, dt=0.5, uniform-local
   continuous dissipation (the "CD_paper (uniform local)" method), s_gamma=0.3:
   - Compute short-term memory capacity MC = Σ_τ capacity(delay τ) over τ=1..15 using
     `task_families.stm_targets` + `pipeline.evaluate_feature_matrix`, averaged over
     the stage1 coupling seeds (full) or 3 seeds (smoke).
   - Reference: `results/best_vs_best.json` → `results["CD_paper (uniform local)"]`
     `stm_test_MC = 8.914 ± 1.210`. PASS if our mean MC ∈ [ref − 2·std, ref + 2·std]
     (i.e. [6.49, 11.33]). Also compute NARMA-10 NMSE and check it is < 0.7 (ref
     0.479 ± 0.139), a directional sanity bound.
   - Ordering sub-check from `results/confirm_stm.json`: CD capacity ≥ FN capacity at
     the short delays (τ ≤ 5) on average (dissipation-engineered reservoir must beat
     the Fuji-Nakajima baseline where the paper claims it does). Build the FN baseline
     via `core.reservoirs.FujiNakajimaReservoir` at the same operating point.
   Record measured vs reference (value, tolerance band, pass) for each.

   **V2 — QELM sanity** (the static map behaves as a delay embedding must). At d=N=5,
   γ=0, a `T_evolve` from `matching.recommend_T_evolve` (live regime):
   - On the Legendre p=1 family, `capacity(QELM, τ)` must be ≥ 0.99 for τ < d and
     ≈ 0 (≤ 0.05) for τ ≥ d + a margin (memoryless map can only see the window).
     This is the falsifiable signature of "static map on a d-delay embedding".
   - `assert_features_live` passes on the QELM bundle at this config (no collapse).

   **V3 — matched-budget audit**. For one frozen QRC/QELM config pair at N=5 (the
   intended primary operating point): config-level `matching.assert_matched` passes AND
   the bundle-level `matching.assert_bundles_matched(qrc_bundle, qelm_bundle)` passes
   (the real D-parity gate — reads actual built dimensions); `physical_shot_budget`
   equal on both sides; `matching.assert_features_live` passes on BOTH bundles; report
   `effective_dimension` + multiplex `gain` for both (logged transparency, not pass/fail).

2. `scripts/bench_configs.py` — timing pilot. For each config class
   (`qrc_n4`, `qrc_n5`, `qelm_n5`, `qrc_n6_sparse`, `parity_n5`), measure wall-clock
   median + p90 over a few feature builds at smoke length; write
   `results_freeze/timing_pilot.json` with per-class medians, machine metadata
   (platform, cpu count), and `project_runtime(timing: dict, cells: dict) -> dict`
   that projects total stage runtime from per-class medians × planned cell counts.
   This output SIZES the frozen row caps in WP7 — it does not itself freeze anything.

   IMPORTANT COST FINDING (measured during WP3): `build_qrc_features` builds one `expm`
   propagator PER quantized input level, so cost scales with `quantize_levels × 4^N`.
   At `quantize_levels=201`, N=5 is ~1 min/build and N=6 is many minutes — the propagator
   cache is NOT free. The timing pilot MUST sweep `quantize_levels ∈ {33, 65, 129, 201}`
   at N=5 and N=6 and report the build-time vs level count, so WP7 can freeze a
   level count that scales DOWN with N (attribution-audit precedent: 33 at N=6). Also
   report a fidelity check: capacity/NRMSE at 33 vs 201 levels on STM at N=5, so the
   coarser grid's accuracy cost is quantified before it is frozen.

3. Tests (`tests/test_validation.py`):
   - `run_validation.py --profile smoke` exits 0 and writes a well-formed
     `validation.json` with `passed=true` (run at reduced N/seeds so it finishes in
     CI-time; if smoke can't hit the N=5 reproduction bands cheaply, the smoke profile
     may assert only V2+V3 and a RELAXED V1 directional check — document this).
   - `project_runtime` arithmetic is correct on a synthetic timing dict.
   - V2's memory-cutoff signature holds (this is also a strong end-to-end integration
     test of QELM + pipeline).

## Acceptance

```
.venv/bin/pytest -q
.venv/bin/python scripts/run_validation.py --profile smoke   # exits 0
.venv/bin/python scripts/bench_configs.py                     # writes timing_pilot.json
```

## Rules

Reference numbers are READ from the source results JSONs at runtime (record the file
path + the value used in validation.json) — do not hardcode magic numbers without
provenance. Explicit rng. `results_validation/` and `results_freeze/` are gitignored.
No commits. Don't touch `.specs/`. Full-profile V1 must pass before WP7 freeze.
