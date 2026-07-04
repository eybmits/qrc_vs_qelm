# WP8b — Stage-2 (RQ2 shot tiers) + Stage-3 (RQ3 dissipation sweep) runners

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Depends on the committed
WP8a (`run_stage1.py`, `manifests.py`, `checkpointing.py`) and WP3 `shots.add_shot_noise`.
REUSE WP8a helpers (`_build_targets`, `_score_*`, `_selected_or_cached_bundle`,
`_feature_cache_path`, manifest/checkpoint infra) — do not reimplement them.

## Stage-2 — `src/qrc_vs_qelm/run_stage2.py` (RQ2: shot-budget scaling)

Stage-2 does NO new simulation: it reuses the stage-1 selected-HP feature bundles.
- CLI: `--stage1-out` (dir with stage-1 cached features + manifest), `--out`,
  `--profile`, `--allow-dirty`. Verify the stage-1 manifest's feature-cache sha256s
  match before use (fail loudly if stage-1 features changed).
- For each shot tier in the frozen `shot_tiers` ({∞,1e5,1e4,1e3}) × each `shot_seed` in
  `stage2_shots`: load the cached QRC/QELM/control feature matrices, apply
  `shots.add_shot_noise(features, shots, rng(shot_seed))` (∞ = passthrough), then
  RE-SELECT alpha on validation per tier per model (regularization compensates noise —
  this is the one thing that re-runs; structural HPs stay as stage-1 selected), rescore
  the phase-diagram + control targets.
- A test MUST assert stage-2 imports/calls NO simulator (core/reservoirs/qelm channel)
  functions — it only touches cached features + ridge. (Grep-style guard.)
- Output: `stage2_scores.csv` (adds `shots`, `shot_seed` columns) + manifest.
- Gate G2a (shot-boundary shift) is evaluated in WP9, not here — stage-2 just produces
  the per-tier scores.

## Stage-3 — `src/qrc_vs_qelm/run_stage3.py` (RQ3: dissipation memory-vs-nonlinearity)

Stage-3 rebuilds features because γ changes the channel.
- CLI: `--out`, `--profile`, `--allow-dirty`, `--checkpoint-every`.
- For each γ in the frozen `gamma_grid` ({0.005…0.5}) × dissipator in `dissipator_grid`
  ({local_loss, collective_loss}) × seed cell in `stage3_gamma`: build QRC + QELM
  bundles at that (γ, dissipator) with the OTHER structural HPs FIXED at the frozen
  operating values (dt=0.5, h_in=0.5, d=5, T_evolve from recommend_T_evolve at that γ) —
  so the tradeoff is attributable to γ, not HP re-selection; re-select alpha only.
  assert_bundles_matched + assert_features_live per point (both models channel-matched
  at each γ). QELM T_evolve MUST come from recommend_T_evolve at each γ (live regime).
- Compute per (γ, dissipator, seed, model): **MC** = Σ capacity over the p=1 Legendre
  delays (memory capacity) and **NC** = Σ capacity over p≥2 cells (nonlinear capacity),
  plus per-cell capacity for the phase-diagram delta.
- Output: `stage3_scores.csv` (adds `gamma`, `dissipator`, `MC`, `NC` columns) + manifest.

## Common

- Deterministic (rng only from SeedCell/shot_seed), checkpointed/resumable, git+seed
  manifest with feature-cache hashes, runtime-budget guard with the pre-registered
  shrink message.
- Tests (`tests/test_run_stage2.py`, `tests/test_run_stage3.py`): smoke end-to-end exit
  0 + well-formed CSV/manifest; determinism (two smoke runs bit-identical); stage-2
  no-simulator guard; stage-3 MC/NC columns present and MC decreasing / NC
  non-decreasing on a synthetic monotone fixture.

## Acceptance

```
.venv/bin/pytest -q
PYTHONPATH=src .venv/bin/python -m qrc_vs_qelm.run_stage2 --profile smoke --stage1-out results_stage1_smoke --out results_stage2_smoke   # exit 0
PYTHONPATH=src .venv/bin/python -m qrc_vs_qelm.run_stage3 --profile smoke --out results_stage3_smoke   # exit 0
```

## Rules

`results_*/` gitignored. No commits. Don't touch `.specs/`. Follow WP8a as-built helper
names; note deviations.
