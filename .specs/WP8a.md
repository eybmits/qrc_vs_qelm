# WP8a — Stage-1 runner (RQ1 phase diagram) + manifest/checkpoint infra

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Depends on WP1–WP6 (needs
WP5 `seeds`/`stats` and WP6 validation to exist). You may READ
`/Users/superposition/Coding/qrc_attribution_audit/src/qrc_attribution_audit/`
(`manifests.py`, `checkpointing.py`, `run_stage0.py`) as port sources.

## Real APIs to use (verified as-built)

- `models.api.build_features(cfg, inputs, coupling_seed, observables=None) -> FeatureBundle`
  (fields: `features (T,D)`, `model`, `n_obs`, `n_multiplex`, `first_valid`, `couplings`).
- `matching.assert_bundles_matched(qrc_bundle, qelm_bundle)`,
  `matching.assert_features_live(bundle)`, `matching.effective_dimension(features,
  min_col_var=…)`, `matching.multiplex_efficacy(features, n_multiplex, n_obs)`.
- `pipeline.valid_indices(...)`, `pipeline.make_splits(...)`,
  `pipeline.standardize_train_only(X, splits)`,
  `pipeline.evaluate_feature_matrix(X, targets, splits, alphas) -> pd.DataFrame`
  (long-format; `df.attrs["evaluation_context"]`).
- `task_families.build_family_targets(u, taus, degrees)`,
  `build_product_targets(u, specs)`, `narma_targets(u)`, `stm_targets(u, taus)`,
  `load_santa_fe(...)`; `controls.linear_delay_features(u, d, D)`,
  `controls.poly_delay_features(u, d, D)`.
- `seeds.SEED_GRIDS[name]` of `SeedCell(input_seed, coupling_seed, shot_seed)`,
  `seeds.grid_manifest(name)`. (Follow the as-built field names if they differ.)

## Part 0 — port reproducibility infra (verbatim adapt)

Port to `src/qrc_vs_qelm/`:
- `manifests.py`: `git_state`, `sha256_file`, `sha256_frame`, `write_manifest`,
  `require_clean_tree`, `_package_versions` (adjust package list to this repo's deps).
- `checkpointing.py`: `run_chunked` (+ its helpers) — resumable chunked execution.
Port their tests too (adjust imports). These must pass before the runner is trusted.

## Deliverables

1. `src/qrc_vs_qelm/run_stage1.py` with `def main(argv=None) -> int`. CLI:
   `--out` (required, dir), `--profile {smoke,full}`, `--seed-grid` (default
   stage1_full for full / stage1_smoke for smoke), `--checkpoint-every` (default 20),
   `--allow-dirty`. Config from `configs/{smoke,full}.yaml` (N, T, washout, HP grids,
   observable set, Legendre τ/degree grids, primary V=1).
2. Procedure per seed cell (checkpointed via `run_chunked`, keyed by cell id):
   - Draw the i.i.d. input stream from `input_seed` (frozen). Build QRC + QELM V=1
     bundles at the frozen matched config (couplings from `coupling_seed`).
   - GATE: `assert_bundles_matched` + `assert_features_live` on both — a cell that
     fails is a hard error (do not silently score collapsed features), recorded in the
     manifest.
   - Cache each bundle's feature matrix to `<out>/features/<model>_<cell>.npz` and
     record its `sha256_file` in the manifest (resumable: skip rebuild if the npz +
     hash already present).
   - Targets: `build_family_targets` (the primary phase-diagram grid) +
     `build_product_targets` (secondary) + anchors (`narma_targets`, `stm_targets`,
     Mackey-Glass via core, Santa Fe if cached). Controls: `linear_delay_features`,
     `poly_delay_features` sized to the SAME D as the bundles.
   - One shared `valid_indices` + `make_splits` across QRC/QELM/controls (identical
     rows). Score all four via `evaluate_feature_matrix`. Record per (cell, model,
     target): capacity_test, nrmse_test, alpha, effective_rank + multiplex gain, D.
3. Outputs in `--out`: `stage1_scores.csv` (long format, all cells/models/targets),
   `stage1_manifest.json` (git state, seed-grid manifest, per-cell feature-cache
   sha256s, config snapshot, runtime, package versions), `checkpoints/`.
4. Runtime-budget guard: at startup, project runtime (reuse WP6 `project_runtime` if
   importable) and abort with the pre-registered shrink message if over the stage
   budget — cut input seeds first, then the secondary product family; never gates/cells.

## Tests (`tests/test_run_stage1.py`)

- `main(["--profile","smoke","--out",tmp])` exits 0 end-to-end at reduced N/T/cells;
  writes well-formed scores CSV + manifest with the required columns/keys.
- DETERMINISM: two smoke runs → `stage1_scores.csv` bit-identical (sha256_frame equal).
- Resumability: interrupt (partial checkpoint) then re-run → completes, same result.
- A deliberately mismatched config pair makes the run raise (fairness gate active).

## Acceptance

```
.venv/bin/pytest -q
.venv/bin/python -m qrc_vs_qelm.run_stage1 --profile smoke --out results_stage1_smoke  # exits 0
```

## Rules

`results_*/` gitignored. Explicit rng from SeedCell only — no global RNG, no
`Date.now()`-style nondeterminism. No commits. Don't touch `.specs/`. Follow as-built
names where they differ from this spec; note deviations.
