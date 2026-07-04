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
     bundles at the frozen matched config (couplings from `coupling_seed`), γ=0.3,
     quantize_levels=33 (frozen).
   - HP SELECTION (frozen protocol): sweep the equal-cardinality HP grids from config
     (QRC dt×h_in = 6; QELM d×T_evolve = 6, T_evolve values inside the recommended live
     window), select each model's HP on VALIDATION capacity (mean over the phase-diagram
     targets), refit, score test. Same alpha grid, same splits, same val criterion for
     both — this is the fair per-model tuning. γ is NOT tuned.
   - GATE: `assert_bundles_matched` + `assert_features_live` on both selected bundles — a
     cell that fails is a hard error (do not silently score collapsed features), recorded
     in the manifest.
   - Apply the frozen UNINFORMATIVE-CELL EXCLUSION when forming gate inputs: a cell where
     BOTH models' test capacity < 0.05 is flagged `uninformative` and excluded from the
     G1b/G1c fractions (but still recorded in the scores CSV). This is what realizes the
     degree-{1,2,3} informative axis; the full {1,2,3,5} grid is still scored + saved for
     the robustness appendix.
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
5. FDR family-integrity guard (WP5 adversarial concern 2): the BH-FDR correction in
   `gates.evaluate_stage1_gates` is applied ONCE across all phase-diagram cells, which
   is correct ONLY if the scores frame passed to it contains the full confirmatory cell
   set. Before evaluating gates, ASSERT `df["family_cell"].nunique()` equals the
   expected frozen grid size (from config) — so the FDR family can never silently split
   by calling gates per target-family. Score linear_delay + poly_delay controls per seed
   cell (G1c needs per-(cell,seed) `capacity_linear_delay`).

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
