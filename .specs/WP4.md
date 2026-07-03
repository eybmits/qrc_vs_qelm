# WP4 — Task families, shared ridge pipeline, classical controls

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Depends on WP1–WP3
(core, QELM, unified API). You may READ
`/Users/superposition/Coding/qrc_dataset_profiler/src/qrc_dataset_profiler/baselines.py`
(split/ridge patterns) and `/Users/superposition/Coding/qrc_relaxation/qrc_relaxation/tasks.py`
(Santa Fe loader, ~line 282).

## Principles

- **Targets are always computed from UNQUANTIZED inputs.**
- **One shared evaluation path** for QRC, QELM, and classical controls: same valid
  indices, same splits, same standardization, same α grid, same metrics. Any
  model-specific branching in evaluation is a spec violation.

## Deliverables

1. `src/qrc_vs_qelm/task_families.py` (core benchmark generators already live in
   `core/tasks.py` — do not duplicate them):
   - `legendre_target(u: np.ndarray, tau: int, degree: int) -> np.ndarray` —
     `y[t] = P_degree(2*u[t-tau] - 1)` (`numpy.polynomial.legendre`), `y[t]=nan`
     for t < tau.
   - `product_target(u, tau1, p1, tau2, p2) -> np.ndarray` — product of two such
     terms; nan where undefined.
   - `build_family_targets(u, taus, degrees) -> dict[str, np.ndarray]` — keys
     `"leg_tau{τ}_p{p}"`; plus `build_product_targets(u, specs)` with keys
     `"prod_tau{τ}_p{p1}{p2}"`.
   - `narma_targets(u) -> dict[str, np.ndarray]` for orders 2/5/10 reusing
     `core.tasks` recursions (input rescaling per the source convention);
     `stm_targets(u, taus)`; `parity_targets(bits, taus)` (binary-input stream).
   - `load_santa_fe(path="data/santa_fe.npy") -> np.ndarray | None` — cached-file
     loader, `None` + warning if absent (no network fetch).
2. `src/qrc_vs_qelm/pipeline.py`:
   - `valid_indices(T: int, washout: int, max_delay: int) -> np.ndarray` — the ONE
     shared mask: `t >= max(washout, max_delay)`; also drops rows where any target
     under evaluation is nan (takes the targets dict).
   - `make_splits(idx: np.ndarray, fractions=(0.6, 0.2, 0.2)) -> Splits` —
     contiguous train/val/test over the valid indices (no shuffling; time series).
   - `standardize_train_only(X, splits) -> np.ndarray` — mean/std from TRAIN rows
     only, applied everywhere; guard σ=0 columns.
   - `evaluate_feature_matrix(X, targets: dict[str, np.ndarray], splits,
     alphas=(1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2)) -> pd.DataFrame` — per target:
     ridge (closed-form, bias column) per α on train → select α on val capacity →
     refit train+val → report test `capacity` and `nrmse` (reuse `core.readout`
     metrics), plus selected α, split sizes, `effective_rank(X_train)` via
     `core.diagnostics` if available (else participation-ratio of singular values —
     check diagnostics.py first).
   - Long-format output columns: `target, alpha, capacity_test, nrmse_test,
     n_train, n_val, n_test, effective_rank, feature_dim`.
3. `src/qrc_vs_qelm/controls.py` — classical matched-dimension baselines:
   - `linear_delay_features(u, d, D) -> np.ndarray` — raw window, zero-padded
     columns up to exactly D.
   - `poly_delay_features(u, d, D) -> np.ndarray` — degree-≤2 polynomial features
     of the window in a DETERMINISTIC order (linear terms, then squares, then cross
     terms lexicographic), truncated or zero-padded to exactly D. Document the order.
   - Both full-length (T rows), consistent with the WP3 full-length contract.
4. Tests (`tests/test_task_families.py`, `tests/test_pipeline.py`, `tests/test_controls.py`):
   - Legendre: empirical orthogonality on u~U[0,1] (large T): E[P_i(x)P_j(x)] ≈ 0
     for i≠j; `leg_tau_p1` equals shifted linear rescale exactly.
   - NARMA-2 against a hand-rolled 10-step reference; parity target correctness.
   - **Fairness test**: build QRC features, QELM features (matched configs, WP3),
     and both controls on one input stream; run `evaluate_feature_matrix` on all
     four with the same targets dict → assert the target vectors and index sets
     used are BIT-IDENTICAL across models (expose them from the pipeline for this).
   - **Leakage guard**: standardization stats computed on train slice only (assert
     changing test rows doesn't change transformed train rows); splits contiguous,
     ordered, disjoint, covering the valid set.
   - Controls: exact dimension D, determinism, zero-pad correctness.
   - nan-handling: targets with leading nans shrink the valid set consistently.
   - Keep runtime small (N ≤ 4, T ≤ 400).

## Acceptance

```
.venv/bin/pytest -q
```
All green. Summary: files, test count, any name deviations from WP1–WP3 code.

## Rules

Follow existing code names where they differ. Explicit rng. No results files.
No commits. Don't touch `.specs/`.
