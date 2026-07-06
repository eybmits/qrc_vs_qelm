# WP-I2 — Information Processing Capacity estimator (Dambre et al. 2012)

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. New module `src/qrc_vs_qelm/ipc.py`
+ tests. Do NOT regress existing tests. This is the "one measurable quantity" of the thesis.

## Why

The study's headline is the **total IPC** of the full QRC vs the windowed QELM. Total IPC
(Dambre 2012) = the sum, over an orthonormal basis of nonlinear-with-memory functions of
the input, of the capacity with which the system's linear readout reconstructs each basis
function. It is bounded by the readout rank (≤ D). The existing Legendre tasks are exactly
the Dambre basis; this WP aggregates them into total IPC with a proper significance cut.

## Deliverables — `src/qrc_vs_qelm/ipc.py`

1. `dambre_basis(max_delay, max_degree, max_total_degree, max_functions) -> list[BasisSpec]`
   — enumerate basis functions = products of Legendre polynomials `Π_i P_{d_i}(2·u_{t−τ_i}−1)`
   over sets of DISTINCT delays `τ_i ∈ {0..max_delay}` with per-factor degree `d_i ≥ 1`,
   total degree `Σ d_i ≤ max_total_degree` and per-factor degree ≤ `max_degree`. Enumerate
   degree-by-degree (degree 1 = single delays; degree ≥ 2 = distinct-delay products),
   capped at `max_functions` (record if the cap truncates — log it, do not silently drop).
   `BasisSpec` carries the (τ_i, d_i) list, total degree, and a stable key.
2. `basis_target(u, spec) -> np.ndarray` — build the product target for one BasisSpec,
   REUSING `task_families.legendre_target` per factor (product of the per-factor Legendre
   targets; leading NaNs up to max τ). (Generalize `task_families.product_target` if a
   clean n-ary helper is cleaner — but keep task_families' existing 2-factor API intact.)
3. `significance_threshold(features, u, basis, splits, alphas, *, n_surrogates, rng) -> float`
   — shuffle-surrogate noise floor: for each surrogate, evaluate the SYSTEM's capacity
   against basis targets built from a temporally-shuffled input copy (same marginal, no
   temporal structure → true capacity ≈ 0); return a high percentile (e.g. max, or 99th)
   of the surrogate capacities. Deterministic given `rng`.
4. `total_ipc(features, u, basis, splits, alphas, *, threshold=None, n_surrogates=... , rng)
   -> dict` — for each BasisSpec compute capacity (REUSE `pipeline.evaluate_feature_matrix`
   or `core.readout.capacity` via the same train/val/α-select/test path), zero out
   components below `threshold` (compute it via `significance_threshold` if not given),
   and return `{"total_ipc", "mc"(Σ degree-1), "nc"(Σ degree≥2), "n_significant",
   "n_basis", "threshold", "truncated": bool, "by_degree": {deg: sum}}`.

## Tests (`tests/test_ipc.py`)

- **Fundamental bound**: for a random feature matrix with D columns, `total_ipc ≤ rank(features) ≤ D`
  (Dambre's theorem — the single most important invariant). Assert it on a real small QRC
  feature matrix and on a synthetic full-rank matrix.
- **Linear-reservoir reference**: a purely linear delay-line feature set (features = lagged
  inputs) has IPC concentrated in degree-1 (MC ≈ number of independent lags), NC ≈ 0.
- **Significance cut**: with pure-noise features (no relation to inputs), `total_ipc ≈ 0`
  after thresholding (the surrogate floor removes spurious capacity).
- **Determinism**: same rng → identical total_ipc; basis enumeration is order-stable.
- Keep runtime small (N ≤ 4, T ≤ 600, modest basis caps).

## Acceptance

```
.venv/bin/pytest -q
```
All green. Summary: basis size at the default caps, the bound-test result, and the
noise-features IPC≈0 result.

## Rules

Reuse `task_families` (Legendre) + `pipeline`/`core.readout` capacity — do NOT reimplement
capacity or Legendre. Explicit rng everywhere (surrogates, shuffles). No results files.
No commits. Don't touch `.specs/`.
