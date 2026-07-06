# WP-I2-fix — enforce the Dambre bound; control cumulative noise; shared-protocol contract

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. The WP-I2 adversarial review
(reconsider) reproduced a **violation of the estimator's own theorem** `total_ipc ≤
rank(features) ≤ D` at realistic T (200–600) with richer bases (20/20 trials over at
T=220). Root cause: each basis component's capacity is estimated INDEPENDENTLY (own α on
val, scored on a small test slice); summing many independently-noisy, α-optimized test
correlations lets sampling noise stack past the true orthogonal budget — and it inflates
ASYMMETRICALLY (a low-capacity model sits closer to its bound → biases Δ(k)). Do NOT
regress the 172 tests; the architecture (shared basis, shuffle-surrogate threshold, reuse
`pipeline`) is right — fix the statistics.

## Fix 1 — PER-DEGREE significance threshold + cumulative-noise control (the real fix)

Both reviewers traced the failure to the single global `max`-over-all-surrogate-capacities
threshold, which pools wildly different noise scales (degree-1 vs degree-3 cross-delay
products) — so higher-order products regressed against correlated lag features slip
through. Fix:
- Compute the surrogate noise floor **per degree class** (or per number-of-factors): pool
  surrogate capacities within each degree, threshold each class separately. This directly
  fixes the demonstrated bilinear-cross-term-on-a-linear-space leak.
- Additionally control the **cumulative** retained noise: choose thresholds so the TOTAL
  retained capacity on shuffled-input surrogates is ≤ a small ε (not just per component).
- Apply the SAME protocol (same per-degree θ-selection, same `n_surrogates`, same basis) to
  both models. Deterministic given rng.

## Fix 2 — theorem-enforced clamp + diagnostic (backstop)

`total_ipc` must return `total_ipc = min(Σ significant capacities, effective_rank(features))`
(the bound is a theorem; exceeding it is estimation error). Also return the **raw**
unclamped sum and `bound_excess = max(0, raw_sum − rank)` as diagnostics, and LOG when the
clamp triggers. `effective_rank` here = the numerical rank of the (centered) feature matrix
used for reconstruction.

## Fix 3 — stress-test the bound via the DEFAULT codepath (not the toy)

The shipped bound test used `threshold=0.0` (bypassing the surrogate floor) — it never
tested the codepath users run. New tests MUST use `threshold=None` (the default,
surrogate-derived path):
- **Pin the exact reproductions** both reviewers gave: (a) `T≈150-220`, `linear_delay_features`
  (strictly rank-D), **mixed-degree basis** (`max_total_degree ≥ 2-3`) with default
  `n_surrogates` → assert the bilinear cross-delay term is zeroed AND `total_ipc ≤ rank + eps`
  across ≥ 20 seeds; (b) a small QRC feature set, same regime.
- **Production-scale**: a few trials at T ∈ {200, 600} with the WP-I3 default basis
  (max_delay~20, max_total_degree~4, up to ~600 functions) → returned (clamped)
  `total_ipc ≤ rank + 1e-9` in ALL trials, RAW excess small after Fix 1.

## Fix 6 — cleanups (from code review)

- Remove the dead `basis_target` `BasisSpec(**spec)` mapping fallback (not in the signature).
- Add `dambre_basis` input validation (reject `max_delay<0`, `max_degree<1`, `max_total_degree<1`).

## Fix 4 — per-delay diagnostic + truncation transparency

`total_ipc` output adds `by_delay` (capacity summed per max-delay-of-component) alongside
`by_degree`, and a `basis_edge_capacity` = fraction of retained capacity in components at
the largest delay/degree in the basis. LOG a warning if `basis_edge_capacity` is
non-negligible (the basis caps may be too small — real capacity beyond `max_delay`/
`max_total_degree` is being missed, which would under-count a long-memory model).

## Fix 5 — document + prepare the shared-protocol / shared-splits contract for WP-I3

- Docstring on `total_ipc`: for a fair Δ, the caller MUST pass the SAME `basis`, `alphas`,
  `n_surrogates`, and threshold-selection to both models.
- Since QRC and TD-QELM(k) have different `first_valid` (washout vs k−1), "same splits" is
  not literally achievable across a k-sweep. Add a helper (or document the contract) so
  WP-I3 uses **one shared valid-index set = the intersection across the most-restrictive k
  (largest first_valid) in the sweep**, so every model at every k is scored on IDENTICAL
  rows. Add a test that the shared-rows path gives QRC and QELM bit-identical index sets.

## Acceptance

```
.venv/bin/pytest -q
```
All green. Summary: the bound now holds (clamped) at T∈{200,600} production-basis stress
test with raw excess small after the cumulative-noise threshold; the by_delay diagnostic;
and the shared-rows contract test.

## Rules

Fix the statistics, keep the architecture. Reuse `pipeline`/`core.readout`. Explicit rng.
No commits. Don't touch `.specs/`.
