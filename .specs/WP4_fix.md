# WP4-fix — pipeline correctness closeout (then WP4 commits)

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Closes the WP4 dual-review
findings that affect the correctness of numbers WP6+ will produce. Do NOT regress the
77 passing tests. Keep `core/` (the verbatim port) pristine unless a fix explicitly says
to touch it.

## Fix 1 — unpenalized intercept in the pipeline ridge (affects nrmse; fix-first)

`core.readout.train_readout` regularizes the bias column along with features. At high
alpha the intercept shrinks, degrading `nrmse` for nonzero-mean targets (NARMA has a
+0.1 offset) while `capacity` barely moves — and since `evaluate_feature_matrix` selects
alpha on `capacity_val` only, a high-alpha model can be chosen while its `nrmse_test` is
silently degraded. Standard practice is to NOT penalize the intercept.
- Give the PIPELINE its own ridge (in `pipeline.py`) with an unpenalized intercept:
  center X and y on TRAIN means, solve the penalized least-squares on centered data,
  recover intercept = mean_y_train − mean_X_train @ w (equivalent to sklearn
  `fit_intercept=True`). Use it in `_fit_predict`. Do NOT modify `core/readout.py`
  (it stays the verbatim reproduction reference; only the pipeline path changes).
- Preserve train-only fitting: centering means come from TRAIN rows only (never
  val/test) — this is a leakage-critical detail, test it.
- Regression test: a nonzero-mean target (e.g. `y = P_1(...) + 5`) evaluated across the
  full `_DEFAULT_ALPHAS` grid → `nrmse_test` stays bounded (no blow-up at alpha=1e2);
  and capacity is unchanged vs the old path to within ~1e-6 for a zero-mean target.

## Fix 2 — `first_valid` safety (silent-corruption landmine; fix-first)

Nothing folds a bundle's `first_valid` (QELM warm-up = d−1) into the valid-index cutoff;
a runner that computes `max_delay` from task τ alone would silently score QELM on
zero-padded fake history. Add a runner-facing helper so this cannot be forgotten:
- `pipeline.stage_valid_indices(bundles: Sequence[FeatureBundle], targets, washout,
  max_delay=0) -> np.ndarray` — computes the shared valid index set with
  `effective_max_delay = max(max_delay, *(b.first_valid for b in bundles))`, then the
  usual nan-drop over targets. This is THE function all runners (WP6, WP8) must call.
- Defense in depth: `evaluate_feature_matrix` (or `_validate_split_rows`) raises a clear
  error if any scored row lies below a provided `first_valid` bound.
- Test: QELM at d=3 (first_valid=2) + a task with τ=0 → the scored index set excludes
  rows 0,1 even though the τ=0 target is defined there; no zero-padded QELM row ever
  enters train/val/test.

## Fix 3 — validate feature matrix for NaN/inf (contract gap)

`evaluate_feature_matrix` validates targets but never X. A NaN in a val/test-only row
silently yields NaN capacity/nrmse. Add: raise `ValueError` naming the offending
column/row region if X has non-finite entries within the valid split set.

## Fix 4 — one `effective_rank` definition (reporting clarity)

Two formulas share the name: `core.diagnostics.effective_rank` (Shannon-entropy of the
singular spectrum) vs `matching`'s participation-ratio `(Σs)²/Σs²`. The pipeline's
reported `effective_rank` column currently uses the entropy version while
`matching.effective_dimension` reports participation-ratio — same name, different
quantity, confusing in the paper. Make the PIPELINE report participation-ratio (reuse
the same helper `matching` uses, or a shared `stats`/`diagnostics` function) so
`effective_rank` means ONE thing everywhere. Leave `core.diagnostics.effective_rank`
in place (core stays pristine) but stop using it for the reported column; document the
choice in a one-line comment.

## Fix 5 — clean `d > T` guard in controls (robustness)

`controls._delay_window` crashes with a confusing broadcast error when `d > T`. Raise a
clear `ValueError` (d exceeds stream length) or zero-pad correctly; test `T=3, d=5`.

## Fix 6 — strengthen two thin tests

- Fairness test: use 2–3 targets with DIFFERENT τ (different leading-nan lengths) so the
  "identical index set across all four feature sources" claim is exercised non-trivially.
- Add a `make_splits` boundary test at n=3 and near-degenerate fractions.

## Acceptance

```
.venv/bin/pytest -q
```
All green (more tests than 77). Summary: confirm capacity is unchanged (~1e-6) for a
zero-mean target after the intercept fix, and that the `first_valid` helper excludes
warm-up rows.

## Rules

Minimal diffs; `core/` stays verbatim except where a fix explicitly says otherwise
(none do). Explicit rng. No commits. Don't touch `.specs/`.
