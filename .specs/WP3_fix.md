# WP3-fix — close the assert_matched hollow-D bug + recalibrate the liveness guard

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. This fixes two fix-first
findings from the WP3 dual review before WP4 builds on the matching layer. Do NOT
regress any of the 59 passing tests.

## Fix 1 — `assert_matched` certifies hollow D-parity (REAL BUG, fix-first)

Currently `matching.assert_matched(qrc_cfg, qelm_cfg, observables)` recomputes `n_obs`
from each side's `n_qubits` and its own `observables` default. Because `n_obs` is a pure
function of `n_qubits` there, the `readout_dimension` axis can NEVER fire independently:
two bundles actually built with different observable lists (same `n_qubits`) get
certified MATCHED while having different D (demonstrated: D=4 vs D=18, both n_qubits=2,
reported MATCHED). This is exactly the hollow-parity class the fairness layer exists to
prevent.

Required:
- Add `assert_bundles_matched(qrc_bundle: FeatureBundle, qelm_bundle: FeatureBundle)
  -> None` that reads the ACTUAL built quantities off the bundles —
  `features.shape[1]` (D), `n_obs`, `n_multiplex` — and raises `MatchingError` naming
  any that differ. THIS is the real gate the pipeline/runners (WP6, WP8) must call
  before scoring; config-level `assert_matched` remains a cheap pre-flight.
- Make config-level `assert_matched` honest: it must derive `n_obs` from the SAME
  observable set that will build both bundles (single shared `observables`), and its
  docstring must state that true D-parity is certified by `assert_bundles_matched`, not
  by the config check alone. If callers can pass different per-side observable counts,
  raise rather than silently pass.
- Tests: build two bundles with intentionally different observable subsets (same
  n_qubits / n_virtual / n_snapshots) → `assert_bundles_matched` RAISES on D mismatch.
  Keep an isolated-axis test for each `assert_matched` axis that can fire independently;
  document that `readout_dimension` at config level co-fires with n_qubits/multiplex by
  construction (that's why the bundle-level check exists).

## Fix 2 — degeneracy: `recommend_T_evolve` t_lo at n_snapshots=1

`t_lo = timescale / max(n_snapshots, 1)` collapses to `t_lo == t_recommended` when
`n_snapshots == 1` (a legal QELMConfig value), breaking the `t_lo < t_recommended`
monotonicity. Fix so `t_lo < t_recommended < t_hi` strictly for all `n_snapshots >= 1`
(e.g. base the window on the timescale itself, not divided by the snapshot count).
Test `n_snapshots=1`.

## Fix 3 — replace the D-scaling `eff_rank_frac` floor with a multiplex-efficacy guard (fix-first)

CONFIRMED by both reviewers with end-to-end numbers: `eff_rank_frac = eff_rank/D` is a
D-SCALING ARTIFACT, not a liveness measure. `eff_rank` grows sub-linearly in
`n_snapshots` while `D = V·n_obs` grows linearly, so the fraction decays toward 0 as
n_qubits/n_obs grow. At the study's intended scale (n_qubits 4-5, default 3-axis
weight-2 observables) it false-rejects nearly EVERY config on BOTH models — including
the recommended `T_evolve` (QELM ~0.08; best QRC ~0.17; all < 0.5). The 59 tests pass
only because they run at n_qubits=3, D=3 where the ceiling sits above 0.5. This floor
would functionally block the study. Remove it from the gate.

Root cause: the real confound is snapshot COLLAPSE (redundant or dead snapshots
overstating the declared V-fold multiplexing), NOT low overall rank (which is a
legitimate property of a scalar-input feature map — and if a channel is genuinely weak,
that is a SCIENTIFIC finding for the G1c kill-rule to catch, not a config error).

Required redesign:
- `multiplex_efficacy(features, n_snapshots, n_obs) -> dict` — snapshot blocks are
  column-major per WP2 (block k = `features[:, k*n_obs:(k+1)*n_obs]`). Compute
  `r_full = eff_rank(features)` and `r_single_mean = mean_k eff_rank(block_k)` using the
  SAME participation-ratio `eff_rank` as `effective_dimension`. Return
  `{r_full, r_single_mean, gain = r_full / max(r_single_mean, 1e-12)}`. Because both
  ranks are at the same n_obs scale, the ratio is SCALE-INVARIANT — the D-scaling
  artifact cancels.
- New `assert_features_live(bundle, *, min_col_var=1e-8, min_multiplex_gain=1.15)`:
  * ALWAYS: dead-column floor — every feature column's across-input variance ≥
    min_col_var (catches steady-state saturation at large T_evolve).
  * IF `n_multiplex > 1`: also require `multiplex_efficacy(...).gain ≥
    min_multiplex_gain` (catches T_evolve→0 identical snapshots: gain→1.0, and
    dead-later-snapshot cases). For `n_multiplex == 1` (the PRIMARY phase-diagram
    configs use V=1) apply ONLY the dead-column floor — there is no multiplexing to
    police, and a shallow-but-input-dependent V=1 map is legitimate.
  * REMOVE the global `eff_rank_frac ≥ min_rank_frac` check from the gate.
- Keep `effective_dimension` REPORTING `eff_rank` and `eff_rank_frac` (WP7 manifest
  transparency) — but they are diagnostics, not gate criteria.
- CALIBRATION (required in your summary): re-run these sweeps and show the gain metric
  SEPARATES cleanly — (a) the WP2 extremes n_qubits=4,d=4,gamma=0.3 at T_evolve ∈
  {0, recommended, 100×mixing_time}; (b) n_qubits ∈ {2,3,4,5} at recommended T_evolve;
  (c) a QRC sweep over h_in/dt/n_virtual at n_qubits=4. Target behavior: gain ≈ 1.0 at
  T_evolve=0 and full steady-state (FAIL), gain comfortably > 1.15 at recommended
  T_evolve for n_qubits 4-5 (PASS). If 1.15 doesn't cleanly separate on the measured
  data, pick the threshold that does and REPORT the numbers + chosen value (false-reject
  is the demonstrated danger — bias the threshold LOW; the dead-column floor and the
  G1c kill-rule are the real backstops). The re-review will re-run these exact sweeps.
- Tests: T_evolve=0 → gain≈1.0 → RAISES (n_multiplex>1); recommended T_evolve at
  n_qubits=4, V=4, DEFAULT observables → PASSES (this is the regression test the old
  suite lacked — it MUST use the default observable set, not a single-axis reduction);
  V=1 config with a live encoded state → PASSES; a synthetic all-dead-column bundle →
  RAISES via the variance floor.

## Fix 4 — thread `min_col_var` through `effective_dimension` (non-blocking, do it)

`effective_dimension`'s `n_dead_cols` hardcodes `1e-8` while `assert_features_live` uses
its own `min_col_var`. Thread the threshold through so the manifest-logged
`n_dead_cols` matches the configured floor. Also update the recommend/assert docstrings
to state the gate no longer uses `eff_rank_frac`.

## Handoff notes to record in your summary (for WP6/WP7, so they aren't rediscovered)

- `recommend_T_evolve` is ADVISORY; nothing forces the frozen grid into `[t_lo, t_hi]`.
  WP7 checklist must verify the frozen `T_evolve` grid sits inside `[t_lo, t_hi]` for
  BOTH models, not merely that it clears `assert_features_live`.
- `eff_rank_frac` is compressed into a low band (~0.1-0.2) for both models at realistic
  D; it is a transparency diagnostic, not a fairness discriminator — WP7's
  "report and discuss effective-D difference" should lean on the multiplex-gain and
  per-cell capacity, not on eff_rank_frac magnitudes.
