# WP5-fix — gate significance-gating + missing gate tests (then WP5 commits)

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Closes WP5 dual-review
findings. Do NOT regress the 95 passing tests. `stats.py`/`gates.py`/`seeds.py` math
that the review verified correct (delta sign, BH-FDR, verdict wiring) MUST stay correct.

## Fix 0 — G1c kill rule needs a significance test (fix-first; STRONGEST finding)

G1c (the QELM_TRIVIAL kill rule that protects the whole study from a meaningless
comparison) currently counts a p≥2 cell as "QELM beats linear_delay" via a BARE
`qelm_mean > linear_mean` — no CI, no margin. Confirmed gameable: passes on a 1e-9
fluke, and under pure noise ~half of cells clear it by chance, so the kill rule can pass
on noise alone. Fix to match the significance pattern used by every other gate:
- A p≥2 cell counts as "QELM beats linear_delay" ONLY if the paired (over seed cells)
  per-cell delta `capacity_qelm − capacity_linear_delay` is SIGNIFICANTLY positive —
  its bootstrap CI (reuse `stats.paired_delta_stats` / `bootstrap_ci`) excludes 0.
  (This requires the scores frame to carry per-(cell,seed) `capacity_linear_delay`
  alongside `capacity_qelm`; WP8a produces the linear_delay control per seed cell.)
- G1c PASSES iff ≥25% of p≥2 cells show a SIGNIFICANT QELM-over-linear_delay advantage;
  else QELM_TRIVIAL. Keep the 25% floor (a WP7-freeze value); the fix is adding
  significance, not changing the floor.
- Test: a pure-noise frame (QELM ≈ linear_delay) → G1c FAILS → QELM_TRIVIAL (this is the
  regression that pins the fix); a frame where QELM significantly beats linear_delay on
  ≥25% of p≥2 cells → G1c PASSES.

## Fix 0b — remove the decorative `gates.TIE_BAND` freeze trap (fix-first)

`gates.TIE_BAND = 0.02` is defined next to the wired constants (FDR_Q,
STAGE1_MIN_REGIME_FRACTION, QELM_LINEAR_DELAY_FLOOR) but is NEVER passed to
`label_cell` — the actual tie band is `stats.label_cell`'s own default. Today both are
0.02 so no live bug, but a future freeze that edits `gates.TIE_BAND` would silently
change nothing. Thread `TIE_BAND` through `label_cell(..., tie_band=gates.TIE_BAND)` so
the constant is the single source of truth (preferred), or delete it. A test should pin
that changing the wired tie band actually changes labeling.

## Fix 1 — G1b headline gate must count only SIGNIFICANT cells (fix-first)

Currently `label_cell` uses direction + tie_band + q only; `evaluate_stage1_gates`
counts QRC/QELM regime fractions from that label, decoupled from the per-cell G1a
significance (CI-excludes-0 AND BH-q ≤ 0.10). So a cell whose own bootstrap CI straddles
zero can still count toward G1b's "both regimes exist ≥20%" headline. That overstates
the evidence. Fix:
- G1b's regime-fraction counts must include a cell in the QRC (resp. QELM) region ONLY
  if that cell is BOTH direction-labeled QRC/QELM AND per-cell significant (its paired
  bootstrap CI excludes 0 AND its BH-FDR q ≤ 0.10 — i.e. the same criterion as G1a).
  Ties and non-significant cells count toward neither region.
- Keep `label_cell` as the direction+tie primitive (used for plotting), but add the
  significance filter in the G1b evaluation path (or a `significant: bool` column the
  gate consumes). Document the distinction: the phase-diagram PLOT may show direction
  labels; the GATE counts significant cells only.
- Test: a frame with cells that are direction-QRC but CI-straddling-zero → those cells
  do NOT count toward G1b's QRC fraction; a clean two-regime significant frame → PASS;
  a frame where one regime's cells are all non-significant → SINGLE_REGIME.

## Fix 2 — add the missing G2a and G3a tests (fix-first: required gates, zero coverage)

Both are `required: True` with branchy logic and NO tests. Add synthetic-frame tests
driving each verdict:
- G2a (shot sensitivity): a frame with shot-tier column ({inf, 1e4} both string and
  numeric np.inf/10000 forms — test both alias paths) where the QRC-favored cell count
  SHIFTS between tiers with CI excluding 0 → PASS; a frame where counts don't move →
  SHOT_INSENSITIVE; a frame missing the shot column → sensible error/verdict.
- G3a (dissipation tradeoff): a frame with monotone MC(γ)-decreasing + NC(γ)-increasing
  AND a crossover γ* where QELM ≥ QRC on ≥50% of cells → PASS; a frame with no crossover
  or non-monotone trend → NO_TRADEOFF.

## Fix 3 — cap/flag Cohen's dz on degenerate variance (robustness)

`dz = mean_delta/(delta_std+1e-12)` blows to 1e10 for near-constant delta. Return
`np.inf` (or `np.nan` with a flag) when `delta_std` is below a small threshold, so a
degenerate effect size reads as undefined, not a nonsense finite number. dz is not gate-
consumed, so this is presentation-safety only. Add a near-zero-variance test.

## Fix 4 — call Wilcoxon once (perf nit)

`paired_delta_stats` computes the Wilcoxon p twice (for `p_wilcoxon` and `wilcoxon_p`
aliases). Compute once, assign both. No behavior change; add nothing.

## Acceptance

```
.venv/bin/pytest -q
```
All green (more than 95). Summary: confirm G1b now excludes CI-straddling cells, and
G2a/G3a each have PASS + failure-verdict tests.

## Rules

Minimal diffs. Explicit rng. No commits. Don't touch `.specs/`.

<!-- Adversarial-review threshold-justification items (20%/25%/tie_band=0.02, power at
30 samples/cell) are being triaged separately; most are WP7-freeze justification items
informed by the design pilot, NOT WP5 code changes. This spec covers only the code fixes. -->
