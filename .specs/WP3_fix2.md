# WP3-fix2 — small closeout of the WP3-fix review (then WP3 commits)

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. The WP3-fix code-review
verdict was fix-first on ONE substantive point plus two cleanups. Do NOT regress the
66 passing tests. Keep changes minimal.

## Context for you (the scope decision that resolves the blocker)

The reviewer found the multiplex-gain gate false-rejects most QRC configs at V>1
(n_virtual>1). This is NOT a guard bug — it correctly detects that QRC virtual-node
multiplexing (substep states within one dt) is strongly correlated, so declared
D=V·n_obs overstates QRC's effective dimension. The STUDY DECISION (now pre-registered)
is: **the phase-diagram comparison runs at V=1 only**; V>1 matched-D comparison is
dropped as inherently unfair (hollow QRC-D vs live QELM-D). Therefore the multiplex
gate is NOT on the primary path — V=1 configs skip it by design. It remains a guard for
any exploratory V>1 use. Do not lower `min_multiplex_gain` (that would re-hollow the
guard); instead document the scope and close the test gap.

## Required changes

1. **Close the untested-QRC-side gap** (the reviewer's blocker). Add tests exercising
   `assert_features_live` on a QRC-built `FeatureBundle`:
   - QRC at N=5 (or N=4 for speed), `n_virtual=1`, typical `h_in`/`dt` (e.g. h_in=1.0,
     dt=0.5), live drive → `assert_features_live` PASSES (V=1 skips multiplex gate; the
     dead-column floor is satisfied by an input-driven reservoir). This proves the
     PRIMARY path works for QRC.
   - A QRC bundle with `h_in=0` (no drive → dead/constant features) → RAISES via the
     dead-column floor (guard still protects against genuinely dead QRC features).
2. **Document the V>1 scope in code** so this isn't rediscovered. In `matching.py`,
   extend the `assert_features_live` / `multiplex_efficacy` docstrings: state that the
   multiplex-gain gate applies only to n_multiplex>1, that the study runs V=1, and that
   low multiplex gain at V>1 (especially QRC virtual nodes at small dt) is EXPECTED and
   correctly indicates hollow declared multiplexing — not a config to force through by
   lowering the threshold.
3. **Remove the dead `n_snapshots` parameter** from `recommend_T_evolve` (the fix made
   its output independent of `n_snapshots`; the reviewer confirmed two values give
   byte-identical dicts). Delete the parameter, update all call sites and tests. If any
   caller genuinely needs it later it can be re-added meaningfully.

## Do NOT change

- `min_multiplex_gain=1.15` (correct; leave it).
- The gate logic (dead-col floor always; multiplex gate iff V>1). It is right.
- `assert_bundles_matched`, `assert_matched` pre-flight, `multiplex_efficacy` math,
  `min_col_var` threading — all verified correct by review; leave them.

## Acceptance

```
.venv/bin/pytest -q
```
All green (was 66; will be a few more with the new QRC-side tests, minus any that
referenced the removed `n_snapshots` arg). Summary: the QRC V=1 liveness test result,
and confirmation `recommend_T_evolve` no longer takes `n_snapshots`.

## Rules

Minimal diff. No commits. Don't touch `.specs/`. Explicit rng.
