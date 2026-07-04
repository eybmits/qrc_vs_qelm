# WP8a-fix — pin down the uninformative-cell rule + test it (then WP8a commits)

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Closes the WP8a dual-review
findings. Do NOT regress the 128 passing tests or change HP-selection / FDR-family /
determinism logic (all verified correct).

## Context (do not re-litigate)

The pilot's GO came from the FROZEN degree-{1,2,3} gated axis (27 cells) — the runner
already gates exactly those, matching the pilot. The `uninformative_exclusion`
(capacity<0.05) is a SECONDARY safeguard added at the WP7 freeze; the review flagged its
granularity is under-specified and untested. It operates at the family_cell (τ,p) level,
which is correct (family_cell is the unit of analysis for all gates). We keep it, define
it precisely, make its effect fully transparent, and test it.

## Fix 0 — enforce the clean-tree pre-registration gate (fix-first; TOP PRIORITY)

`run_stage1.py` never calls the ported `manifests.require_clean_tree`; its private
`_dirty_state` helper sets `preregistration_violated = bool(dirty and allow_dirty)` —
BACKWARDS. Running against a dirty tree WITHOUT `--allow-dirty` silently proceeds (exit
0) instead of hard-erroring. For an audit-grade confirmatory runner this defeats the
whole clean-tree contract. Fix: call `manifests.require_clean_tree(allow_dirty)` at
startup so a dirty tree without `--allow-dirty` raises `SystemExit`; record the returned
state in the manifest. Add a test asserting `main` exits nonzero on a dirty tree without
`--allow-dirty` (and proceeds, recording the dirty flag, WITH `--allow-dirty`). Apply the
same fix to run_stage2/run_stage3 in WP8b (note it there — but fix stage1 here).

## Fix 1 — precise definition + docs + amendment entry (fix-first)

The CSV `uninformative` column ALSO diverges (code review blocker 2): `_mark_uninformative`
flags it PER-ROW (single seed cell) while the gate excludes per-family_cell mean-over-
seeds. Unify to the family_cell-mean semantics EVERYWHERE: the CSV `uninformative` column
must reflect the SAME family_cell-level determination the gate uses (a row's flag =
whether its family_cell is uninformative by the mean rule), so a downstream filter on the
CSV reproduces exactly the gate's excluded set. Add a test asserting CSV `uninformative`
values are consistent with the manifest `informative_cells`/`uninformative_cells` for
every row.

- In `run_stage1.py`, ensure/keep: a family_cell is `uninformative` iff BOTH models'
  MEAN test capacity across the cell's seed replicates < 0.05. Add a clear docstring on
  `_gate_input_frame` / `_mark_uninformative` stating this exact semantics (family_cell
  mean over seeds, not per-seed-observation).
- Document it in `PROTOCOL.md` (methods) and `PREREGISTRATION.md`: state that (a) the
  PRIMARY gated set is the frozen 27 degree-{1,2,3} cells (from the pilot), and (b) the
  capacity<0.05 rule is a secondary safeguard defined as the family_cell mean-over-seeds
  criterion. Add an AMENDMENT LOG entry recording this definition explicitly (mirrors
  the freeze checklist's per-threshold amendment discipline).

## Fix 2 — full transparency: report gate fractions WITH and WITHOUT the exclusion (fix-first)

Because the exclusion can shrink the G1b/G1c denominator (and thus move a marginal
GO/NO-GO), the runner MUST record BOTH in the manifest / a gate-summary output:
`g1b_fraction_with_exclusion` and `g1b_fraction_without_exclusion` (and the list of
excluded family_cells with their per-model mean capacities). `stage_verdict` uses the
pre-registered WITH-exclusion version, but the WITHOUT version is always reported so a
reader can see the rule's effect. (The reviewer noted `records_full_unfiltered` already
exists — surface it into the manifest rather than discarding it.)

## Fix 3 — test the exclusion-triggered path (fix-first: currently zero coverage)

Add tests in `tests/test_run_stage1.py` (or a gate-input helper test) that:
- construct a synthetic scores frame where one family_cell has both models' mean
  capacity < 0.05 (uninformative) and another straddles it → assert the uninformative
  cell is EXCLUDED from the G1b/G1c denominators, still present in the scores CSV, and
  listed in the excluded-cells report;
- assert the with-exclusion and without-exclusion fractions differ on that frame (so the
  reporting is exercised);
- a lucky-seed case: a family_cell with most seeds < 0.05 but mean just above 0.05 stays
  informative (pins the mean semantics deterministically).

## Fix 4 — small robustness items (non-blocking, do them)

- Add a bundle-level fairness-gate test: force `assert_bundles_matched` or
  `assert_features_live` to be the failing gate independent of config `assert_matched`
  (e.g. monkeypatch `build_features` to return a wrong-`n_obs` bundle), so the
  bundle-level gate is proven load-bearing, not just present.
- Add a one-line comment on `_select_hp` noting the HP-sweep-not-cached tradeoff (a
  mid-chunk interrupt redoes the sweep for that chunk's cells — deliberate).

## Acceptance

```
.venv/bin/pytest -q
PYTHONPATH=src .venv/bin/python -m qrc_vs_qelm.run_stage1 --profile smoke --out results_stage1_smoke   # exit 0
```
All green. Summary: the uninformative definition (family_cell mean-over-seeds), the
with/without-exclusion fractions in the manifest, and the new exclusion-path test result.

## Rules

Minimal diffs. Explicit rng. No commits. Don't touch other `.specs/`.
