# WP8b-fix — correct the RQ3 dissipation-sweep protocol (then WP8b commits)

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Both WP8b reviewers
(fix-first / reconsider) empirically confirmed the stage-3 blocker: QELM capacity is
bit-identical across γ values → QELM is γ-INVARIANT → RQ3 cannot measure the QELM side of
the dissipation tradeoff. Fix without regressing the 141 passing tests or the verified-
correct parts (stage-1 cache verification, stage-2 no-simulator guard, MC/NC math,
clean-tree, determinism, checkpoint staleness).

## Fix 1 — stage-3 QELM `T_evolve`: FIXED across the γ sweep (fix-first blocker)

Remove the disguised `min(recommend_T_evolve, matched_unitary_timescale)` cap
(`run_stage3.py:486-502`, call site :305-306) — it always binds (~0.766) and freezes
QELM's horizon γ-independently.

Replace with a genuinely FIXED per-coupling-seed horizon:
`T_evolve = recommend_T_evolve(qelm_cfg_at_gamma_0.3, coupling_seed)["t_recommended"]`
computed ONCE at the RQ1 operating point γ=0.3, held CONSTANT across the entire γ grid.

Rationale (put in a code comment + the amendment log): this MATCHES QRC's design — QRC
evolves a fixed `dt=0.5` per input symbol while γ varies only the dissipator, so QELM
must likewise hold its per-window evolution horizon fixed while γ varies only the
dissipator. Then γ cleanly = dissipation strength for BOTH models, and the MC/NC-vs-γ
curves are attributable to γ, not to a co-varying horizon. (The spec's original
"recommend_T_evolve at each γ" was wrong: mixing_time ∝ 1/γ-ish, so per-γ recommend makes
γ·T_evolve ≈ const → QELM artificially γ-insensitive, AND unmatched to QRC's fixed dt.)

## Fix 2 — stage-3 graceful dead-feature recording (fix-first)

With a fixed T_evolve, QELM (or QRC) MAY collapse to dead features at high γ — that is the
RQ3 PHENOMENON, not an error. So in stage-3 ONLY (stage-1 keeps its hard error): if
`assert_features_live` fails for a model at a (γ, dissipator, seed) point, DO NOT crash —
record that point with `features_dead=True` and MC=NC=0 and per-cell capacity=0 for that
model, and CONTINUE the sweep. (assert_bundles_matched still hard-errors — that's a
config bug, not physics.) Add the `features_dead` column to `stage3_scores.csv`.

## Fix 3 — stage-2 shot noise only on QUANTUM models, independent per-model RNG (important)

`run_stage2.py:428-435` applies `add_shot_noise` to ALL models including the CLASSICAL
controls (`linear_delay`, `poly_delay`) — but those are analytic features, not finite-shot
quantum measurements, so shot noise on them is meaningless. Apply shot noise ONLY to
`qrc` and `qelm`; score `linear_delay`/`poly_delay` at their exact (noiseless) values at
every shot tier. Also derive an INDEPENDENT rng per (model, tier, shot_seed) — e.g.
`np.random.default_rng((shot_seed, hash_of_model_name, tier_index))` — so QRC and QELM
noise realizations are independent, not the same re-seeded stream. Document that controls
are shot-noise-exempt.

## Fix 4 — wire the stage-3 runtime-budget guard (non-blocking, do it)

`run_stage3.py:240-256` hardcodes `total_hours=0.0` → the shrink `SystemExit` is dead
code. Wire `bench_configs.project_runtime` (as stage-1 does) to project the γ×dissipator×
seed grid runtime and raise the pre-registered shrink message if over the 10h stage
budget. (Stage-2 is ridge-only/cheap — a disclosed no-op guard is acceptable there.)

## Fix 5 — amendment log + tests

- PREREGISTRATION.md amendment entry: "RQ3 QELM T_evolve corrected PRE-EXECUTION (no
  stage-3 data existed): fixed at the γ=0.3 operating-point recommendation across the
  sweep (was per-γ recommend, which was unmatched to QRC and made QELM γ-insensitive);
  dead-feature points recorded as MC/NC=0." Update PROTOCOL.md RQ3 methods accordingly.
- Tests: (a) QELM capacity/MC/NC VARY across the real γ_grid (the test the suite lacked —
  would have caught this); (b) T_evolve is constant across γ for a fixed coupling seed;
  (c) a synthetic dead-feature point is recorded (features_dead=True, MC=NC=0) not crashed;
  (d) classical controls are identical across shot tiers (not shot-noised); (e) QRC vs
  QELM shot-noise draws are independent.

## Acceptance

```
.venv/bin/pytest -q
PYTHONPATH=src .venv/bin/python -m qrc_vs_qelm.run_stage2 --profile smoke --stage1-out results_stage1_smoke --out results_stage2_smoke --allow-dirty  # exit 0
PYTHONPATH=src .venv/bin/python -m qrc_vs_qelm.run_stage3 --profile smoke --out results_stage3_smoke --allow-dirty  # exit 0
```
Summary: confirm QELM capacity now varies with γ, controls are shot-exempt, and the
dead-feature path records rather than crashes.

## Rules

Minimal diffs to the flagged logic; leave verified-correct parts alone. Explicit rng.
No commits. Don't touch other `.specs/`.
