# WP-I2-fix2 — the threshold must not exceed a single-function capacity bound

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Both WP-I3 reviewers (fix-first)
proved the IPC estimator is miscalibrated: the per-degree **cumulative** noise threshold I
added in WP-I2-fix reaches **5.1–7.0**, even though any single basis function's capacity is
bounded in **[0,1]**. It therefore zeroes ALL distributed/degree≥2 real capacity (the full
QRC's thin, spread-across-delays memory), collapsing QRC IPC to 0 exactly in the weak-γ,
long-memory regime where the QRC should dominate — producing large SPURIOUS negative Δ
(QELM > the QRC ceiling). Do NOT regress the 176 tests. The runner (WP-I3) is correct; the
bug is here.

## Root-cause fix

The design mistake: I used the THRESHOLD to enforce the Dambre bound (inflating it). The
**clamp** (`total_ipc = min(Σ significant, rank)`, already added) is what enforces the bound.
So the threshold should be a genuine per-function NOISE FLOOR, and the clamp carries the
theorem.

1. **Remove `_raise_thresholds_for_cumulative_noise`** and the cumulative-sum logic.
2. **Per-function (per-degree) significance threshold, bounded [0,1]**: for each degree class,
   the threshold = a high percentile (e.g. 99th) of the distribution of SINGLE-function
   capacities of the model's real features against shuffled-target surrogates. This is a
   capacity, so it is in [0,1] by construction. A basis function is significant iff its real
   capacity exceeds this per-degree floor. (This retains the QRC's distributed per-delay
   capacities that individually clear the floor, instead of erasing them.)
3. **Keep the clamp** `min(Σ significant capacities, effective_rank)` as the bound enforcer.
4. **Hard invariant + test**: `thresholds_by_degree[d] ≤ 1.0 + 1e-9` for ALL degrees
   REGARDLESS of basis size (this is the checkable version of the bug — assert it in the
   test suite with a large basis, hundreds of functions per degree).

## Diagnostics (so a future miscalibration is visible, not silent)

5. `total_ipc` returns `n_significant` and a `threshold_saturated: bool` flag =
   (features are non-degenerate — nonzero column variance — AND n_significant == 0). Do NOT
   force `basis_edge_capacity = 0.0` when `raw_total ≤ 0`; compute it from the raw (pre-cut)
   capacities so it can still flag basis-width problems. Log a loud warning on
   `threshold_saturated`.

## Re-verify — the ceiling must hold on REAL data (not a synthetic fixture)

6. Add a test that runs the REAL IPC path (real small QRC + windowed TD-QELM features at a
   PRODUCTION-shape basis: many degree-2/3 functions) and asserts **IPC(QRC) ≥ IPC(QELM,k)
   within noise** across a few γ (including a weak-γ cell), and that a weak-dissipation QRC
   has **nonzero** total IPC (its distributed memory is retained, not zeroed). If the ceiling
   only holds at larger T than smoke, DOCUMENT the T requirement in the test/summary (it
   informs the pilot's confirmatory T) — but the threshold-≤1 invariant and nonzero-QRC-IPC
   must hold regardless.
7. Re-run `python -m qrc_vs_qelm.run_stage_ipc --profile smoke --out results_ipc_smoke
   --allow-dirty` and report the new per-(γ,k) delta_ipc — QRC IPC must no longer be 0 at
   γ=0.02, and delta_ipc should be ≥ 0 within noise (report any residual negatives + their
   magnitude vs the per-cell estimation noise, for the pilot to assess).

## Acceptance

```
.venv/bin/pytest -q
PYTHONPATH=src .venv/bin/python -m qrc_vs_qelm.run_stage_ipc --profile smoke --out results_ipc_smoke --allow-dirty
```
All tests green. Summary: the threshold-≤1 invariant test, the real-data ceiling test result
(and any documented T requirement), and the new γ=0.02 delta_ipc values.

## Rules

The clamp enforces the bound; the threshold is a per-function floor. Reuse
`pipeline`/`core.readout`. Explicit rng. No commits. Don't touch `.specs/`.
