# WP5 — Frozen seeds, pre-registered gates, statistics

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Depends on WP1–WP4.
You may READ these as design references (adapt, don't blindly port — schemas differ):
- `/Users/superposition/Coding/qrc_attribution_audit/src/qrc_attribution_audit/seeds.py`
  (SeedCell + SEED_GRIDS + grid_manifest pattern).
- `/Users/superposition/Coding/qrc_attribution_audit/src/qrc_attribution_audit/gates.py`
  (frozen GateRecord / stage_verdict / kill-rule / markdown-table pattern).
- `/Users/superposition/Coding/qrc_sym/experiments/paper_experiments/statistics_utils.py`
  (paired_stats, benjamini_hochberg, bootstrap CI — reuse the math).

## CRITICAL orientation convention (get this right or every gate inverts)

Our primary metric is **capacity** (in [0,1], HIGHER is better) — the OPPOSITE of the
NRMSE orientation in qrc_sym's `paired_stats` (where positive gain = lower is better).
Define ONE convention and use it everywhere: **delta = capacity_QRC − capacity_QELM**,
so **positive delta ⇒ QRC wins**. Every gate, CI, and label derives from this signed
delta. Write this convention as a module docstring in `stats.py` and assert it in tests.

## Deliverables

1. `src/qrc_vs_qelm/seeds.py`:
   - `@dataclass(frozen=True) class SeedCell` with fields for THIS project:
     `input_seed: int` (i.i.d. input stream u), `coupling_seed: int` (J matrix),
     `shot_seed: int = 0` (shot noise). No `esn_seed` (there is no ESN here).
   - `SEED_GRIDS: dict[str, tuple[SeedCell, ...]]` — frozen named grids:
     - `stage1_smoke`: 1 cell.
     - `stage1_full`: input_seed 0..9 × coupling_seed {314159, 271828, 161803} (30 cells).
     - `stage2_shots`: the stage1 coupling set × shot_seed {0,1,2} at fixed input_seed=0.
     - `stage3_gamma`: input_seed 0..4 × coupling_seed {314159, 271828} (10 cells).
     - `pilot`: a DISJOINT design-stage grid — input_seed {900,901,902} ×
       coupling_seed {987654, 123456}. These seeds must NOT appear in any confirmatory
       grid (assert this in a test — the pilot exists to tune τ/p ranges without
       burning confirmatory seeds; see PROTOCOL amendment-log rationale).
   - `grid_manifest(name: str) -> dict` — name, cell count, and the explicit cell list
     (JSON-serializable) for embedding in run manifests.
   - `__all__` exported.
2. `src/qrc_vs_qelm/stats.py`:
   - `bootstrap_ci(values, seed, n_boot=2000, alpha=0.05) -> (lo, hi)` — percentile
     bootstrap of the mean (adapt qrc_sym `_bootstrap_ci`).
   - `paired_delta_stats(cap_qrc, cap_qelm, *, seed, n_boot=2000) -> dict` — paired
     over seed cells: `delta = cap_qrc − cap_qelm`; returns mean delta, bootstrap
     CI (lo/hi), SE, Cohen's dz, win_rate = mean(delta>0), Wilcoxon p (two-sided,
     graceful nan on failure). POSITIVE ⇒ QRC. (This is qrc_sym `paired_stats`
     re-oriented for capacity — verify the sign by a test with synthetic data where
     QRC is known-better.)
   - `benjamini_hochberg(pvals) -> np.ndarray` — port verbatim from qrc_sym.
   - `add_q_values(df, p_col, q_col)` helper.
   - `label_cell(delta_stat, tie_band=0.02, q=None, q_thresh=0.10) -> str` — returns
     `"QRC"` / `"QELM"` / `"tie"`: tie if |mean delta| < tie_band OR (q is not None
     and q > q_thresh); else sign of delta. (Ties are a first-class third label per
     the plan — NOT a failure.)
3. `src/qrc_vs_qelm/gates.py` — adapt the frozen-gate PATTERN (not the pocket schema):
   - `@dataclass(frozen=True) class GateRecord` with: `gate_id, passed: bool,
     verdict: str, value, threshold, ci_low, ci_high, q_value, notes`.
   - Frozen `GATE_DEFS` mapping gate_id → {description, kind, threshold, kill: bool}.
     Implement these gates operating on a per-cell scores DataFrame (columns include
     `cell_id, family_cell (tau,p), capacity_qrc, capacity_qelm, delta`):
     - **G1a** (per-cell significance): for each family cell, `paired_delta_stats`
       CI excludes 0 AND BH-FDR q ≤ 0.10 across cells. Records per-cell.
     - **G1b** (headline, required): fraction of family cells labeled QRC ≥ 0.20 AND
       fraction labeled QELM ≥ 0.20 (both regimes exist). Else verdict `SINGLE_REGIME`.
     - **G1c** (kill rule): QELM beats the `linear_delay` control on ≥ 25% of p≥2
       cells (capacity). Else verdict `QELM_TRIVIAL` (the whole comparison is
       meaningless if the quantum map adds nothing over linear delay embedding).
     - **G2a**: count of QRC-favored cells changes between S=∞ and S=1e4 with CI
       excluding 0, else `SHOT_INSENSITIVE`.
     - **G3a**: MC(γ) trend non-increasing and NC(γ) trend non-decreasing (Spearman,
       report rho + CI) AND a crossover γ* exists where QELM ≥ QRC on ≥ half the
       cells; else `NO_TRADEOFF`.
   - `evaluate_stage1_gates(df) / evaluate_stage2_gates(...) / evaluate_stage3_gates(...)`
     → `list[GateRecord]`.
   - `stage_verdict(records, stage) -> dict` — PASS only if all required gates pass
     and no kill rule triggered; kill rules short-circuit to their named verdict.
   - `gates_markdown_table() -> str` — renders GATE_DEFS as a markdown table for
     auto-insertion between the `<!-- GATES:BEGIN/END -->` markers in
     PREREGISTRATION.md (WP7 consumes this).
4. Tests (`tests/test_seeds.py`, `tests/test_stats.py`, `tests/test_gates.py`):
   - Seeds: grids frozen/immutable; pilot seeds DISJOINT from all confirmatory grids
     (explicit set-intersection test); grid_manifest round-trips to JSON.
   - Stats: orientation — synthetic data with cap_qrc > cap_qelm gives positive mean
     delta, win_rate > 0.5, CI lo > 0; BH monotonicity + known-vector check; tie_band
     labeling; `label_cell` returns tie when q > threshold.
   - Gates: synthetic score frames driving EACH verdict — a clean two-regime frame →
     G1b PASS; an all-QRC frame → SINGLE_REGIME; a frame where QELM ≤ linear_delay
     everywhere → QELM_TRIVIAL kill; gate immutability (GATE_DEFS is frozen);
     markdown table contains every gate_id.
   - Keep everything pure-numpy/pandas, no simulation — fast.

## Acceptance

```
.venv/bin/pytest -q
```
All green. Summary: files, test count, and a note confirming the delta-orientation
convention is tested.

## Rules

Reuse the referenced math (bootstrap, BH-FDR) faithfully; adapt schemas to our
capacity-based, QRC-vs-QELM setting. Explicit rng/seed args. No results files.
No commits. Don't touch `.specs/`.
