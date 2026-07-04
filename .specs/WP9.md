# WP9 — Gate evaluation, figures, report

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Consumes the stage-1/2/3
outputs (scores CSVs + manifests) from WP8a/WP8b and produces the paper-facing
deliverables. Build + TEST on SYNTHETIC fixtures now (the real runs may not exist yet);
it will be applied to real outputs after the confirmatory runs.

## Deliverable: `src/qrc_vs_qelm/run_report.py`

CLI: `--stage1`, `--stage2`, `--stage3` (dirs, each optional — degrade gracefully if a
stage is absent), `--out results_report`. Steps:

1. Load each stage's `stage{N}_scores.csv` + `stage{N}_manifest.json`.
2. **Gate evaluation** (reuse `gates.evaluate_stage{1,2,3}_gates` + `stage_verdict`; do
   NOT reimplement): emit `stage_gate_summary.csv` with every GateRecord (gate_id,
   passed, verdict, value, threshold, ci, q). Report the headline G1b verdict WITH and
   WITHOUT the uninformative exclusion (both are in the stage-1 manifest — surface both).
3. **Figures** (matplotlib, paper quality, reuse `qrc_sym` figure conventions if a style
   helper is importable; else clean defaults):
   - `phase_diagram.png` — Δcapacity (cap_QRC − cap_QELM) heatmap over (τ, degree) at the
     frozen γ=0.3; diverging colormap centered at 0 (blue=QELM, red=QRC); FDR-significant
     cells hatched/outlined; tie cells masked (neutral); uninformative (excluded) cells
     marked distinctly; anchor tasks (STM/NARMA/MG/parity) overlaid as labeled points at
     their (τ,p) profile coordinates. Title carries the G1b verdict + fractions.
   - `boundary_vs_shots.png` (RQ2) — QRC-favored cell count (and QELM-favored) vs shot
     tier {∞,1e5,1e4,1e3}; error bars from shot_seed spread; shows how the boundary moves
     with finite sampling (G2a).
   - `gamma_tradeoff.png` (RQ3) — MC(γ) and NC(γ) curves (both models) over the γ grid,
     per dissipator; mark the crossover γ* where QELM ≥ QRC on ≥half the cells (G3a).
4. `REPORT.md` — the results narrative: per-RQ verdict, the phase-diagram description
   (which regions each model wins), the with/without-exclusion headline, the anchor-task
   placements, the shot-scaling and γ-tradeoff findings, and an explicit limitations
   section (N=5 scale, degree-5 excluded, moderate reproduction, the intentional
   input-injection asymmetry). Pull numbers from the data, do NOT invent them; if a stage
   is absent, say so rather than fabricating.

## Tests (`tests/test_run_report.py`)

- Build SYNTHETIC stage-1/2/3 scores CSVs + manifests (a fixture factory) with a KNOWN
  two-regime structure → `run_report` produces all three PNGs (assert files exist and are
  non-empty) + a well-formed `stage_gate_summary.csv` + REPORT.md containing the injected
  verdict.
- Gate summary matches `gates.stage_verdict` on the fixture.
- Graceful degradation: only `--stage1` given → still produces the phase diagram + a
  REPORT that notes stages 2/3 absent.
- Determinism: figures reproducible (same fixture → same summary CSV).

## Acceptance

```
.venv/bin/pytest -q
PYTHONPATH=src .venv/bin/python -m qrc_vs_qelm.run_report --stage1 <synthetic-or-smoke> --out results_report   # exits 0, writes figures + REPORT.md
```

## Rules

Never fabricate numbers — everything in REPORT.md traces to loaded data. `results_*/`
gitignored (but the report/figures may go to a committed `paper/figures/` later — for now
write under `--out`). No commits. Don't touch `.specs/`.
