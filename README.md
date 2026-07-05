# qrc_vs_qelm

A **pre-registered, resource-matched comparison** of Quantum Reservoir Computing
(QRC) — a recurrent dissipative Lindblad spin reservoir with intrinsic quantum
memory — against Quantum Extreme Learning Machines (QELM) — a static, memoryless
quantum channel applied to a *classical* delay-embedding of the input.

It answers the open problem left as future work in arXiv:2602.21544v2: **under
matched resources, when does intrinsic quantum memory beat a static quantum map fed
a classical delay embedding — and how do finite sampling and engineered dissipation
move that boundary?**

## Status: complete

All work packages (WP0–WP10) are implemented, tested (153 passing), and the frozen
confirmatory run has been executed. The modeling stack is validated against an
independent reference (reproduces the source reservoir's short-term memory capacity
in-band and the continuous-dissipation-beats-Fuji-Nakajima ordering).

## Findings (N=5, confirmatory run)

- **RQ1 — memory×nonlinearity phase diagram (GO).** QRC and QELM occupy distinct,
  individually significant regions: QRC wins the intermediate/long-delay **memory
  band** (delays beyond QELM's classical embedding window, within the 5-qubit memory
  horizon ≈ τ 3–10); QELM wins **short-delay nonlinear** tasks. A pre-registered
  kill rule confirms QELM significantly beats a classical delay-embedding baseline
  (it is a non-trivial quantum comparator). *Caveat:* the GO holds on the
  pre-registered informative cell set; on the full grid — including long-delay cells
  where **both** models fail at N=5 — it reads `SINGLE_REGIME`. Both are reported.
- **RQ2 — shot scaling (suggestive, underpowered).** The QRC-favored cell count
  falls as shots shrink (9→7→5), but the CI on the shift includes zero at the
  24-seed budget: a candidate for a higher-powered replication.
- **RQ3 — a between-model division of labour.** QRC is dissipation-*powered*
  (near-useless at zero dissipation; memory MC climbs 2.4→6.8 as γ grows); QELM is
  dissipation-*robust* (flat, high nonlinear capacity that exceeds QRC's throughout).
  QRC's memory overtakes QELM's around γ≈0.05.

See `paper/manuscript_draft.md` for the full writeup and `paper/figures/` for the
committed figures and gate summary.

## Repo layout

- `src/qrc_vs_qelm/` — package: `core/` (ported Lindblad simulator), `models/`
  (QRC + QELM), `matching.py`, `pipeline.py`, `tasks.py`/`task_families.py`,
  `controls.py`, `seeds.py`, `gates.py`, `stats.py`, `run_stage{1,2,3}.py`.
- `scripts/` — `run_validation.py`, `bench_configs.py`, `design_pilot.py`.
- `configs/` — frozen `full.yaml` and scaled-down `smoke.yaml`.
- `data/frozen/` — the frozen operating-point config (single source of truth).
- `PROTOCOL.md` / `PREREGISTRATION.md` — frozen methods and commitments + amendment log.
- `paper/` — manuscript draft and committed figures.
- `.specs/` — per-work-package implementation specs (development provenance).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Reproducing the study

The runners refuse to execute against an uncommitted tree (pre-registration
integrity); each writes a manifest with the git commit, seed grid, per-cell
feature-cache SHA-256 hashes, and package versions. Result directories
(`results_*/`) are gitignored and regenerated locally:

```bash
# validation gate (reproduces reference physics) — must pass before main runs
python -m qrc_vs_qelm.run_validation --profile full

# confirmatory pipeline (frozen operating point: N=5, gamma=0.3, quantize=33)
python -m qrc_vs_qelm.run_stage1 --profile full --out results_stage1_full
python -m qrc_vs_qelm.run_stage2 --profile full --stage1-out results_stage1_full --out results_stage2_full
python -m qrc_vs_qelm.run_stage3 --profile full --out results_stage3_full

# gate evaluation, figures, REPORT.md
python -m qrc_vs_qelm.run_report --stage1 results_stage1_full --stage2 results_stage2_full --stage3 results_stage3_full --out results_report_full
```

Use `--profile smoke` for a fast (minutes) end-to-end check at N=4.

## Reproducibility & governance

The pre-registration (`PREREGISTRATION.md`) freezes the operating point, gate
thresholds, seed grids, and analysis plan, with a full amendment log of the
design-pilot trail (on disjoint seeds) and every pre-execution correction. The
`stage1-frozen` git tag marks the frozen plan.

`AGENTS.md` / `CLAUDE.md` document the AI-assisted development workflow (planner +
implementer + reviewers) used to build the package.
