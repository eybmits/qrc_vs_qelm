# WP0 — Scaffold + governance for `qrc_vs_qelm`

You are implementing work package WP0 of a new research repo. Work ONLY inside
`/Users/superposition/Coding/qrc_vs_qelm/` (your working directory). You may READ
sibling repos under `/Users/superposition/Coding/` for templates as instructed below.

## Project context (for the docs you will write)

Pre-registered, resource-matched comparison of **Quantum Reservoir Computing (QRC)**
(recurrent dissipative Lindblad spin reservoir, intrinsic memory) vs **Quantum Extreme
Learning Machines (QELM)** (static memoryless quantum channel applied to a classical
delay-embedding of the input). Solves the future-work problem of arXiv:2602.21544v2.
Research questions: RQ1 phase diagram over task memory-demand (Legendre delay τ) ×
nonlinearity-demand (degree p); RQ2 shot-budget scaling of the QRC/QELM boundary;
RQ3 dissipation γ trading memory capacity vs nonlinear capacity. Backend: NumPy/SciPy
Lindblad core ported (in WP1, not now) from `qrc_dissipation_engineering`.

## Deliverables

1. `pyproject.toml` — src layout, package `qrc_vs_qelm`, Python `>=3.11`.
   Deps: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `pyyaml`.
   Optional dev group: `pytest`. No quantum libraries.
2. `AGENTS.md` — READ `/Users/superposition/Coding/qrc_ecg/AGENTS.md` and adapt it:
   keep the rules (determinism via explicit `seed`/`rng` args — never global RNG;
   train-only standardization; matched feature budgets across models; `pytest -q`
   must pass before declaring done; implementers make no scientific-results claims).
   Replace reuse pointers with: `qrc_dissipation_engineering/src/qrc/` (core simulator),
   `qrc_dataset_profiler/src/qrc_dataset_profiler/reservoir.py` (shot noise) and
   `baselines.py` (splits/ridge patterns), `qrc_sym/experiments/paper_experiments/statistics_utils.py`
   (paired stats), `qrc_attribution_audit` (seeds/gates/manifests patterns).
3. `CLAUDE.md` — short workflow doc: Claude writes one spec per WP in `.specs/WP*.md`;
   Codex implements; Sonnet subagents review; one WP per delegated task; results
   directories are never committed by implementers.
4. `PROTOCOL.md` and `PREREGISTRATION.md` — skeletons only (content lands in WP7):
   section headers for task construction, resource-matching protocol, shot-noise
   semantics, hyperparameter grids, split pipeline, gates, seed grids, row caps,
   shrink rules, amendment log. Include literal `<!-- GATES:BEGIN -->` and
   `<!-- GATES:END -->` marker comments in PREREGISTRATION.md.
5. `README.md` — project one-paragraph description, repo layout, quickstart
   (venv + editable install + pytest), WP index table.
6. `.gitignore` — Python defaults plus: `.venv/`, `results_*/`, `data/`, `*.npz`,
   `__pycache__/`, `.pytest_cache/`.
7. `configs/smoke.yaml` and `configs/full.yaml` — stubs with: `profile` name,
   `n_qubits` (smoke 4 / full 5), `T` (400 / 3000), `washout: 100`, and empty
   placeholder keys for `hp_grids`, `seed_grid`, `shot_tiers`, `gamma_grid`.
8. `src/qrc_vs_qelm/__init__.py` with `__version__ = "0.1.0"`.
9. `tests/test_scaffold.py` — imports `qrc_vs_qelm`, checks `__version__`, parses both
   YAML configs and asserts the required keys above exist.

## Environment + acceptance (run these yourself; all must pass)

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Rules

- Do not create any results files or make any scientific claims.
- Do not run `git commit` (orchestrator commits).
- Do not modify anything under `.specs/`.
- Keep it minimal — no speculative modules beyond the deliverables list.

When done, print a short summary: files created + pytest result.
