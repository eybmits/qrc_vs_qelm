# AGENTS.md - qrc_vs_qelm

You implement one work package at a time. `PROTOCOL.md` and `PREREGISTRATION.md`
are the contract; do not change comparison conditions, fairness rules, gates, or
win conditions unless the orchestrator explicitly assigns an amendment work package.

## Rules

- **Package:** `src/qrc_vs_qelm/`, importable as `qrc_vs_qelm`. Python >=3.11.
- **Determinism:** seed every reservoir, channel, feature map, split, and readout
  initialization. Accept an explicit `seed` or `rng` argument; never rely on global
  RNG state.
- **No leakage:** fit all standardization, scaling, basis construction, and model
  selection state on train data only. Apply the fitted objects to validation and
  test data. Tests must guard train-only standardization.
- **Matched budget:** QRC and QELM comparisons must use matched resource budgets:
  comparable feature counts, readout protocol, train/validation/test rows, shot
  budgets, hyperparameter-search budget, and reporting rows. Do not give one model
  extra features, extra tuning, or extra data.
- **Results hygiene:** never commit `results_*/`, `data/`, `.npz` artifacts, or
  generated experiment outputs.
- **Tests:** add or extend `tests/` for each module you implement. Run `pytest -q`
  before declaring done.
- **Claims:** implementers make no scientific-results claims in code, docs, commit
  messages, or handoff notes. Report only verified engineering status unless the
  pre-registered gates have been executed and reviewed.
- **Reuse:** prefer porting proven patterns over reinvention:
  - `qrc_dissipation_engineering/src/qrc/` for the Lindblad core simulator.
  - `qrc_dataset_profiler/src/qrc_dataset_profiler/reservoir.py` for shot-noise
    patterns.
  - `qrc_dataset_profiler/src/qrc_dataset_profiler/baselines.py` for split and
    ridge-readout patterns.
  - `qrc_sym/experiments/paper_experiments/statistics_utils.py` for paired stats.
  - `qrc_attribution_audit` for seed, gate, and manifest patterns.

## Stubs

Modules may raise `NotImplementedError` with the intended signature in the
docstring. Replace the body, keep or refine the signature, and update exports when
the assigned work package requires it.

## Done

`pytest -q` green; no results directories or generated artifacts committed; no
scientific-results claims added by implementers.
