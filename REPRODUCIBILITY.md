# Reproducing the paper

This guide maps *When Can Finite-Window QELMs Emulate Quantum Reservoirs?*
to its numerical inputs, protocols, and reproduction commands. Paths are
relative to the repository root.

## Evidence and study identity

| Result | Protocol and configuration | Results and implementation |
| --- | --- | --- |
| Scalar window optimum, observable forgetting, constructive QELM approximation | Paper Appendices A–C; `CANONICAL_EMULATION_THEORY.md` | `pra_extension/code/bernstein_qelm_validation.py`, `sharpness_benchmarks.py`; `pra_extension/outputs/bernstein_qelm_validation_v1/`; `paper/figures/sharpness/` |
| Canonical history/feature frontier and classical controls | `CANONICAL_EMULATION_PREREGISTRATION.md`; `configs/canonical_emulation.yaml` and `_freeze.json`; roots `confirm01`–`confirm12` | `results_canonical_confirmatory_v1/`; `scripts/run_canonical_emulation.py`, `evaluate_canonical_gates.py`; `src/qrc_vs_qelm/canonical_*.py` |
| Dense N=2/3 replication | `DENSE_PHASE_SECONDARY_PROTOCOL.md`; `configs/canonical_emulation_dense_secondary.yaml` and `_freeze.json`; roots `dense01`–`dense12` | `results_dense_phase_secondary_v1/` and `_analysis/`; `scripts/{run,analyze}_dense_phase_secondary.py` |
| Separate N=4 replication | `DENSE_PHASE_N4_PROTOCOL.md`; `configs/canonical_emulation_dense_n4.yaml` and `_freeze.json`; roots `dense_n4_01`–`dense_n4_12` | `results_dense_phase_n4_v1/` and `_analysis/`; `scripts/{run,analyze}_dense_phase_n4.py` |
| Separate N=5 longer-context follow-up | `LONG_CONTEXT_N5_PROTOCOL.md`; `configs/canonical_emulation_long_n5.yaml` and `_freeze.json`; roots `long_n5_01`–`long_n5_12` | `results_n5_long_context_v1/{run,analysis}/`; `scripts/{run,analyze}_long_context_n5.py` |
| Protected-memory construction and perturbations | Paper Appendix D; `configs/DARK_SECTOR_VALIDATION_PROTOCOL.md`, `configs/dark_sector_validation.json` | `pra_extension/code/run_dark_sector.py`; `pra_extension/outputs/dark_sector_validation/` |
| Post-confirmatory memory, finite-shot, continuous-input, architecture, and finite-prefix checks | Executed profiles/seeds in `pra_extension/code/` and run summaries | Current inputs and hashes: `pra_extension/PRA_EXTENSION_RELEASE.json`; runner commands below |
| Laser-driven emulation and NARMA10 prediction | Paper Appendix E.5; source ledgers record seeds and display selection | `pra_extension/code/{laser_demonstration,narma_memory,narma_dense_replay}.py`; `paper/figures/{memory_prediction,narma_dense_replay}/` |

The cohorts have separate roots and are not pooled. `root_id` is the independent
replication unit; windows, feature budgets, outputs, and time rows within one
root are not independent replicates. N=4 was specified after inspecting N=2/3;
N=5 is a subsequent follow-up. Extension runs labelled
`EXPLORATORY_POST_CONFIRMATORY` are separate from the original G0–G6 gates.
`pra_extension/PROTOCOL.md` describes a prospective design, not those executed
extensions. Older `PROTOCOL.md`, `PREREGISTRATION.md`, and recurrence-fingerprint
results describe separate historical studies.

Each configuration records target/student couplings and input, prefix, suffix,
and classical-feature seeds separately. Bootstrap seeds are in the analysis
code or configuration. Preserve these identities and the original decision
rules when reproducing results. A rerun of opened roots is not a fresh test.
Local freeze hashes establish file identity; they do not independently certify
registration time or that roots were previously unopened. No third-party
timestamped registration receipt is supplied.

## Available files and limits

Canonical, dense N=2/3, dense N=4, and N=5 root-level CSV/JSON results are tracked.
The N=5 directory contains **12 tracked run/analysis files**, including stream
metrics, suffix decomposition, system diagnostics, contrasts, and summaries.
Selected authoritative `pra_extension/outputs/` files are also tracked despite
the general ignore rule. The release manifest distinguishes current evidence
from historical regression fixtures. Check inclusion with `git ls-files PATH`.

Not every transient trajectory, feature cache, or execution `.npz` is supplied.
N=5's strict runner **and analyzer** also read the original absolute smoke/JUnit
paths stored in its freeze. Those machine-local records are not delivered by
the tracked result tables; their presence in a manifest does not make a fresh
clone a portable full rerun. Preserve frozen checks rather than bypassing them.
Dependencies are declared but not fully version-locked; original runtime
versions are recorded in run manifests. PDF source ZIPs reproduce manuscript
builds, not numerical experiments.

## Interpreting the empirical decomposition

For each common suffix, the target is evaluated after several old prefixes;
the window model gives the same prediction to all members of that group.

| Field | Meaning |
| --- | --- |
| `history_mse` | Mean squared deviation of target outputs from their sampled group mean. |
| `representation_mse` | Mean squared difference between that mean and the fitted window-model prediction. |
| `total_mse` | Sum of those two components, up to numerical round-off. |
| Corresponding `*_nrmse` | `sqrt(MSE / target_variance)`, with the same variance for every component. MSEs, not NRMSEs, add. |

`representation_mse` is a **fitted-emulator residual**, not a best-in-class
approximation error. It includes finite features, fitting, regularization, and
finite training/prefix sampling. Replay uses a different representation; its
`oracle_nrmse` field is not an optimal-readout certificate for the QELM.
At four prefixes, the variance divisor is four for the exact empirical identity;
the separate `4/3` population-numerator correction is not applied to that identity.

For live targets (`target_variance >= 1e-8`), `REPRESENTATION_LIMITED` requires
history NRMSE ≤0.10, fitted QELM NRMSE >0.10, and replay NRMSE ≤0.10.
If replay fails, the label is `UNRESOLVED`. `HISTORY_IMPOSSIBLE` concerns the
sampled panel and tested window, not a universal worst-case proof. Aggregate
success requires at least 10 of the fixed 12 roots. The reported 67.3% feature
gain is G5's root-median reduction in fitted `representation_nrmse` from 18 to
72 features at `gamma=4, d=6`, aggregated within roots across N=2/3.

## Commands

Use Python 3.11+ and run from the repository root. PDF/figure checks also need
LaTeX, `latexmk`, BibTeX, and Poppler. Choose new, empty output directories.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" -r requirements-ci.txt
.venv/bin/pytest -q
.venv/bin/pytest -q pra_extension/tests/test_release_manifest.py::test_checked_in_release_manifest_matches_live_inputs
```

Canonical and dense reproduction interfaces (the full profiles require matching
frozen material, including hash-bound documentation):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_canonical_emulation.py --profile smoke --out reproduction/canonical_smoke
PYTHONPATH=src .venv/bin/python scripts/run_canonical_emulation.py --profile confirmatory --out reproduction/canonical
PYTHONPATH=src:scripts .venv/bin/python scripts/run_dense_phase_secondary.py --out reproduction/dense
PYTHONPATH=src:scripts .venv/bin/python scripts/analyze_dense_phase_secondary.py --results reproduction/dense --output-dir reproduction/dense_analysis
PYTHONPATH=src:scripts .venv/bin/python scripts/run_dense_phase_n4.py --out reproduction/n4
PYTHONPATH=src:scripts .venv/bin/python scripts/analyze_dense_phase_n4.py --results reproduction/n4 --output-dir reproduction/n4_analysis
```

Full canonical gate re-evaluation additionally requires source-bound JUnit
and two independent same-material smoke runs; see the canonical preregistration
and `scripts/evaluate_canonical_gates.py --help`. Do not create a new freeze to
make changed material pass. N=5's interfaces, subject to the portability limit above:

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/run_long_context_n5.py run --profile long_n5 --out reproduction/n5
PYTHONPATH=src:scripts .venv/bin/python scripts/analyze_long_context_n5.py --results reproduction/n5 --out reproduction/n5_analysis
```

Deterministic construction checks and separate exploratory runs:

```bash
.venv/bin/python pra_extension/code/run_bernstein_qelm_validation.py --output-dir reproduction/bernstein
.venv/bin/python pra_extension/code/run_dark_sector.py --output-dir reproduction/dark_sector
.venv/bin/python pra_extension/code/render_sharpness_figure.py --output-dir reproduction/sharpness
.venv/bin/python pra_extension/code/run_observable_memory.py --profile full --output-dir reproduction/observable
.venv/bin/python pra_extension/code/run_observable_memory.py --profile continuous --output-dir reproduction/continuous
.venv/bin/python pra_extension/code/run_finite_shots.py --profile full --output-dir reproduction/shots
.venv/bin/python pra_extension/code/run_finite_shots.py --profile n4 --output-dir reproduction/n4_shots
.venv/bin/python pra_extension/code/run_finite_shots_eiv.py --profile full --output-dir reproduction/eiv
.venv/bin/python pra_extension/code/run_student_architecture.py --profile full --output-dir reproduction/architecture
```

Matching `analyze_*.py` scripts in `pra_extension/code/` expose inputs through
`--help`; use the release manifest's source hashes to select the recorded runs.
Sharpness checks use constructed readouts and exact expectations, not learned
readouts or an efficiency benchmark. Figure 1 traces are schematic illustrations;
scientific plots use their source tables and ledgers, not digitized values.

Rebuild both manuscript editions from the committed figures:

```bash
.venv/bin/python paper/icaart2027/build_submission.py
.venv/bin/python paper/arxiv/build_arxiv.py
```

The builders check page limits and compile extracted source ZIPs. After an
intentional source/artifact update, regenerate the scientific inventory with
`.venv/bin/python pra_extension/code/build_release_manifest.py --figures-dir pra_extension/figures_pra_final_v6`.
