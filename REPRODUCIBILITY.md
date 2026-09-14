# Reproducibility and evidence guide

This is the repository guide for *When Can Finite-Window QELMs Emulate
Quantum Reservoirs?* It maps the paper's claims to their original
protocols, deterministic inputs, root-level results, and validation code.
It does not amend a frozen protocol or turn a post-confirmatory analysis into
a prospective result. The [README](README.md) contains the numerical overview
and the longer command inventory.

## What a reader can access

Availability audit: 7 September 2026, against the local Git index. The configured
repository is <https://github.com/eybmits/qrc_vs_qelm>; remote visibility, uploaded
contents, and anonymous reviewer access have **not** been verified here.

- The canonical confirmatory and dense N=2/3 and N=4 CSV/JSON files mapped below
  are tracked. They include root-by-condition metrics, not every transient
  trajectory, feature cache, or `.npz` file from execution.
- Selected authoritative `pra_extension/outputs/` CSV/JSON files are tracked
  despite the general ignore rule. The release manifest identifies these
  selected inputs and distinguishes historical regression fixtures from
  current evidence. Other local generated outputs must not be assumed present
  in a clone.
- `results_n5_long_context_v1/` is ignored and has **no tracked files**, including
  its raw tables, smoke evidence, JUnit report, and `evidence_bundle.zip`.
  The N5 protocol/configuration/runner and current paper additions are also
  working-tree additions at this audit, not yet a verified remote handoff.
  Thus the current repository link alone does not deliver the full N5 evidence.
  The existing local ZIP would need to be supplied through an authorized
  accessible channel, together with its receipt/hash; none is published here.
- `paper/icaart2027/ICAART2027_anonymous_source.zip` contains only the 19 inputs
  needed to compile that PDF. Its clean-extraction check establishes manuscript
  build reproducibility, not scientific-result reproduction. It is not a
  numerical review bundle; see the [venue README](paper/icaart2027/README.md)
  for the separate submission/disclosure restrictions.

A hash inventory is not a file server. Check actual availability with
`git ls-files <path>` and `git check-ignore -v <path>` rather than inferring it
from a path appearing in a manifest. Before a review handoff, supply the missing
evidence and verify access from the intended reader's context; this guide does
not claim that either step has occurred.

## Claim-to-evidence map

Paths below are repository-relative. Cohorts have separate roots and must not
be pooled. In the canonical and dense tables, `root_id` is the independent
replication unit; windows, feature budgets, loss rates, outputs and time rows
within a root are not independent replicates.

| Claim / evidence layer | Protocol, configuration and seeds | Results and executable audit |
| --- | --- | --- |
| Scalar history optimum, output-visible memory criterion, constructive growing-QELM theorem | [Theory note](CANONICAL_EMULATION_THEORY.md), [literature positioning](CANONICAL_EMULATION_LITERATURE_POSITIONING.md), paper proofs and their stated assumptions | These are mathematical claims, not fits to the numerical phase map. The deterministic constructive audit is `pra_extension/code/bernstein_qelm_validation.py`; tables and checks are in `pra_extension/outputs/bernstein_qelm_validation_v1/`. |
| Canonical window gain, 67.3% fitted-emulator residual reduction, classical negative control; G0–G6 | [Canonical preregistration](CANONICAL_EMULATION_PREREGISTRATION.md), `configs/canonical_emulation.yaml`, sibling `_freeze.json`; `profiles.confirmatory.roots`, `confirm01`–`confirm12` | `results_canonical_confirmatory_v1/`: `suffix_decomposition.csv`, `stream_metrics.csv`, `system_diagnostics.csv`, `canonical_block_contrasts.csv`, `canonical_gate_report.json`, run manifest/summary and resource record. Runners/evaluator: `scripts/run_canonical_emulation.py`, `scripts/evaluate_canonical_gates.py`; numerical definitions: `src/qrc_vs_qelm/canonical_emulation.py`, `canonical_study.py`, `canonical_gates.py`. |
| Dense loss/window frontier, N=2/3 secondary cohort | [Secondary protocol](DENSE_PHASE_SECONDARY_PROTOCOL.md), `configs/canonical_emulation_dense_secondary.yaml` and `_freeze.json`; `profiles.dense_secondary.roots`, `dense01`–`dense12` | `results_dense_phase_secondary_v1/` contains root tables and preflight/run records; `results_dense_phase_secondary_v1_analysis/` contains `dense_phase_cells.csv`, `dense_phase_root_scaling.csv`, summary and analysis manifest. Audit: `scripts/{run,analyze}_dense_phase_secondary.py`. |
| Separate N=4 replication after the N=2/3 dense results were inspected | [N4 protocol](DENSE_PHASE_N4_PROTOCOL.md), `configs/canonical_emulation_dense_n4.yaml` and `_freeze.json`; `profiles.dense_n4.roots`, `dense_n4_01`–`dense_n4_12`; freeze includes prior-seed sources | `results_dense_phase_n4_v1/` contains root tables and preflight/run records; `results_dense_phase_n4_v1_analysis/` contains `dense_n4_phase_cells.csv`, `dense_n4_root_scaling.csv`, summary and analysis manifest. Audit: `scripts/{run,analyze}_dense_phase_n4.py`. |
| Separate N=5, longer-context follow-up; larger ten-qubit emulator | [N5 protocol](LONG_CONTEXT_N5_PROTOCOL.md), `configs/canonical_emulation_long_n5.yaml` and `_freeze.json`; `profiles.long_n5.roots`, `long_n5_01`–`long_n5_12`, explicit `analysis_seed` | **Local only:** `results_n5_long_context_v1/run/{suffix_decomposition,stream_metrics,system_diagnostics}.csv`, run/resource records; `analysis/long_n5_cells.csv`, `long_n5_history_root_contrasts.csv`, `long_n5_classical_root_comparisons.csv`, summary/manifest. Code: `scripts/{run,analyze}_long_context_n5.py`, `long_context_n5_common.py`, `long_context_qelm_backend.py`. |
| Protected two-qubit obstruction and perturbation validation | [Dark-sector protocol](configs/DARK_SECTOR_VALIDATION_PROTOCOL.md), `configs/dark_sector_validation.json` including root coupling/suffix seeds | `pra_extension/code/run_dark_sector.py`; `pra_extension/outputs/dark_sector_validation/` contains trajectories, audits, leakage and summary. This numerical validation accompanies a proved construction; it is not a hardware experiment. |
| Post-confirmatory observable-memory, finite-shot, continuous-input, architecture and finite-K checks | [Extension evidence/commands](README.md#post-confirmatory-mechanism-extensions), `pra_extension/theory_and_literature.md`; executed defaults/seeds in `pra_extension/code/{observable_memory,finite_shots,finite_shots_eiv,student_architecture}.py` and run-summary JSONs | Exact current run/analysis directories and per-file hashes are enumerated in `pra_extension/PRA_EXTENSION_RELEASE.json`. Keep `EXPLORATORY_POST_CONFIRMATORY` separate from the original gates. `pra_extension/PROTOCOL.md` and its prospective YAMLs describe a future study, not these executed arms. |

Every canonical/dense/N5 root configuration lists target and student coupling,
train/validation/test input, common-suffix input, old-prefix input, and classical
feature seeds separately. The canonical bootstrap seed is recorded in
`canonical_gates.py`; follow-up analysis seeds are in their configuration or
analysis code and summaries. Use those records, not a single replacement seed.
The older `PROTOCOL.md`, `PREREGISTRATION.md`, and recurrence-fingerprint study
remain historical layers, not alternative definitions of the current canonical
experiment.

## What the empirical residual means

For each common-suffix group, the target is evaluated after several different
old prefixes. Let `group_mean` be the mean target output in that finite group,
and `prediction` the one fitted window-model output shared by its members.
The frozen schema and exact empirical definitions are:

| Original field | Definition / interpretation |
| --- | --- |
| `history_mse` | Mean squared deviation of target outputs from `group_mean`; missing-history ambiguity on the sampled panel. |
| `representation_mse` | Mean squared difference between `group_mean` and `prediction`; the **fitted-emulator residual**, not an infimum over all readouts or all QELMs. |
| `total_mse` | Mean squared target–prediction error; exactly `history_mse + representation_mse` up to numerical round-off. |
| `history_nrmse`, `representation_nrmse`, `total_nrmse` | `sqrt(MSE / target_variance)`, using the same variance for every component. MSEs, not their square roots, add. |

The residual can reflect the chosen finite feature map, finite training data,
readout fitting and validation-selected regularization. This decomposition
does not separately identify their contributions or establish an irreducible
feature-expressivity floor. Finite-shot analyses are separate evidence layers.
The finite-prefix estimate of the suffix mean also contributes sampling
variability to the residual. Replay uses a different representation; its
`oracle_nrmse` field is not a population-optimal readout control for the same
QELM features.
At four prefixes per suffix, the empirical history numerator uses division by
four to preserve the exact identity. The separate `4/3` population-numerator
calibration is not substituted into that identity or silently applied to the
normalized ratios.

For live targets (`target_variance >= 1e-8`) at tolerance 0.10, the original
`REPRESENTATION_LIMITED` label requires all three conditions: `history_nrmse <=
0.10`, fitted QELM `total_nrmse > 0.10`, and same-physics reset-replay
`oracle_nrmse <= 0.10`. If replay also fails, the remaining label is
`UNRESOLVED`, not automatically “representation limited.” `HISTORY_IMPOSSIBLE`
is only the sampled-panel, tested-window classification, not the universal
theorem's worst-case impossibility statement. Aggregate history or emulator
success requires at least 10 of the fixed 12 roots, not merely a median below
the threshold.

The reported **67.3%** is G5's `0.6726414464648288` median relative reduction
in `representation_nrmse`, comparing 18 to 72 features at the prespecified
`gamma=4, d=6`. Gains are computed per root and target size, aggregated within
root across N=2/3, then summarized over the 12 roots. It is an empirical
fitted-emulator residual reduction, not a measured reduction of the theoretical
best-in-class representation error. Original CSV names and gate labels are
retained for byte-level reproducibility.

## Freeze chronology and its limits

The freeze records contain local timestamps, material/configuration hashes,
seed identities and investigator attestations about unopened roots. They
record canonical freezing on 28 August, dense N=2/3 and N=4 on 31 August, and
N5 on 6 September 2026. N4 explicitly acknowledges the already opened N=2/3
results; N5 is a separate subsequent follow-up.

Matching hashes establish file identity and internal consistency. They do
**not** independently certify registration time, prove that no copied checkout
was run earlier, or establish a third-party preregistration. No externally
timestamped registration receipt is supplied here. The run manifests, bound
test evidence and duplicate-smoke records document the local claimed sequence;
their “not previously opened” assertions remain investigator attestations.

The scientific release is a `POST_CONFIRMATORY_REPRODUCIBILITY_SNAPSHOT`.
Its `evidence_layers`, `canonical_frozen_identity` and
`prospective_followups.long_context_n5` keep study identities separate. The
release builder permits disclosed post-freeze edits to the theory and
literature notes only; frozen numerical code/configuration/protocol identities
are not replaced by the current prose. A new release checksum does not
retroactively preregister a result.

## Reproduction and validation

Run from a fresh checkout with Python 3.11 or newer. Dependencies are declared
in `pyproject.toml` (not a fully pinned environment lock); original runtime
versions are recorded in run manifests. Figure/PDF validation additionally
needs LaTeX, `latexmk`, BibTeX and Poppler. Use new, empty output directories;
do not overwrite frozen results or relax a failed integrity check.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" -r requirements-ci.txt
.venv/bin/pytest -q
.venv/bin/pytest -q pra_extension/tests
# Read-only reconstruction/comparison of the complete scientific manifest:
.venv/bin/pytest -q pra_extension/tests/test_release_manifest.py::test_checked_in_release_manifest_matches_live_inputs
```

CI's PDF dependency is in `requirements-ci.txt`, keeping the experiment-bound
`pyproject.toml` unchanged. The figure reader verifies completed feature-bridge
results using `feature_bridge_release.py`. It checks every required frozen
source and production artifact; only the two explicitly named, unpublished
parent development outputs may be absent. Its audit reports those absences
and does not claim a full development-provenance audit. Present development
files must still match their frozen hashes. The original runner's strict
`verify()` and its source lock remain unchanged.

The frozen bridge smoke regression receives a newly generated parent smoke
run through a test-only I/O fixture. Numerical routines and anchor checks are
unchanged; the test does not require untracked historical files or overwrite
production results. This checks fresh-run determinism, not byte-identical
reproduction of historical timing metadata.

The full release check requires **all** declared artifacts, including the
currently local-only N5 evidence; missing-file failure is an incomplete
handoff, not a reason to skip that evidence layer. A current test run checks
current code and is not a replacement for the original source-bound JUnit
record. These instructions do not claim a clean-clone full-suite pass while
the documented files are unavailable.

The [README reproduction commands](README.md#reproducing-the-study) cover
canonical smoke/confirmatory and both dense cohorts. For example:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_canonical_emulation.py \
  --profile smoke --out results_canonical_smoke_reproduction
PYTHONPATH=src:scripts .venv/bin/python scripts/analyze_dense_phase_n4.py \
  --results results_dense_phase_n4_v1 \
  --output-dir results_dense_phase_n4_reproduction_analysis
```

The sealed confirmatory runner/evaluator require the original frozen material;
later edits to even hash-bound documentation can make a fresh execution fail.
Use the matching frozen material, not a new freeze or edited hash, for a
same-material rerun. A complete gate re-evaluation additionally requires a
source-bound full-suite JUnit file and two independent same-material smoke
directories (`scripts/evaluate_canonical_gates.py --help`). A rerun of opened
roots is reproduction only, never a new confirmatory test or tuning set.

N5's command interface is:

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/run_long_context_n5.py run \
  --profile long_n5 --out results_n5_long_context_reproduction
PYTHONPATH=src:scripts .venv/bin/python scripts/analyze_long_context_n5.py \
  --results results_n5_long_context_reproduction \
  --out results_n5_long_context_reproduction_analysis
```

These N5 commands are **not yet portable clean-clone instructions**: the frozen
verification also reads the original absolute smoke/JUnit evidence paths.
The local evidence ZIP preserves those records but implements no relocation
layer. Delivering the archive permits inspection; a portable rerun still needs
an explicit, separately reviewed relocation procedure that preserves the
original records. Neither is performed by this documentation update.

For deterministic construction/obstruction checks and all exploratory runners
and analyzers, use the [extension command inventory](README.md#post-confirmatory-mechanism-extensions).
For figures, use the declared CSVs and recorded source hashes, not values
digitized from plots. Figure 1's Mackey–Glass inputs and independent output
drawing templates are schematic illustrations, not fitted predictions or an
additional experiment. The venue source-build check is documented separately
in [paper/icaart2027/README.md](paper/icaart2027/README.md).
