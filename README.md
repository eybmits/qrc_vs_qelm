# qrc_vs_qelm

[![CI](https://github.com/eybmits/qrc_vs_qelm/actions/workflows/ci.yml/badge.svg)](https://github.com/eybmits/qrc_vs_qelm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
![tests](https://img.shields.io/badge/tests-414%20passing-brightgreen.svg)

A study of when a reset-window quantum extreme learning machine can reproduce a
recurrent quantum reservoir.  If `D_d` is the largest output gap between
histories with the same latest `d` inputs, the best unrestricted suffix
predictor has uniform error exactly `D_d/2`.  Finite-window approximation is
therefore possible precisely when this observable memory vanishes.

For uniformly contractive, input-Lipschitz QRCs, the positive direction is
constructive.  History replay contributes `2 M eta^d`; a standard parallel
Bernstein quantum feature map contributes
`M L_u (1-eta^d)/((1-eta)sqrt(r))`; finite shots add a separate Hoeffding term.
This gives explicit window, qubit, outcome, and measurement resources for a
growing reset-QELM family.  It is a one-way observable approximation theorem,
not fixed-architecture equivalence or efficient learnability.

The specifically quantum contribution is a two-qubit open-system realization.
An addressable control changes a singlet population, collective symmetry
protects it under the allowed later inputs, and two-body Pauli correlations
make it visible.  The paper compares information access, not fundamental
computational power, and claims no quantum advantage.

## Status: approximation theorem, protected-memory certificate, and phase map

The frozen numerical study asks when a parallel time-delayed QELM can reproduce
the measured trajectory of recurrent QRC teachers.  Its registered verdict is
`CONFIRMS_FINITE_RESOURCE_FRONTIER`.  A separate frozen recurrence-fingerprint
addendum remains a fully reported `NO_GO`.

The paper fixes one two-qubit teacher and one time-homogeneous input alphabet
with a symmetric drive and an addressable-control flag.  It then fixes one pair
of infinite trajectories: the pair differs once in the flag and shares every
later symmetry-preserving input.  The singlet projector is conserved throughout
every allowed suffix, while the earlier flag changes its population.  The same
pair therefore witnesses every finite reset window.  At
the registered one-radian pulse the output gap is `0.1149244235`, the worst-case
absolute-error floor is `0.0574622`, and the balanced two-history MSE floor is
`0.0033019058`.  The statement covers every stateless prediction law
conditioned only on the finite raw-input suffix.  It does not cover stateful
students, supplied initial states, prefix summaries, branch labels, external
memory, or arbitrary symmetry-breaking noise.

Exact symmetry is not required for a finite certificate. If suffix channels have
uniform per-step observable-protection defect `eps_prot`, every common suffix
retains a gap of at least `max(0, |Delta p_0| - 2 d eps_prot)`.  This turns exact
protection, imperfect protection, and mixing into a single observable-memory
classification.

## Canonical QRC-to-QELM emulation result

**Exact interface criterion, constructive QELM upper bound, and protected
obstruction:** one recurrent QRC lies beyond every reset-based finite-window
learner on its declared input language, whereas every uniformly contractive,
input-Lipschitz QRC observable admits a finite-accuracy reset-QELM
approximation.

- The exact scalar minimax law is `E_d = D_d/2`; hence `D_d -> 0` is necessary
  and sufficient for uniform approximation by unrestricted suffix functions.
- Combining contraction with the established Bernstein quantum feature map
  gives `d*r` qubits, `(r+1)^d` grouped outcomes, explicit accuracy choices for
  `d` and `r`, and a finite-shot concentration term.  The coefficient table is
  exponentially large in `d` and uses target-specific node values.
- A deterministic one-qubit QRC audit matches statevector and binomial feature
  probabilities to `8.33e-17`.  Its Bernstein error falls by 96.72% from
  `r=1` to `r=32`, while replay error falls by 99.9615% from `d=1` to `d=8`;
  every value lies below its certificate.
- The registered finite-prefix panel uses an exact empirical decomposition.
  Its separate population-numerator calibration is `K/(K-1)=4/3` at `K=4`;
  the manuscript does not conflate the two estimands.
- The frozen confirmatory run contains 12 independent root blocks, 144 live
  two- and three-qubit teacher conditions nested within them, and 5,760 fitted stream cells. All primary
  gates G0--G6 pass and the complete suite passes 414/414 tests.
- Increasing the registered window from 1 to 6 reduces QELM NRMSE by a median
  78.5% on transition cells; after history is feasible, increasing the nested
  feature budget from 18 to 72 reduces representation NRMSE by 67.3%. Both
  effects are positive in all 12 root blocks and replicate by teacher size.
- The quadratic classical delay model is the strongest registered control in
  all 24 fast-forgetting root--teacher-family cells and the tested QELM loses to it in
  every root.  This is evidence for the information-interface interpretation,
  not quantum advantage.

The publication artifact is the single manuscript
`paper/canonical_emulation/main.tex`, compiled as
`paper/canonical_emulation/main.pdf`.  The paper is titled *Reset-Window
Emulation of Quantum Reservoirs: Constructive Approximation and
Protected-Memory Limits*.  It is 12 pages, uses 50
cited references, contains exactly three main figures, and includes proofs,
calibration, numerical controls, Methods, and the technical appendix in the
same PDF.  The frozen
protocol, theory note, and literature audit are
`CANONICAL_EMULATION_PREREGISTRATION.md`, `CANONICAL_EMULATION_THEORY.md`, and
`CANONICAL_EMULATION_LITERATURE_POSITIONING.md`. Final evidence is in
`results_canonical_confirmatory_v1/`.  The three deterministic source-backed
paper figures are in `paper/figures/information_interface/`; the detailed
source plots remain in `paper/figures/canonical_emulation/`.

## Post-confirmatory mechanism extensions

The repository also contains a more detailed observable-memory
characterization.  In traceless
Pauli coordinates, for suffix transfer `T_s`, old-prefix covariance `Sigma_s`,
and declared measurement projection `M`,

```text
H(s) = ||M T_s Sigma_s^(1/2)||_F^2 / n_outputs
```

is the minimum population MSE of any predictor that receives only the suffix.
It vanishes exactly when the occupied prefix directions lie in the kernel of
`M T_s`.  For a frozen affine QELM readout, finite measurements add a separate
exact covariance-weighted penalty.  This gives three diagnosed failure modes:
missing history, ideal representation/readout error, and measurement
resolution.

Several 12-root stress-test arms were run only after the canonical result was frozen:

- Across `N_T={2,3,4}` and a declared matched-jump convention for local loss,
  collective loss, and dephasing, the same 79 common-live endpoint cells give
  root-median reductions 48.7%, 55.0%, and 9.0% from `d=1` to `d=8`. Local
  and collective targets remain live in every row; dephasing clears the fixed
  variance floor in only 413/1,296 rows. At
  `gamma=4,d=8`, collective loss retains a strong size-dependent memory that
  local loss does not.
- With a fixed total student-shot budget, validation most often selects
  `B=36` at 300 shots (17/24 paired root-teacher cells) but `B=72` at 19,200
  shots (22/24). Thus more ideal features help only when measurements resolve
  them. Joint multinomial simulation agrees with the analytic shot penalty
  (median ratio 1.001 after enforcing globally unique entropy tuples).
- A disjoint exploratory 12-root `N_T=4` shot arm preserves the crossover. Validation
  selects `B=36` in 12/12 roots at 300 shots, then `B=72` in 11/12 roots at
  both 4,800 and 19,200 shots. The exact-feature `B=72`-over-`B=18` gain is
  60.55% (95% root-bootstrap interval 46.76--65.50%).
- A 324-cell mid-drive spectrum audit finds traceless spectral radii near one
  under collective loss, but an exactly peripheral `N_T=2` eigenvalue coexists
  with negligible sampled post-suffix covariance. Together with the exact
  Pauli-Parseval visibility fraction, this demonstrates that channel spectrum,
  input-ensemble excitation, and measured accessibility are distinct resources.
- A fresh 12-root arm with unsnapped continuous inputs preserves the
  dissipator separation at matched `gamma=4`: root-median `d=1 -> 8`
  reductions are 99.94% (local loss), 88.06% (collective loss), and 7.89%
  (dephasing). The effect is therefore not specific to the 13-symbol grid,
  although its quantitative boundary remains input-distribution dependent.
- With independently shot-noisy train, validation, and test QELM features, a
  validation-selected `N_T=4` policy improves from test NRMSE 0.231 at 300
  shots to 0.0800 at 19,200 shots, a 65.2% root-median reduction.
- The `B=18 -> 72` gain is positive in all 144 root-teacher-topology cells
  across dense, chain, ring, and star six-qubit QELMs. A validation-selected
  topology still loses to quadratic delay in all 36 held-out comparisons.

These are exact-simulation mechanism tests, not hardware evidence,
confirmatory extension gates, universality of the fixed six-qubit QELM, or
quantum advantage. The quadratic classical delay control remains stronger than
the tested QELM in every frozen comparison.

Authoritative extension artifacts are:

- `pra_extension/theory_and_literature.md`
- `pra_extension/outputs/observable_memory_full_exploratory_v3_rigorous_12roots/`
- `pra_extension/outputs/finite_shots_full_exploratory_v4b_refit_design_12roots/`
- `pra_extension/outputs/channel_spectrum_full_exploratory_v1_12roots/`
- `pra_extension/outputs/observable_memory_continuous_exploratory_v1_12roots/`
- `pra_extension/outputs/continuous_input_analysis_v1_strict/`
- `pra_extension/outputs/finite_shots_n4_exploratory_v2b_refit_design_12roots/`
- `pra_extension/outputs/finite_shots_n4_analysis_v2_refit_design_strict/`
- `pra_extension/outputs/finite_shots_eiv_n4_exploratory_v2_final_helpers_12roots/`
- `pra_extension/outputs/finite_shots_eiv_n4_analysis_v6_final_helpers_strict/`
- `pra_extension/outputs/student_architecture_full_exploratory_v1_12roots/`
- `pra_extension/outputs/student_architecture_analysis_v1_strict/`
- `pra_extension/outputs/finite_k_history_calibration_v1_strict/`
- `pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/`
- `pra_extension/outputs/dark_sector_validation/`
- `pra_extension/outputs/bernstein_qelm_validation_v1/`
- `configs/dark_sector_validation.json`
- `configs/DARK_SECTOR_VALIDATION_PROTOCOL.md`
- `pra_extension/figures_pra_final_v6/`
- `paper/figures/compact_pra/`
- `paper/figures/information_interface/`
- `pra_extension/PRA_EXTENSION_RELEASE.json` (fail-closed release record)
- `paper/canonical_emulation/main.pdf` (compiled single paper)

Reproduce the extension without altering the frozen configuration, executable
code, preregistration, or stored confirmatory results:

```bash
.venv/bin/pytest -q pra_extension/tests
.venv/bin/python pra_extension/code/run_bernstein_qelm_validation.py \
  --output-dir <new-empty-bernstein-directory>
.venv/bin/python pra_extension/code/run_dark_sector.py \
  --config configs/dark_sector_validation.json \
  --output-dir <new-empty-dark-sector-directory>
.venv/bin/python pra_extension/code/render_dark_sector_figure.py \
  --data <dark-sector-trajectories.csv> --output-dir <figure-directory>
.venv/bin/python pra_extension/code/render_compact_phase_map.py \
  --source paper/figures/canonical_emulation/fig04_regime_map_source.csv \
  --output-dir <figure-directory>
.venv/bin/python pra_extension/code/render_information_interface_figures.py \
  --output-dir paper/figures/information_interface
.venv/bin/python pra_extension/code/analyze_finite_k_history_calibration.py \
  --output-dir <new-empty-finite-k-directory>
.venv/bin/python pra_extension/code/run_observable_memory.py \
  --profile full --output-dir <new-empty-observable-directory>
.venv/bin/python pra_extension/code/run_finite_shots.py \
  --profile full --output-dir <new-empty-shot-directory>
.venv/bin/python pra_extension/code/analyze_pra_extension.py \
  --observable <observable-results.csv> --shots <finite-shot-results.csv> \
  --output-dir <new-empty-analysis-directory>
.venv/bin/python pra_extension/code/analyze_channel_spectrum.py \
  --profile full --input-value 0.5 \
  --output-dir <new-empty-spectrum-directory>
.venv/bin/python pra_extension/code/run_observable_memory.py \
  --profile continuous --output-dir <new-empty-continuous-directory>
.venv/bin/python pra_extension/code/analyze_continuous_memory.py \
  --observable <continuous-results.csv> --run-summary <continuous-summary.json> \
  --output-dir <new-empty-continuous-analysis-directory>
.venv/bin/python pra_extension/code/run_finite_shots.py \
  --profile n4 --output-dir <new-empty-n4-shot-directory>
.venv/bin/python pra_extension/code/analyze_n4_shots.py \
  --shots <n4-finite-shot-results.csv> --run-summary <n4-shot-summary.json> \
  --output-dir <new-empty-n4-analysis-directory>
.venv/bin/python pra_extension/code/run_finite_shots_eiv.py \
  --profile full --output-dir <new-empty-eiv-directory>
.venv/bin/python pra_extension/code/analyze_eiv_shots.py \
  --shots <eiv-results.csv> --run-summary <eiv-summary.json> \
  --output-dir <new-empty-eiv-analysis-directory>
.venv/bin/python pra_extension/code/run_student_architecture.py \
  --profile full --output-dir <new-empty-architecture-directory>
.venv/bin/python pra_extension/code/analyze_student_architecture.py \
  --results <architecture-results.csv> --profile full \
  --output-dir <new-empty-architecture-analysis-directory>
.venv/bin/python pra_extension/code/render_pra_extension_figures.py \
  --observable <observable-results.csv> --shots <finite-shot-results.csv> \
  --observable-root-effects <observable-root-effects.csv> \
  --spectrum <channel-spectrum.csv> --continuous <continuous-results.csv> \
  --continuous-root-effects <continuous-root-effects.csv> \
  --n4-shots <n4-finite-shot-results.csv> --n4-summary <n4-analysis.json> \
  --architecture <architecture-results.csv> \
  --architecture-analysis <architecture-analysis.json> \
  --architecture-selected-effects <architecture-selected-effects.csv> \
  --eiv-shots <eiv-results.csv> --eiv-analysis <eiv-analysis.json> \
  --eiv-root-effects <eiv-root-effects.csv> \
  --eiv-selected-rows <eiv-validation-selected-rows.csv> \
  --summary <analysis.json> \
  --output-dir <new-empty-figure-directory>
```

`pra_extension/PROTOCOL.md` and its YAML files are a draft prospective design
for a future, larger continuous-input study. The current runners do not execute
its P0--P6 gates, so it must not be described as frozen or completed.

Rebuild the paper and then regenerate the fail-closed release record only after
all source, analysis, and figure inputs are final:

```bash
(cd paper/canonical_emulation && \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
.venv/bin/python pra_extension/code/build_release_manifest.py \
  --figures-dir pra_extension/figures_pra_final_v6
```

The release builder verifies the frozen canonical identity and every declared
paper/figure dependency before writing a new manifest.

The recurrence-fingerprint addendum was run once on 24 untouched systems. Its
balanced accuracy is 0.8327, below the frozen 0.85 threshold. The failure is
informative and fully reported: 895/960 primary decisions are correct, 65
abstain, no decisive errors are observed, and 767/768 finite cutoffs are
recovered within one step. All eight dephasing QRC systems are below the frozen
observable-signal floor at exact precision. An independent task-ordering gate
passes strongly.

## Findings (N=5, confirmatory run)

![Memory×nonlinearity phase diagram: QRC vs QELM advantage regions](paper/figures/phase_diagram_clean.png)

*RQ1 headline: the memory×nonlinearity architecture map. Red = QRC-favored;
blue = QELM-favored; cross-hatching marks cells where both models have mean
capacity below 0.05.*

- **RQ1 — memory×nonlinearity architecture map (GO on the informative set).**
  Seven of 15 informative cells are significantly QRC-favored, five are
  QELM-favored, and three are ties. QELM is favored on selected short-delay
  nonlinear targets; QRC is favored on selected intermediate- and longer-delay
  targets. On the full 27-cell grid, including 12 cells where both models have
  mean capacity below 0.05, the registered result is `SINGLE_REGIME`. Both
  denominators are reported. QELM also beats the raw linear-delay control on the
  registered fraction of nonlinear cells, establishing a nontrivial encoded
  feature map but not a uniquely quantum advantage.
- **RQ2 — finite shots (no directional claim).** The committed figure and G2a
  record use incompatible favored-cell estimands and give opposite directional
  summaries. Stage 2 must be re-evaluated under one explicitly frozen estimand.
- **RQ3 — dissipation (exploratory after `NO_TRADEOFF`).** The registered gate
  fails. The observed QRC and QELM response curves are retained as conditional,
  post-hoc descriptions, not as a universal dissipation mechanism.

## Recurrence fingerprint (N=2–4, untouched held-out run)

![Observable-memory visibility and held-out classification accounting](paper/figures/fingerprint_visibility.png)

- **Overall:** `NO_GO`; balanced accuracy 0.8327 is below 0.85.
- **Safety on this panel:** 895 correct, 65 abstentions, and 0 observed wrong
  decisive labels across 960 primary 10,000-shot decisions.
- **Finite-window recovery:** 767/768 cutoffs recovered within one step and no
  false-recurrence calls.
- **Visibility boundary:** 8/8 dephasing QRC blocks are low-signal under the
  frozen Pauli readout; 16/16 local/collective-loss blocks are resolved. This
  partition is descriptive, not a registered subgroup claim and not evidence
  that dephasing erases all memory.
- **Independent task link:** crossover shift 0.5364, 95% cluster interval
  [0.4333, 0.6448], with 23/24 positive shifts.

The current paper source is `paper/main.tex`; `paper/manuscript_draft.md` is a
plain-language evidence synopsis. See `paper/figures/` for figures and their
source tables.

## Repo layout

- `src/qrc_vs_qelm/` — package: `core/` (ported Lindblad simulator), `models/`
  (QRC + QELM), `matching.py`, `pipeline.py`, `tasks.py`/`task_families.py`,
  `controls.py`, `seeds.py`, `gates.py`, `stats.py`, `run_stage{1,2,3}.py`, plus
  the content-addressed `canonical_emulation.py`, `canonical_study.py`,
  `canonical_gates.py`, `canonical_resources.py`, and `provenance.py`.
- `scripts/` — validation/study runners plus deterministic paper-figure scripts.
- `configs/` — frozen `full.yaml` and scaled-down `smoke.yaml`.
- `data/frozen/` — the frozen operating-point config (single source of truth).
- `PROTOCOL.md` / `PREREGISTRATION.md` — frozen methods and commitments + amendment log.
- `paper/` — current LaTeX paper, legacy Markdown draft, and committed figures.
- `.specs/` — per-work-package implementation specs (development provenance).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Reproducing the study

The current canonical study has separate smoke, calibration, and frozen
confirmatory profiles. A same-material confirmatory rerun is a reproducibility
check only; its roots have already been opened and must never be used to retune
the registered decision rules.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_canonical_emulation.py \
  --profile smoke --out results_canonical_smoke_reproduction

PYTHONPATH=src .venv/bin/python scripts/run_canonical_emulation.py \
  --profile confirmatory --out results_canonical_confirmatory_reproduction

PYTHONPATH=src .venv/bin/python scripts/render_canonical_emulation_figures.py \
  --results results_canonical_confirmatory_reproduction
```

See `CANONICAL_EMULATION_PREREGISTRATION.md` for the complete bound test,
duplicate-smoke, evaluator, and freeze-verification procedure.

### Earlier architecture phase-map study

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

The recurrence addendum is governed by
`RECURRENCE_FINGERPRINT_PREREGISTRATION.md` and
`data/frozen/recurrence_fingerprint_v1.json`. The held-out blocks H01–H24 have
already been opened and must not be rerun after analysis changes. New classifier,
observable, or threshold variants require fresh systems and seeds. Paper-only
figures can be regenerated safely from the sealed outputs:

```bash
python scripts/render_recurrence_paper_figures.py
```

Use `--profile smoke` for a fast (minutes) end-to-end check at N=4.

## Reproducibility & governance

The pre-registration (`PREREGISTRATION.md`) freezes the operating point, gate
thresholds, seed grids, and analysis plan, with a full amendment log of the
design-pilot trail (on disjoint seeds) and every pre-execution correction. The
`stage1-frozen` git tag marks the frozen plan.

`AGENTS.md` / `CLAUDE.md` document the AI-assisted development workflow (planner +
implementer + reviewers) used to build the package.

## Citation

If you use this software or its results, please cite it (see `CITATION.cff`):

```bibtex
@software{baumann_qrc_vs_qelm_2026,
  author  = {Baumann, Markus},
  title   = {qrc\_vs\_qelm: A prospectively frozen, resource-matched comparison of
             Quantum Reservoir Computing and Quantum Extreme Learning Machines},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/eybmits/qrc_vs_qelm}
}
```

## License

Released under the [MIT License](LICENSE).
