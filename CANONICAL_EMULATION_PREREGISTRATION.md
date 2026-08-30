# History Before Features: Canonical QRC-to-QELM Emulation Study

**Status:** frozen for confirmation on 28 August 2026. The 12 confirmatory roots
remain unopened until the independently hashed freeze record, full bound test
suite, and two same-material smoke runs are present. The earlier reset-replay
pilot and all `design*` calibration roots are exploratory prior information and
are not part of the confirmatory denominator defined here.

## The simple scientific law we test

> A window model must first contain enough recent history. Only after that
> history is sufficient can a richer feature map improve the approximation.

The study asks when a genuinely nonrecurrent, parallel delay-encoded QELM can
reproduce the observable trajectory of a recurrent QRC, when it cannot, and
whether failure comes from missing history or from inaccessible features.

## Architecture names are fixed

| Label | State carried between rows? | How history enters | Role |
|---|---:|---|---|
| `qrc_teacher` | yes | sequential input-dependent state updates | target |
| `rr_same_identity` | no | last `d` inputs are sequentially replayed through the teacher channel | truncation oracle |
| `ptd_parallel` | no | `d` delayed inputs are encoded once on distinct qubits, then one fixed input-independent channel is applied | primary QELM student |
| `linear_delay` | no | raw delay vector | classical control |
| `poly2_delay` | no | linear and degree-two delay monomials | classical sensitivity control |
| `classical_tanh` | no | fixed random tanh features of the same delay vector | dimension-matched classical ELM |
| `constant` | no | training-target mean | liveness floor |

The legacy repository name `encoding="time_delayed"` denotes reset-and-replay.
It will be reported as `rr_*`, not as the canonical TD-QELM. The primary QELM
uses `encoding="parallel"`: each lag is uploaded to one qubit, unused qubits
remain in `|0>`, the row is reset independently, and only the linear readout is
trained.

This is a controlled **Kawanabe-style fixed-register generalization**, not an
exact replication of that paper's readout protocol: keeping six student qubits
fixed isolates window length from register width, unused qubits are `|0>`, and
the primary readout uses single-site XYZ rather than Z only. A Z-only,
`N_s=d` sensitivity is reserved as a descriptive post-primary check.

## Canonical models

The recurrent teacher is

`rho_t = Phi[u_t](rho_{t-1})`, `y_t = measured Pauli expectations of rho_t`.

Every deterministic length-`d` student instead has the form

`prediction_t = g(u_t, ..., u_{t-d+1})`.

Two complete histories with the same final `d` inputs must therefore receive
the same student prediction, even when the recurrent teacher produces different
outputs.

## Exact, elementary error decomposition

For each suffix `s`, we simulate `K` independent old prefixes followed by that
same suffix. Let `y[s,k]` be the teacher output and `mean_y[s]` their group mean.
A window student makes one prediction `p[s]` for the complete group. Then,
exactly on the sampled panel,

`total MSE = history MSE + representation MSE`, where

- `history MSE = mean ||y[s,k] - mean_y[s]||^2`;
- `representation MSE = mean ||mean_y[s] - p[s]||^2`;
- `total MSE = mean ||y[s,k] - p[s]||^2`.

This is the canonical separation:

- history MSE is a student-independent window floor;
- representation MSE measures what the chosen features and linear readout fail
  to capture after conditioning on the available suffix.

The older half-diameter certificate remains an implementation check and a
worst-pair diagnostic. Passing its triangle inequality is not itself a discovery.

## Possible, impossible, and unresolved

The primary tolerance is normalized RMSE `epsilon = 0.10`. Each held-out cell is
classified using the following frozen order:

1. `DEGENERATE_TARGET`: teacher target variance is below `1e-8`.
2. `HISTORY_IMPOSSIBLE`: history NRMSE is greater than `epsilon`. No
   deterministic model using only this suffix can pass the tolerance on the
   sampled distribution, regardless of its feature count.
3. `EMULATED`: canonical QELM total NRMSE is at most `epsilon`.
4. `REPRESENTATION_LIMITED`: history NRMSE and same-physics oracle NRMSE are at
   most `epsilon`, but the canonical QELM remains above it.
5. `UNRESOLVED`: none of the above.

“Impossible” is always qualified by the tested window, output projection,
input distribution, and tolerance. It is not a theorem that every finite window
fails.

## Simple theoretical predictions

### P1 — common-suffix lower bound (exact proposition)

If two histories share the last `d` inputs but their teacher outputs differ by
`D`, at least one prediction error of every deterministic `d`-window model is
at least `D/2`.

### P2 — fading upper prediction (conditional proposition)

If every common-input teacher step contracts state differences by at most
`kappa < 1`, old-prefix influence decays no slower than `kappa^d`. Same-physics
reset replay must then converge exponentially with `d`.

The present study does not estimate or certify a uniform contraction
coefficient, even though the finite 33-symbol transfer-map family could be
studied separately. It measures common-suffix decay and reports an empirical
exponential fit only; it does not infer uniform contraction from that fit or
from a single constant-input spectral gap.

### P3 — teacher-controlled history scale (falsifiable conjecture)

Increasing teacher dissipation should reduce the minimum window that crosses
the history tolerance. Student architecture or feature budget must not change
the measured history floor.

### P4 — student-controlled feature scale (falsifiable conjecture)

Once the history floor is small, increasing the nested QELM measurement budget
should reduce representation error only when it adds stable, target-aligned
feature directions. Raw feature count alone is not sufficient.

### P5 — no universal quantum-specific advantage

Whether the parallel QELM beats a dimension-matched classical random-feature
ELM is left genuinely open. The finite-resource-frontier claim survives a tie
or classical win; only quantum-specific wording depends on that comparison.

### P6 — dissipation-window scaling (calibration-discovered, prospectively frozen)

The disjoint three-root calibration suggests the simple local-loss law

`history NRMSE approximately A * exp(-c * gamma_T * dt * d)`.

Equivalently, reducing dissipation should require a proportionally longer
window: the history boundary is approximately controlled by the dimensionless
product `gamma_T * dt * d`. This was formulated after inspecting calibration
roots but before opening any confirmatory root, so it is a prospective
secondary prediction rather than part of the original primary gates.

## Frozen primary exact-simulation matrix

The independent statistical unit is a **root block**, never an individual row,
window, gamma value, or student feature seed.

| Axis | Frozen values |
|---|---|
| root blocks | 12 fresh blocks |
| teacher qubits | `N_T = {2,3}` |
| primary teacher dissipator | `{local_loss}` |
| teacher gamma | `{0.0, 0.2, 0.5, 1.0, 2.0, 4.0}` |
| teacher `dt, h, h_in, J_s` | `{0.5, 0.5, 1.0, 1.0}` |
| input alphabet | 33 equally spaced symbols in `[0,1]` |
| target observables | single-site and same-axis two-site Pauli `X,Y,Z` |
| parallel-student qubits | `N_S = 6` |
| parallel-student dynamics | one fixed independent unitary TFIM channel |
| windows | `d = {1,2,4,6}` consecutive recent inputs |
| max snapshot grid | `{T/4, T/2, 3T/4, T}`, `T=0.5` |
| student observables | single-site `X,Y,Z` on six qubits |
| nested feature budgets | `B = {18,36,72}` using `{1,2,4}` nested snapshots |
| stream rows | 480 train, 160 validation, 320 untouched test |
| suffix ensemble | 32 suffix groups, 4 independent prefixes per group |
| prefix burn-in | 128 inputs before the common suffix |
| ridge grid | `{0,1e-10,1e-8,1e-6,1e-4,1e-2,1,1e2,1e4,1e6,+inf}` |
| ridge solver | economy SVD on a fixed resolved subspace, relative cutoff `1e-6` |

Within a root block, teacher coupling and all input streams are held fixed while
gamma changes. The canonical student channel is independent of the teacher and
is reused across the matched teacher conditions in that root block. Root blocks
have mutually disjoint integer seed namespaces.

The unitary statevector QELM backend is used for the primary student. The QRC
teacher remains the validated density-matrix Lindblad simulator.

## Splits and leakage controls

- Train, validation, test, and suffix-ensemble streams reset independently.
- All feature scaling is fitted on training rows only.
- Ridge is selected using validation targets only and refitted on train plus
  validation with the frozen train scaler.
- Every finite ridge candidate uses an economy-SVD solve on the fixed relative
  subspace cutoff `1e-6`, matching the numerical-rank diagnostic. This
  discards floating-point-null feature directions rather than allowing
  round-off differences between independently evaluated but mathematically
  identical suffixes to acquire enormous weights. Positive penalties are
  applied through the standard singular-value factor `s/(s^2+alpha)` on that
  resolved subspace; `+inf` is the exact intercept-only candidate.
- Test and suffix-ensemble targets never select alpha, features, thresholds, or
  model variants.
- Within each root, every suffix group is drawn once at length six; the
  `d={1,2,4}` panels are exact tails of that same suffix. Window contrasts are
  therefore paired across `d` rather than based on different random suffixes.
- The newest suffix symbol is sampled without replacement from the 33-symbol
  alphabet (`G=32` confirmatory groups). Therefore every numerical suffix row
  is distinct at every registered window, so two identical conditioning values
  can never be assigned separate group means in the risk decomposition.
- The largest QELM and classical-tanh feature matrices are generated once per
  root/window and smaller budgets are exact nested column subsets.
- The classical-tanh scaler sees training windows only; its random weights do
  not see targets.
- Constant, linear, polynomial, QELM, and classical-ELM models receive identical
  quantized delay windows and split rows.

## Reported quantities

For every root/system/window/model/budget cell, the deterministic science
tables report:

- held-out MSE, RMSE, NRMSE, and row-RMS mean, median, and maximum;
- suffix history, representation, and total MSE/NRMSE;
- exact decomposition identity error;
- same-physics reset-replay error;
- half-diameter and RMS common-suffix lower witnesses;
- feature count, numerical matrix rank, standardized effective rank,
  resolved-subspace condition number, and target-aligned projection fraction
  on training data. Rank and the resolved subspace use the registered relative
  cutoff `1e-6`; the legacy CSV field `condition_number` stores this
  pseudocondition number rather than the mathematical condition of a singular
  full feature matrix;
- selected ridge penalty and whether the intercept-only limit was selected;
- exact model feature width plus the hashed teacher/student observable
  contracts used to produce the cell.

Null denotes “not applicable” for direct reset-replay and constant predictors;
rank, conditioning, alignment, ridge, and intercept-selection diagnostics are
defined for learned feature/readout models.

A separate non-decision-bearing `computational_resources.json` appendix reports
whole-run wall time and profile-level state-width and algorithmic-depth proxies.
It also separates feature count from a measurement-setting proxy: one global
tensor-product Pauli basis per axis and selected snapshot, with same-axis
single- and two-site observables grouped. Wall time is machine dependent and is
excluded from the duplicated-smoke science-hash equality check; the resource
artifact itself is nevertheless mandatory, strict JSON, and manifest-hashed.

The figure set is fixed as six primary figures plus one supplemental figure:

1. history NRMSE versus window, split by teacher family and gamma;
2. canonical-QELM total error surface over window `d` and budget `B`;
3. at `B=72`, exact stacked decomposition of root-mean total MSE into history
   and student-side representation/readout residual;
4. possible/impossible/representation-limited regime map;
5. held-out **stream NRMSE** for QELM, tanh, polynomial, and linear controls on
   the fixed G5 cells, using the same live/history-feasible mask as G5;
6. per-root/system four-window exponential predicted versus observed minimum
   feasible window, with right-censoring when no registered window is feasible
   and an explicit degenerate/nonpositive-trace category;
7. supplemental: the P6 log-history fit on the exact live positive-loss grid,
   using the configured `R^2` threshold.

## Confirmatory gates

All fractions below use the 12 root blocks as denominator.

- **G0 implementation and hygiene:** full tests pass; expected row coverage;
  strict JSON; hashes recorded; all root seed namespaces disjoint; duplicated
  smoke runs have identical deterministic science hashes. The JUnit suite
  records the exact material/config hashes, and both smoke manifests must match
  the material identity of the evaluated run; same-code agreement alone is not
  accepted as current evidence.
- **G1 decomposition:** absolute identity error is at most `1e-12` in every
  suffix cell; all separately evaluated common-suffix predictions agree; all
  sampled lower bounds hold.
- **G2 mechanism association:** in at least 10/12 roots, the median
  within-system Spearman correlation between suffix-panel history NRMSE and
  reset-replay NRMSE on the independently drawn streaming test set across
  windows is at least `0.8`. A system is eligible only when all four suffix
  panels are live; each root must retain at least 5/6 eligible gamma systems in
  each teacher-size family. Any missing, nonfinite, or constant correlation in
  an eligible system makes that root undefined and unsuccessful.
- **G3 regime coverage:** each `N_T` family contains at least one
  live history-feasible cell and one live history-impossible cell in at least
  10/12 roots.
- **G4 window gain:** on the fixed transition set
  `N_T={2,3}, gamma_T={2,4}`, the `B=72` parallel-QELM median NRMSE
  improvement from `d=1` to `d=6` is at least 20% and is positive in at least
  10/12 roots. A root is eligible only if all four family-by-gamma cells are
  live, history-impossible at `d=1`, and history-feasible at `d=6`; at least
  10/12 roots must be eligible, and an ineligible root counts against the fixed
  12-root denominator.
- **G5 feature gain after forgetting:** on the fixed fast-fading set
  `N_T={2,3}, gamma_T=4, d=6`, the representation-NRMSE improvement from
  `B=18` to `B=72` has median at least 20% and is positive in at least 10/12
  roots. Every contributing root-by-family cell must first be live and have
  history NRMSE at most `0.10`; an infeasible cell fails closed rather than
  being counted as evidence for a post-forgetting feature effect. A root gain
  is defined only when both teacher families are feasible; at least 10/12 root
  gains must be defined, and undefined roots count as failures in the fixed
  denominator. The median is computed over the eligible root gains.
- **G6 family replication:** the directions in G2 and G5 hold independently in
  both teacher-size families, `N_T=2` and `N_T=3`.

The exact one-sided sign probability of at least 10 positive roots out of 12 is
`(C(12,10)+C(12,11)+C(12,12))/2^12 = 0.0193` under a fair-sign null. Effect
sizes and root-block bootstrap intervals are reported alongside gate outcomes.

### Secondary quantum-specific gate

The QELM-versus-`classical_tanh` contrast is reported because it matches feature
count. It is not sufficient by itself for quantum-specific wording. On the same
fixed fast-fading cells used by G5, the QELM must achieve a median NRMSE gain of
at least 10% over the **best registered classical control in each cell** among
`linear_delay`, `poly2_delay`, and dimension-matched `classical_tanh`, positive
in at least 10/12 roots. Only then may the phrase “QELM-specific improvement in
this tested family” be used. The same live/history-feasible mask as G5 is
mandatory, including its eligibility and fixed-denominator rules. Failure or a
tie does not invalidate the primary
two-bottleneck result and must be reported plainly.

### Secondary dissipation-window scaling gate

For each root and each `N_T` family separately, regress `log(history NRMSE)` on
`gamma_T * dt * d` over the fixed positive-loss grid
`gamma_T={0.2,0.5,1,2,4}` and all four windows. The scaling prediction passes in
a teacher-size family when at least 10/12 roots have a negative fitted slope
and `R^2 >= 0.85`; it is declared replicated only if both `N_T=2` and `N_T=3`
families pass. A missing, suffix-dead, nonfinite, nonpositive, or duplicate cell
on the exact `5 x 4` grid makes that root-family fit fail. This secondary gate
does not alter the primary G0–G6 decision.

## Decision labels

- `CONFIRMS_FINITE_RESOURCE_FRONTIER`: G0–G6 all pass.
- `SUPPORTS_HISTORY_ONLY`: G0–G3 pass but G4 or G5 fails.
- `NO_GO`: any of G0–G3 fails.
- `UNRESOLVED_FAMILY_REPLICATION`: G0–G5 pass but G6 fails.
- Append `_QELM_SPECIFIC` only if the secondary gate passes.

No label establishes quantum advantage, hardware robustness, continuous-input
universality, asymptotic scaling, or approximation of arbitrary QRCs.

## Final freeze amendment — 28 August 2026

No `confirm*` root was simulated or inspected before this amendment. The final
three-root calibration was `results_canonical_calibration_stable_v5`, whose
pre-amendment material identity is
`66678cdc8190e4aaa5eb87b8efc3cd67ce7624fb734e766ef5c379c25de710cf` and whose
gate-report SHA-256 is
`d87c5e9516637c543ffdb46e9cc690191aeccef97d3ad9ffc490971f21f53f9e`.
It contained 36 live teacher systems and 1,440 rows in each scientific table.

The following calibration-informed changes are declared before confirmation:

1. The primary student was corrected from legacy reset-replay to the canonical
   reset/parallel one-lag-per-qubit TD-QELM structure; reset-replay remains the
   same-physics truncation oracle.
2. Suffix panels are nested and numerically unique at every registered window,
   preventing duplicate conditioning values from being assigned different
   group means.
3. G2 uses an independently drawn streaming reset-replay error rather than a
   quantity algebraically containing the suffix history term. G4 requires the
   registered impossible-to-feasible history transition, and G5 plus the
   classical safeguard require history feasibility before measuring feature
   gains.
4. P6 was added as a non-primary prospective scaling prediction after the
   disjoint calibration suggested dependence on `gamma_T * dt * d`.
5. The fixed SVD cutoff `1e-6` was selected solely to satisfy the independent
   identical-suffix numerical-invariance check, not to optimize task error.
   All finite ridge candidates and rank diagnostics now use that same resolved
   subspace. In the final calibration the maximum independently recomputed
   prediction disagreement was `1.21e-13`, below the unchanged `1e-12` gate.
6. G0 now binds JUnit and duplicate-smoke evidence to the evaluated material,
   recomputes normalized errors and regime labels, and verifies all
   student-independent quantities across model rows. Figure contracts were
   made fail-closed for missing, nonpositive, censored, and degenerate cells.

Calibration supported the registered mechanisms without changing their
thresholds: each root had G2 median Spearman `1.0`; G4 root gains were
`{0.742, 0.822, 0.689}`; G5 root gains were `{0.598, 0.890, 0.502}`; and all six
P6 root-family fits had negative slopes with `R^2` from `0.927` to `0.996`.
The optional QELM-versus-best-classical gains were instead
`{-2.577, -0.550, -3.045}`. This negative calibration result is retained: the
primary two-bottleneck claim does not require a QELM-specific or quantum
advantage.

The axes, thresholds, seed namespaces, gate logic, and figure contracts above
are now frozen. The independent JSON freeze record supplies the exact final
configuration and material hashes. No confirmatory result may be used to alter
this amendment.

## Freeze and one-opening governance

The confirmatory command is technically disabled until a separate sibling file
`configs/canonical_emulation_freeze.json` exists. That record must contain:

- schema version `1`, study name, profile `confirmatory`, and status
  `FROZEN_CONFIRMATORY`;
- the exact SHA-256 of `configs/canonical_emulation.yaml`;
- the exact content identity of all registered source, test, configuration,
  and protocol files selected by the provenance manifest;
- `confirmatory_roots_previously_opened: false`; and
- a UTC freeze timestamp.

The record is intentionally outside its own material hash. Both the runner and
the evaluator independently verify it; the run manifest records the verified
freeze-record hash. A dirty Git worktree is permitted only because every
selected file is content-addressed. Any later material edit invalidates the
record and blocks a confirmatory run rather than silently changing the study.

This amendment switches the YAML status from
`CALIBRATION_ONLY_CONFIRMATORY_SEALED` to `FROZEN_CONFIRMATORY` and attests that
no confirmatory root was previously opened. Thereafter the 12-root output may
be generated once. Any rerun is a
reproducibility check of the same frozen material and must be identified as
such, never used to alter thresholds or select a preferred result. The material
binding and refusal to overwrite a nonempty output directory are enforced in
code. The claim that no copy of the confirmatory roots was opened elsewhere,
and the labeling of any same-material rerun, are explicit investigator
governance attestations because a local runner cannot discover every copied
checkout or output location.

## Development and stress tiers

Before opening the 12 confirmatory roots, implementation is checked with:

- `smoke`: one root, `N_T=2`, local loss, gamma `{0,1}`, small row counts;
- `calibration`: three nonconfirmatory roots across the full family grid with
  reduced rows and suffix groups. It may detect dead features or infeasible
  compute, but cannot change confirmatory thresholds after any confirmatory
  output is opened.

After the primary decision, descriptive stress tests may add:

- `N_T=4` with four fresh roots;
- collective loss as a nonmixing/decoherence-free boundary family;
- finite-shot budgets `{2^13,2^15,2^17}` with five noise draws averaged within
  root;
- student dissipation crossed independently of teacher dissipation;
- spatially multiplexed parallel QELM as a sensitivity analysis.

Stress tiers cannot rescue a failed primary gate and are never pooled into the
12-root denominator.

## Paper claim boundary

If the primary gates pass, the strongest permitted claim is:

> Finite-window emulation of recurrent QRC observables has two successive
> finite-resource bottlenecks: teacher forgetting sets the necessary history,
> and the accessible feature/readout resources of a canonical nonrecurrent
> QELM set the remaining approximation error.

Forbidden claims include “every QELM approximates every QRC,” “finite-window
QELM is universally impossible,” “the suffix inequality is a new theorem,” and
“quantum advantage.”
