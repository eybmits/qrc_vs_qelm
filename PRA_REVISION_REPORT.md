# Physical Review A revision report

## Current manuscript

The revised paper is titled **Reset-Window Emulation of Quantum Reservoirs:
Constructive Approximation and Protected-Memory Limits**. It is one
self-contained 12-page PDF with 50 cited
references, three main figures, continuous prose,
Methods, and a technical appendix. There is no separate supplemental PDF.

The paper asks one question throughout:

> When can a finite reset-window QELM reproduce a recurrent QRC observable?

The answer has two parts. The common-suffix diameter is an exact information
criterion. A growing Bernstein QELM then supplies an explicit representation
certificate when that diameter fades. The theorem is one-way observable
approximation, not fixed-architecture equivalence or quantum advantage.

## Exact protected-memory result

The quantum contribution is a concrete two-qubit open-system memory mechanism.
The singlet population is:

- **protected** by every allowed later symmetry-preserving input;
- **excitable** by an addressable control; and
- **visible** through two-body Pauli correlations.

The input alphabet is time homogeneous: every step contains a symmetric drive
and an addressable-control flag. One fixed pair of infinite trajectories is
chosen before the window length. The pair differs once in the flag and shares
all later symmetry-preserving inputs. For a window of length `d`, evaluation
after the next `d` common inputs hides the differing event from the student.
The same pair therefore witnesses every finite window.

For the registered one-radian pulse, the exact output gap is
`0.1149244235`. Any stateless predictor conditioned only on the visible suffix
has worst-case expected absolute error at least `0.0574622`. The exact balanced
two-history MSE floor is `0.0033019058`, reported as `0.003302` when rounded.

The theorem covers the declared observable and all allowed
symmetry-preserving suffixes. It does not cover stateful students, supplied
initial quantum states, prefix summaries, branch labels, external memory, or
arbitrary symmetry-breaking noise.

## Exact observable-memory criterion

For each window length, `D_d` is the largest scalar-output gap between histories
that share their final `d` inputs. The optimal unrestricted suffix predictor has
uniform error exactly

```text
E_d = D_d / 2.
```

Thus uniform finite-window approximation is possible if and only if
`D_d -> 0`. This is a criterion for the declared output, not for the full
quantum state. It organizes three checkable regimes:

1. Exact protection keeps a nonzero common-suffix output diameter at every
   delay and produces a persistent error floor.
2. Imperfect protection yields a finite certified persistence horizon from the
   uniform one-step defect bound. The certificate does not assert that memory
   vanishes beyond that horizon.
3. Uniformly contractive channels satisfy
   `D_d^(O) <= 2 ||O||_infinity eta^d`, giving a certified finite window.

Protected memory and uniform contraction are complementary sufficient
conditions, not logical converses. The exact `D_d/2` identity is the complete
scalar-output interface characterization.

## Constructive QRC-to-reset-QELM theorem

For scalar inputs, assume the input-conditioned channels are uniformly
contractive and input Lipschitz. Combining reset replay with the established
parallel Bernstein quantum feature construction gives

```text
uniform error <= 2 M eta^d
                 + M L_u (1-eta^d) / ((1-eta) sqrt(r)).
```

The QELM uses `d*r` freshly prepared qubits and `(r+1)^d` block-Hamming outcome
probabilities. The paper gives explicit sufficient choices of `d` and `r` for
any positive tolerance. A finite-shot estimator adds
`M sqrt(2 log(2/alpha)/Q)` with confidence `1-alpha`. The three terms are
discarded history, representation of the retained suffix, and measurement.

Bernstein quantum-feature universality and generic non-asymptotic circuit
bounds are established prior art. The
contribution is the end-to-end QRC observable bound, its resource ledger, and
its integration with the exact protected-memory obstruction. More generally,
continuous fading-memory QRC filters on compact scalar intervals are uniformly
approximable by a growing reset-QELM family.

The theorem is a function-class existence result. Its target-dependent
coefficient table has `(r+1)^d` entries and no sample-efficient learning
guarantee. It does not grant universality to the fixed six-qubit local-Pauli
QELM in the phase map. Persistent recurrence may save resets, channel calls,
latency, lag storage, and coefficient storage.

The deterministic construction audit uses the repository's existing
one-qubit local-loss QRC. Quantum statevector and product-binomial feature
probabilities agree to `8.33e-17`. Bernstein error falls from `0.043774` at
`r=1` to `0.0014373` at `r=32`; replay error falls from `0.099479` at `d=1`
to `3.83e-5` at `d=8`. Every evaluated point lies below its certificate.

The external task-level recurrence audit was not merged into the confirmatory
evidence. Its raw reproducibility bundle was not supplied, and its additional
task figures would create a separate empirical story. Only the self-contained
analytic contraction result was integrated.

For mean-squared prediction, conditional-mean orthogonality separates the
student-independent history floor from the student-side error. Finite-shot
noise adds an exact term only for a fixed trained affine readout with
conditionally unbiased, independently sampled test-shot noise. Noisy training
randomizes the fitted readout and is evaluated by rerunning the complete fitting
and selection procedure.

## Numerical diagnosis

The frozen experiment is organized around four questions.

### 1. Does the window contain enough information?

Common-suffix panels measure target variation caused by older prefixes. On the
registered transition cells, increasing the window from `d=1` to `d=6` reduces
held-out QELM NRMSE by a root median `78.45%`, with the same direction in all 12
root blocks. Local-loss forgetting collapses against the loss-window product,
while channel and observable controls show that only output-visible surviving
state contributes to the floor.

### 2. Can the learner represent the available map?

On the history-feasible slice `gamma_T=4, d=6`, increasing the feature budget
from `B=18` to `B=72` reduces representation NRMSE by a root median `67.26%`,
again in all 12 roots. Feature expansion is not interpreted in
history-impossible cells.

### 3. Are the features measured accurately enough?

With independent finite-shot train, validation, and test features, the
validation-selected policy's median test NRMSE falls from `0.231` at 300 shots
per input to `0.0800` at 19,200 shots. Richer features become useful only when
the measurement budget resolves them.

### 4. Is the transition specifically quantum?

No. The quadratic classical delay model is the strongest registered classical
control in all 24 fast-forgetting root--teacher-family cells and outperforms the tested
QELM in every root. The separate quantum-specific gate fails with 0/12 positive
roots. This negative control strengthens the information-interface
interpretation and rules out the simple quantum-advantage story for the tested
models.

All registered checks G0--G6 are defined in Methods and pass. Root blocks,
rather than condition rows, observables, or shot replicates, are the statistical
units.

## Figure system

The main paper contains exactly three figures:

1. the fixed trajectory pair, protected--excitable--visible mechanism, three
   error sources, and the two-sided approximation boundary;
2. a simplified information/feature frontier over loss and window length; and
3. the evidence for forgetting, feature richness, and shot count.

Every main figure is generated from frozen source artifacts by
`pra_extension/code/render_information_interface_figures.py`. The renderer
validates the expected grids and headline protected-memory numbers before
writing PDF, PNG, source table, and SHA-256 ledger artifacts. Its tests verify
completeness, determinism, aggregation semantics, and rejection of a tampered
protected-memory gap.

## Reproducibility and release boundary

Methods records teacher parameters, input distribution, windows, feature
budgets, split sizes, root blocks, ridge selection, shot counts, and bootstrap
procedure in prose. Detailed channel definitions, observables, seeds,
root-level results, hyperparameter grids, and software checks remain in the
repository and are included in the content-addressed release manifest.

Rebuild the paper and release record only after all sources and figures are
final:

```bash
.venv/bin/python pra_extension/code/run_bernstein_qelm_validation.py \
  --output-dir <new-empty-bernstein-directory>
MPLCONFIGDIR=/tmp/qrc_vs_qelm_mpl \
  .venv/bin/python pra_extension/code/render_information_interface_figures.py
(cd paper/canonical_emulation && \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
PYTHONPATH=pra_extension/code:. .venv/bin/python -m pytest -q
.venv/bin/python pra_extension/code/build_release_manifest.py \
  --figures-dir pra_extension/figures_pra_final_v6
```

The release builder checks the frozen canonical identity, pinned analysis
artifacts, embedded source hashes, all manuscript sources including the
same-PDF appendix, the three new source-backed figures, extension code/tests,
and the compiled PDF. Any stale or incomplete dependency causes it to fail.

## Remaining limitations

The confirmatory matrix uses two- and three-qubit local-loss teachers and one
six-qubit QELM architecture. Larger-teacher, alternative-dissipator,
continuous-input, topology, and finite-shot studies are post-confirmatory
numerical extensions. All results are simulations with exact teacher labels;
there is no hardware or gate/SPAM/drift/back-action study. The theorem assumes
continuous scalar inputs, a finite-dimensional QRC, and either fading memory
for existence or the stronger incremental contraction condition for explicit
rates. It establishes neither efficient learning nor full-state emulation.

The supported contribution is an exact scalar information criterion, a
constructive observable-level QRC-to-reset-QELM bound, a protected-memory
obstruction, and a held-out resource diagnosis.
