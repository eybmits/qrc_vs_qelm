# Physical Review A revision report

## Current manuscript

The revised paper is titled **When Can Finite-Context QELMs Replace
Quantum Reservoir Memory?** It is one
self-contained 12-page journal-format PDF with 45 cited
references, four main figures, continuous prose,
Methods, and a technical appendix. There is no separate supplemental PDF.

The separate ICAART-format rendition remains one 12-page anonymous paper.

The latest requested editorial revision restores the circuit-style QRC channel
block in Figure 1, uses identical gold observable banks and readout symbols,
and shortens its caption. The target readout remains fixed (identity in the
experiments); only the emulator is trained to match it. No experiment changes.
The requested removal of tool disclosures applies to the manuscript text.
The conference-format artifact is therefore an editorial draft, not a
submission-ready ICAART paper: the official disclosure requirement is still
recorded as unmet in the external checklist and validation report.

### Reference-style figure revision (6 September 2026)

The supplied `2608.25511v1 (5).pdf` was inspected at matched PDF rendering
scale and through its embedded font/vector resources. Its Helvetica-family
labels, bold axis text, thin four-sided axes, compact inset legends and
panel lettering guide the redesign. Our larger text is retained for reading
at normal manuscript scale. Actual reference colors are now shared: blue
#3B86FF, green #00A14B, red #FC4D30, purple #A93BFF and gold #FFC943;
ordered context/feature encodings use its viridis sequence. No reference
data or physical-regime claims are imported.

Figure 2 combines the former six panels into three square curve panels.
All 198 median-history measurements and all five root-level outcome classes
remain: context is encoded by curve color and direct end labels, while
open dots, crosses and filled budget-specific shapes show the unchanged
10-of-12 success rule. The caption distinguishes that rule from median
threshold crossings. Figure 3 retains its exact source values, root medians
and capped 10th--90th root-percentile whiskers, with square axes and
in-panel legends. All faint individual-root lines and bands are removed;
the whiskers describe variability rather than confidence intervals. Both
quantitative CSV exports remain byte-identical. Figure 1 retains native
TikZ and the exact simulated time-series samples. Two aligned processing
lanes now show the QRC node network and reset-QELM encoding/fixed gates,
with gold observable/feature banks and compact theory panels below. The
current-input arrow and blue carried-state loop distinguish fresh input
from accumulated history; the caption states the sequential update explicitly.

### Separate prospective larger-system study (6 September 2026)

A new, separately frozen study extends the numerical scope to five-qubit
targets, 45 declared Pauli outputs, a ten-qubit fixed reset-QELM, windows
2/4/6/8/10, losses 1 and 4, and budgets 30/60/120. Twelve fresh roots produce
24 live systems. The backend was checked against the original dense model,
108 bound tests passed, and two development runs agreed before fresh data
were generated. The original confirmatory gates and cohorts are unchanged.

Window 2 to 10 reduces history NRMSE in all twelve roots at each loss:
median paired reductions 93.55% and 99.63%. At loss 1, windows 6, 8 and 10
meet the robust history rule but no tested QELM budget meets the total-error
rule. At loss 4, the 60-feature emulator reaches the target in every root
from window 4 onward. Quadratic delay beats the fixed 120-feature QELM in
118 of 120 streaming root-condition comparisons. The fixed emulator can
therefore fail even while the history term continues falling; this is not
evidence of quantum advantage or universal failure of other QELMs.

Figure 4 includes every prespecified window and both losses, with capped
root-percentile whiskers for history, the fixed 120-feature QELM, and the
quadratic control. Both manuscripts remain twelve pages without changing
fonts, margins or publisher formatting files. Full methods and proofs stay
in the same PDF. A 120-entry evidence archive seals the numerical sources,
development checks, raw results and independent analysis; its machine-local
freeze paths are documented rather than claimed portable.

The release manifest seals N5 in the named
`prospective_followups.long_context_n5.artifacts` layer. This prevents the
legacy prior-seed scanner from interpreting the current cohort's own results
as earlier seed use. All files and hashes remain present; frozen numerical
sources and tests are untouched. New studies must explicitly include this
completed cohort when constructing their own prior-seed inventory.

### Accessibility revision (5 September 2026)

The main text now follows six sections: Introduction; Quantum Reservoirs and
Reset-Window Models; Limits of Finite-Window Emulation; Dynamical Origins of
Observable Memory; Numerical Experiments; and Discussion and Conclusions.
Only the dynamics and numerical sections have main-text subsections. A short
model section defines the carried state, declared output, reset, and input
window before the theory. The information bound comes before the dynamical
criterion, with physical meaning introduced before specialized notation.
The numerical section introduces architectures, independent roots, and the
evaluation protocol before interpreting the results. Full proofs and methods
remain in two appendices in this PDF.

Comparison with the preceding revision preserves all 45 citation keys, 26
labeled equations, and all three figure blocks and captions. Figure size and
artwork are unchanged; float declarations are positioned to keep the figures
near the relevant text. The exact protected-memory gap and absolute-error
floor are retained beside their pulse derivation in Appendix A. The model
description explicitly distinguishes identical exact-expectation predictions
from identical prediction distributions under finite-shot sampling. No new
experiment, numerical result, or stronger scientific claim was introduced.

### Title-free square plots and overview design study

A subsequent presentation pass removes narrative and size titles from the
plots and makes all nine quantitative axes physically square. Axis units,
scales, series, and uncertainty are unchanged; the two source CSV exports
remain byte-identical. Captions now carry the two/three/four-qubit panel
mapping. The user selected architecture-first design A as the new Figure 1.
The user's subsequent request replaces the square overview with a compact
17.9197-by-9.2-cm circuit-oriented layout, about 49% lower at the same width.
The schematic curve axis remains square; Figures 2 and 3 are unchanged.
Native 9.5-pt labels, aligned black wires, a general QRC channel block,
reset-QELM gate symbols, gold measurement/readout accents and restrained
panel borders follow the supplied reference style. Explanatory paragraphs
move to the caption, which retains the theorem assumptions and schematic
architecture status. The older input waveform is
blue, and the recent window is teal. Two aligned blue memory paths use solid
and dashed strokes, filled and open markers, and explicit output differences
to distinguish visible from permanently invisible state differences. Both
branches remain visible; stored but unreadable memory is not depicted as
absent. The selected source and two retained alternatives are delivered with
LaTeX sources, vector PDFs, PNG previews, and hashes. Captions follow the new
panel order without altering the scientific statements.

The subsequent simplification keeps this footprint but removes the outer
frames and perimeter window route. Identical local teal traces identify the
same window in the two model lanes. One fixed $U(S_d)$ block replaces the
encoding/brickwork sequence, and grouped measurement interfaces replace the
repeated meter icons. Panel c shows two alternative state-difference cases,
not a physical splitting of one difference. Final-size PDF review found no
overlapping labels; the ICAART copy uses the same vector artwork.

The input glyph is now replaced by a reproducible sampled time series and
an exact suffix zoom: the first 120 post-transient Mackey--Glass samples at
seed 1, with the last 20 supplied to the illustrative reset interface. Both
axes are physically square and retain the same unscaled amplitudes. Lines
join the samples directly, without smoothing. This is a simulated input
illustration, not a new emulation experiment or an evaluation stream; the
caption and source provenance distinguish it from the independent 33-symbol
inputs used for the numerical evaluation. Figures 2 and 3 are unchanged.

The readability revision raises native Figure 1 labels to 11 pt and panel
letters to 12 pt, enlarges channel/measurement/readout symbols and sample
markers, and adds clearance below the feature-map block. The data, axis
geometry and compact canvas stay fixed. Declaring the float after the first
Introduction paragraph brings the overview forward for page-two placement.

The history-access revision adds a blue brace over the entire upper input
history, explicitly labeled "one input at a time" to distinguish sequential
QRC updates from reset-window encoding. Earlier input still acts through the
carried state, not a batch input register. The upper workflow is raised by
0.9 cm on a 17.9197-by-10.6-cm canvas while all fonts and data axes retain
their physical sizes. Plain-language output-difference and output-relevance
labels simplify panels b and c; the caption retains maximization over shared
windows and the distinction between some and every future continuation.

The paper asks one question throughout:

> When can finite explicit input context reproduce the input--output behavior
> of a recurrent quantum reservoir?

The answer has three linked parts. The common-suffix diameter gives the exact
scalar information limit. Under the manuscript's compactness, continuity,
free-suffix, fixed-output, and per-window density assumptions, a growing
reset-QELM family can emulate the QRC output arbitrarily well if and only if
that diameter vanishes. With the stated compact reachable-difference
assumption, vanishing is equivalent to stability on the state directions that
are both reachable and visible in the declared output. A Bernstein
construction supplies one explicit resource certificate. None of these
statements implies fixed-architecture universality, efficient learning, or
quantum advantage.

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
after the next `d` common inputs hides the differing event from the emulator.
The same pair therefore witnesses every finite window.

For the registered one-radian pulse, the exact output gap is
`0.1149244235`. Any stateless predictor conditioned only on the visible suffix
has worst-case expected absolute error at least `0.0574622`. The exact balanced
two-history MSE floor is `0.0033019058`, reported as `0.003302` when rounded.

The theorem covers the declared observable and all allowed
symmetry-preserving suffixes. It does not cover stateful emulators, supplied
initial quantum states, prefix summaries, branch labels, external memory, or
arbitrary symmetry-breaking noise.

## Exact observable-memory criterion

For each window length, `D_d` is the supremal scalar-output gap between histories
that share their final `d` inputs. The optimal unrestricted suffix predictor has
uniform error exactly

```text
E_d = D_d / 2.
```

For vector outputs, the exact uniform error is the largest Chebyshev radius of
the compatible output sets, bounded between `D_d/2` and `D_d`. Thus the same
vanishing condition characterizes whether growing windows determine the
declared output arbitrarily well. This is not a claim about reconstructing the
full quantum state.

For a finite-dimensional QRC, let `V` contain reachable state differences and
remove the subspace that remains invisible under every future input. On the
resulting quotient, let `r_vis` be the joint spectral radius of the induced
channel family. The manuscript proves

```text
c a_d <= D_d <= C a_d,
D_d -> 0  iff  r_vis < 1,
lim D_d^(1/d) = r_vis.
```

The constants do not depend on `d`. This yields the asymptotic context law
`d_epsilon / log(1/epsilon) -> 1/(-log r_vis)` when `0 < r_vis < 1`.
Uniform contraction is a simple sufficient special case. The theorem uses a
compact forward-invariant reachable-difference set, a freely concatenable
suffix alphabet, finite dimension, and the declared output norm; constrained
input grammars require the corresponding constrained joint spectral radius.

The physical corollaries organize three regimes:

1. Exact protection keeps a nonzero common-suffix output diameter at every
   delay and produces a persistent error floor.
2. Imperfect protection yields a finite certified persistence horizon from the
   uniform one-step defect bound. The certificate does not assert that memory
   vanishes beyond that horizon.
3. Uniformly contractive channels satisfy
   `D_d^(O) <= 2 ||O||_infinity eta^d`, giving a certified finite window.

Protected memory and uniform contraction are complementary sufficient
conditions. The reachable-observable joint radius supplies the exact general
criterion under the theorem's assumptions.

## Constructive QRC-to-reset-QELM theorem

Assume compact inputs, continuous finite-dimensional QRC channels, a fixed
finite observable output, freely concatenable suffixes, and a reset-QELM
family dense in continuous functions on each finite window. The manuscript
proves the exact asymptotic equivalence

```text
growing reset-QELM emulation to arbitrary uniform accuracy
iff D_d -> 0.
```

Necessity is architecture independent: two histories with one suffix receive
one prediction. Sufficiency replays the final `d` channels from a fixed
reachable state and approximates that continuous replay map with the QELM.
This statement concerns exact expectation values; finite shots are a separate
resource.

For scalar inputs, uniform incremental contraction and input Lipschitzness give
an explicit bound when combined with the established parallel Bernstein
quantum feature construction:

```text
uniform error <= 2 M eta^d
                 + M L_u (1-eta^d) / ((1-eta) sqrt(r)).
```

The QELM uses `d*r` freshly prepared qubits and `(r+1)^d` block-Hamming outcome
probabilities. The paper gives explicit sufficient choices of `d` and `r` for
any positive tolerance. A finite-shot estimator adds
`M sqrt(2 log(2/alpha)/Q)` with confidence `1-alpha`. The three terms are
discarded history, representation of the retained suffix, and measurement.

Bernstein quantum-feature universality, joint spectral radius theory, and
generic non-asymptotic circuit bounds are established prior art. The
contribution is their connection to the exact suffix-information criterion for
QRC-to-reset-QELM emulation, together with the protected-memory realization and
held-out resource diagnosis.

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
emulator-independent history floor from the emulator-side error. Finite-shot
noise adds an exact term only for a fixed trained affine readout with
conditionally unbiased, independently sampled test-shot noise. Noisy training
randomizes the fitted readout and is evaluated by rerunning the complete fitting
and selection procedure.

## Numerical diagnosis

The numerical evidence combines the registered study with separately labeled
post-confirmatory extensions and is organized around four questions.

### 1. Does the window contain enough information?

Common-suffix panels measure target variation caused by older prefixes. On the
registered transition cells, increasing the window from `d=1` to `d=6` reduces
held-out QELM NRMSE by a root median `78.45%`, with the same direction in all 12
root blocks. Local-loss forgetting collapses against the loss-window product,
while channel and observable controls show that only output-visible surviving
state contributes to the floor.

After the two- and three-qubit dense map was inspected, a separate four-qubit
follow-up was frozen on the identical loss--window grid with 12 new roots.  It
passes every prospective check: 12/12 roots jointly have negative decay slope
and `R^2 >= 0.85`; the minimum-window frontier is non-increasing with strict
steps and six finite thresholds; and the inherited `gamma_T=4,d=6,B=72`
anchor succeeds in 12/12 roots.  The cohorts are displayed side by side and
are not pooled.

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
control in all 24 fast-forgetting root--target-family cells and outperforms the tested
QELM in every root. The separate quantum-specific gate fails with 0/12 positive
roots. This negative control strengthens the information-interface
interpretation and rules out the simple quantum-advantage story for the tested
models.

All registered checks G0--G6 pass. The manuscript summarizes their decisive
outcomes; the frozen protocol and gate report retain their full definitions.
Root blocks, rather than condition rows, observables, or shot replicates, are
the statistical units.

## Figure system

The main paper contains exactly three figures:

1. a native-size TikZ diagram contrasting recurrent state with explicit context,
   isolating input-reachable and output-visible memory, and showing the
   boundary between uniformly fading and persistent visible memory;
2. two complementary rows: all 198 measured history-error points above,
   and discrete feature-success maps for the transition region below; and
3. three equal-sized resource panels aligned side by side, with shared legends
   below the axes and unchanged data, colors, and publication typography.

The layout draws on the supplied *Finite-shot access laws for quantum
reservoirs* reference: generous vertical space, readable panel headings,
white backgrounds, and a clear visual hierarchy. Figure labels are 9--11 pt
at their final printed size, compared with the manuscript's 10 pt body text.
Numerical error values are displayed as percentages of normalized RMSE, not
redefined as relative prediction errors. The figures retain the original
root aggregations, tolerance, classifications, and cohort boundaries.

Every main figure is generated from stored, source-backed artifacts by
`pra_extension/code/render_information_interface_figures.py`; confirmatory and
post-confirmatory evidence layers remain labeled separately. The renderer
validates the expected grids and headline protected-memory numbers before
writing PDF, PNG, source table, and SHA-256 ledger artifacts. Its tests verify
completeness, determinism, aggregation semantics, and rejection of a tampered
protected-memory gap.

## Reproducibility and release boundary

Methods records target-QRC parameters, input distribution, windows, feature
budgets, split sizes, root blocks, ridge selection, shot counts, and bootstrap
procedure in prose. Detailed channel definitions, observables, seeds,
root-level results, hyperparameter grids, and software checks remain in the
repository and are included in the content-addressed release manifest.

Rebuild the paper and release record only after all sources and figures are
final:

```bash
.venv/bin/python scripts/analyze_phase_threshold_sensitivity.py
.venv/bin/python pra_extension/code/run_bernstein_qelm_validation.py \
  --output-dir <new-empty-bernstein-directory>
MPLCONFIGDIR=/tmp/qrc_vs_qelm_mpl \
  .venv/bin/python pra_extension/code/render_information_interface_figures.py
(cd paper/canonical_emulation && \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
.venv/bin/python pra_extension/code/build_release_manifest.py \
  --figures-dir pra_extension/figures_pra_final_v6
PYTHONPATH=pra_extension/code:. .venv/bin/python -m pytest -q
```

The release builder checks the frozen canonical identity, pinned analysis
artifacts, embedded source hashes, all manuscript sources including the
same-PDF appendix, the three new source-backed figures, extension code/tests,
and the compiled PDF. Any stale or incomplete dependency causes it to fail.

## Remaining limitations

The confirmatory matrix uses two- and three-qubit local-loss target QRCs and one
six-qubit QELM architecture. The separate four-qubit phase-map follow-up,
alternative-dissipator,
continuous-input, topology, and finite-shot studies are post-confirmatory
numerical extensions. All results are simulations with exact target-QRC labels;
there is no hardware or gate/SPAM/drift/back-action study. The emulation
theorem assumes compact inputs, input-continuous channels, a finite-dimensional
QRC, a fixed finite observable output, freely concatenable suffixes, and
reset-QELM density on each finite window. The spectral characterization
additionally assumes a compact forward-invariant reachable-difference set.
Scalar inputs and the stronger incremental contraction condition are used only
for the explicit Bernstein resource rate. The paper establishes neither
efficient learning nor full-state emulation.

The supported contribution is an exact window-information criterion, an
if-and-only-if growing-QELM emulation theorem, a reachable-observable memory
rate, a constructive resource bound, a protected-memory realization, and a
held-out resource diagnosis.
