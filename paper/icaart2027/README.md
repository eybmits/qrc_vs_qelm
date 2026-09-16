# ICAART 2027 anonymous editorial draft

This directory contains the conference-format version of the current paper.
Only this ICAART version is now maintained. The journal-format draft in
`../canonical_emulation/` is retained unchanged as an inactive version.
Original results are preserved. Figure 2 uses the existing separate history,
feature-density and finite-shot studies without pooling their cohorts.
There is one paper, including its proofs and numerical methods, not a separate
supplement.

## Current revised appendix (16 September 2026)

Appendices A--E use the author's `qrc_qelm_appendix_revised.pdf` (three pages;
SHA-256 `7784cff8db65cc0d76c2c1c6b731a2ccd816b93804ae08aec1e2f3e83821196a`).
The replacement is typeset in the existing manuscript template, preserving
semantic section links and renumbering the retained displayed equations.
The main resource paragraph uses the revision's midpoint-predictor notation
`f_r`; both editions derive from the same appendix source. Figures, data,
title, abstract, introduction and conclusion are unchanged. Earlier appendix
descriptions and equation-number inventories below record superseded versions.
Both rebuilt PDFs have twelve pages, including references and all five revised
appendices, without further cuts or changes to fonts, margins or figure sizes.
The four retained numbered appendix equations are now (12)--(15).

## Current figure refinement (16 September 2026)

Descriptive chart and panel titles are removed from Figures 2 and 3; panel
letters, axis labels, legends and data-condition annotations remain. Figure 1's
model and memory-case annotations remain part of its explanatory diagram.
The paired QRC output gap in Figure 3(b) focuses on shared inputs 1--32; all
48 source points and percentile bands are retained. The NARMA10 and laser
traces share solid 0.9-point lines, filled QRC dots and 2.4-point open QELM
circles every third sample. Black target traces remain fully opaque.
Figure 3 retains its full width and nine equal 1.51-by-0.78-inch plot frames
in a shorter 6.221-by-3.97-inch canvas. These display choices are preserved in
the revised-appendix editions. Source observations, numerical results, captions
and model colors are unchanged.

## Current introduction and conclusions (14 September 2026)

The author's replacement Introduction and Discussion and Conclusions are
typeset in full, with the supplied literature references mapped to existing
BibTeX entries. Figure 1 and its caption are retained after the opening
paragraph. Both the anonymous and authored editions remain twelve pages;
no appendix or Time-Series Demonstrations text was removed.

## Current abstract (14 September 2026)

The 164-word abstract foregrounds observable memory and the irreducible error
floor for every fixed-window QELM emulating the constructed reservoir.
At the author's request, it omits the scalar half-diameter formula, the
"under stated assumptions" phrase, the construction's qubit count and the
quantum/classical comparison. Growing windows, sufficient expressivity, exact
expectations and worst-case scope remain explicit; the full assumptions and
theorems remain in the body. The paper remains 12 pages, with body text,
equations, figures, font sizes and margins unchanged by this abstract update.

## Previous: full-width Figure 3 with consistent plot heights (14 September 2026)

Figure 3 uses the full text width with a 6.221 by 4.17 inch native canvas.
All nine plot frames are 1.51 inches wide and 0.78 inches high. The bottom
three plots match the upper plots, reducing their height from 1.00 inch in
the preceding full-width draft. Row edges and axis labels align. Compact white
legend boxes now sit entirely inside each bottom plot. Removing their former
external space shortens the canvas by 0.30 inch without shrinking any plot.
The ten-input label sits below the dip. History labels abbreviate last d inputs,
full history and last 4 inputs; the y-axis reaches 100% with 5% extra headroom
to keep the legend clear of the uncertainty band (limits 40--105%).
Model names, the Target legend and laser test-error labels stay outside the
curves. The memory legend is horizontal and the window annotation uses one line.
NARMA Target traces remain black at full opacity (alpha 1).
The source data, colors, other limits and excerpts are unchanged. A shorter caption
retains all scientific qualifications; equivalent formula layouts and paragraph
joins recover space without changing the publisher's fonts or margins.

The verified PDF has 12 pages, including references and all appendices.

## Previous: focused NARMA history transition (14 September 2026)

Figure 3(c)'s history plot displays windows 1--20 and test NRMSE from 40% to
90%, with a guide at ten inputs and the legend below. All 48 computed windows
remain in the source. The text reports the verified nine-to-ten-input change
(77.6% to 53.1% median NRMSE; all 24 roots improve), while distinguishing access
to the delayed input from causal isolation or complete recursive history.
The upper prediction traces still display 48 test steps at the existing scale.

## Previous: shorter NARMA prediction display (14 September 2026)

Both NARMA10 prediction traces in Figure 3(c) show the first 48 samples of the
existing 96-sample excerpt, with test-step ticks at 0, 24 and 47. The source
retains all 96 samples, and the original vertical limits and 50% ground-truth
opacity remain. This display crop preserves the selected root, predictions,
full-test statistics and every-window history curve.

## Previous: lighter NARMA ground-truth traces (14 September 2026)

Figure 3(c) draws the black ground-truth line at 50% opacity (alpha 0.5) in both
NARMA10 prediction traces, making the model predictions easier to distinguish.
This opacity-only change preserves predictions, source values, numerical
results and all other panels.

## Previous: distinct memory-control colors (14 September 2026)

Figure 3(b) distinguishes fast loss in orange from slow loss in purple, keeping
the existing circle/diamond markers and solid/dashed lines. Percentile bands
use the matching curve colors. Both earlier input prefixes and their label
are red; the identical shared continuation and its label remain black.
Only colors change; traces, numerical values, geometry and all other panels
are preserved.

## Previous: every-integer history curve (14 September 2026)

Figure 3(c) computes restricted-QRC test error for every integer window from 1
through 48 on the same 24 fresh NARMA roots. Each point refits its readout using
the original training split and validation-only ridge selection, keeping the
previously selected QRC dynamics fixed. This is a descriptive grid extension;
the original replication gates and controls remain unchanged. The separate
dense-replay ledger retains rootwise scores, percentile summaries and agreement
checks at the original six windows. No points are inferred by interpolation.

The history axis is linear, with a small marker at every computed window. The
full-history QRC reference is red and dashed; the restricted QRC stays orange
and the four-input QELM stays blue. Other figure panels and their data remain.

The independent audit checks all 1,152 replay scores, 24 full-QRC scores,
144 archived replay anchors, 168 complete readout fits, and direct evolution
at three fixed test rows for every root/window. The source ledger embeds the
report and its checkpoint hashes. Ill-conditioned original ridge fits produce
small cross-runtime differences: the maximum archived NRMSE difference is
0.000232 percentage points. The separate extension protocol records the
post-diagnostic absolute 1e-5 metric tolerance; strict same-runtime checks and
all selected alphas, dynamics and feature counts are preserved.

Reproduction uses `pra_extension/code/narma_dense_replay.py`, then
`verify_narma_dense_replay.py`, then the runner's `--publish-only` option before
`render_memory_prediction.py`. Use one BLAS thread. Raw checkpoints are local
to `results_narma_dense_replay_v2`. On a clean checkout, restore the
`specification.numerical_reproducibility` object from the portable source ledger
as `results_narma_dense_replay_v1/numerical_preflight.json` before starting a
fresh results directory; this preserves the documented numerical diagnosis.
The portable figure loader does not require the raw checkpoint directories.

## Previous: accessible-history labels and explicit feature choices (14 September 2026)

Figure 3(c) describes the information available to each model: QRC with the last
d inputs, QRC with full history, and QELM with the last four inputs. The title
asks how much history is needed; the x-axis counts recent inputs, including the
current input. Resetting, replay and separate readout fits remain documented in
the caption/methods. The plotted arrays and empirical reference are unchanged.

Figure 2 has a 7.055 by 3.05 inch canvas with a larger staircase, stronger cell
boundary without in-cell numbers. Its display starts at loss rate 1 and omits
the lower-rate annotation; only integer rates are labeled. The four omitted
rows below 1 fail the sampled criterion at every tested window, as the caption
states. Exports retain all 198 source cells across eleven tested loss rates
and windows one through six.
No compatible longer-window data exist; the separate N5 cohort is not mixed in.
The lower measurement panel replaces stacked bars with all twelve directly
labeled choice percentages. These are rounded shares of 192 validation choices
per shot tier, made before testing; bold/pale blue marks the most frequent count.
This revision changes presentation only. All source values, cohorts, intervals,
colors, Figure 1, scientific claims and the selected title are preserved.

## Previous: paired emulation panels and consistent model colors (14 September 2026)

Figure 3 now places laser-driven emulation (a) next to the measured-input memory
control and separate analytical construction (b). The synthetic NARMA10 task is
in (c). The lower prediction plot shows the same selected QRC reset for every
prediction, processing only the last d inputs and using a readout refitted for
that window. Its six unsmoothed estimates are compared with a dashed continuous
QRC reference and the single available four-input QELM estimate. The reference
is empirical, not an error floor; the measured dip at d=12 remains visible.
Quadratic curves are omitted from this display at the author's request, while
all quadratic results remain in the source ledger and manuscript text.

All active figures use orange #D97632 for QRC and blue #3868D9 for QELM;
external targets and raw input traces use charcoal #20252B. QRC outputs that
serve as emulation targets remain orange, with no independent third trajectory.
Blue shades distinguish QELM feature counts and errors against different target
sizes. The red analytical lower bound and yellow observable accents retain
their distinct roles. An active palette layer preserves the frozen legacy
palette and historical figure evidence. Samples, intervals, cohorts, fitting,
title, framing and proofs are unchanged. Figure captions and cross-references
follow the new panel order. Current PDF and clean-source checks are in the
manifest.

```sh
MPLCONFIGDIR=/private/tmp/qrc-mpl-cache .venv/bin/python pra_extension/code/render_active_information_interface.py
MPLCONFIGDIR=/private/tmp/qrc-mpl-cache .venv/bin/python paper/figure2_candidates/render_resource_atlas.py --publication --output paper/figures/resource_atlas
MPLCONFIGDIR=/private/tmp/qrc-mpl-cache .venv/bin/python pra_extension/code/render_memory_prediction.py
.venv/bin/python paper/icaart2027/prepare_assets.py
.venv/bin/python paper/icaart2027/build_submission.py
```

## Previous: intuitive measurement-budget panel (14 September 2026)

Figure 2(c) replaces the numerical matrix with test-error curves over the four
shot tiers. All 16 medians retain their full 10th–90th root-percentile whiskers;
distinct markers identify fixed feature counts and the validation-selected
policy. Small within-tier offsets separate nearly coincident points. The 10%
reference is not a success classification. Enlarged validation-choice bars show
the existing frequencies and label the majority shares. Panels a/b, all source
data, model comparisons and the 13 September framing remain unchanged.
The caption identifies the whiskers and offsets. See the current manifest for
the rebuilt PDF and source archive; the previous matrix remains in the original
candidate artifacts and repository history.

## Previous: finite-window emulation framing (13 September 2026)

The author-selected title is now *When Can Finite-Window QELMs Emulate Quantum
Reservoirs?* The abstract, introduction and conclusion foreground the distinction
between missing history and model limitations, retaining both observable
forgetting and protected memory. Section 4.3 explicitly states the delayed-recall
task already implied by the protected construction: exact QRC expectations
realize the target, while every stateless fixed-window predictor has a worst-case
error floor over unbounded delays. This is a consequence of the existing
construction, not a new experiment or an independent claim of quantum advantage.

All existing display equations, proofs, citations, figures, captions, numerical
results and comparison conditions are retained. Repeated framing prose was
condensed to keep twelve pages, with a 162-word abstract. The source ZIP rebuilds
to an identical PDF. Current validation and visual-review identities are recorded
in `ICAART2027_MANIFEST.json`. Existing submission-policy and author gates remain
separate from this editorial revision. The following entries describe earlier
versions.

## Previous: three-panel Figure 3 with stacked NARMA addition (11 September 2026)

Figure 3 now has three columns in the same 6.221 by 3.65 inch canvas:
laser-driven QRC emulation (a), vertically stacked NARMA10 target/prediction
traces and history restrictions (b), and the measured-input memory control
above the exact red-floor construction (c). The active asset is
`figures/fig04_memory_prediction.pdf`. No old curves are dropped: the
128-row laser excerpt, 96-row NARMA excerpt, all history budgets, both
quadratic controls, both loss rates and all encoding-copy points are retained.
Only tick density, line placement, labels and layout change.

The caption keeps laser emulation, synthetic task prediction, paired QRC gaps
and analytical worst-case error distinct. NARMA's 24 fresh roots are not pooled
with the original 12-root laser cohort. Percentile spread and replay costs
remain explicit. The earlier two-column Figure 3 and standalone NARMA asset
are preserved as alternatives. Figures 1/2, fonts, margins and proofs are
unchanged. Current counts and visual/clean-source checks are in the manifest.

```sh
MPLCONFIGDIR=/private/tmp/qrc-mpl-cache PYTHONPATH=pra_extension/code:src .venv/bin/python pra_extension/code/render_memory_prediction.py
.venv/bin/python paper/icaart2027/prepare_assets.py
.venv/bin/python paper/icaart2027/build_submission.py
```

## Previous: earlier Figure 3 restored, NARMA retained as an addition (11 September 2026)

At the author's request, the active Figure 3 is again the unchanged
`figures/fig04_emulation_evidence.pdf`: laser-driven QRC-output emulation (a),
the paired-prefix memory test (b), and analytical convergence with the bright-red
"Unavoidable window error" reference (c). The laser trace is a measured input,
not the forecasting target. The red limit belongs only to the separate
constructed benchmark, not the laser data. The original 12-root emulation
cohort is not pooled with the later forecasting studies.

Section 5.1 retains the 24-fresh-root NARMA10 prediction check as a short
additional result, including the same-QRC history restriction and the
twelve-input quadratic control. Appendix E.5 documents both demonstrations.
NARMA10 is a synthetic task, not a measured real-world record. An optional
standalone figure is available at
`../figures/narma_additional/narma_additional.pdf`; it is not an extra page
in the paper. The direct-laser forecasting results and three-column task
composite are retained as archived alternatives, with all data and provenance.
No simulations, seed selections, or curve values changed in this restoration.

Repeated methods wording was condensed without changing the publisher fonts,
margins, or proofs. The paper has twelve pages and a 183-word abstract.
Current counts and the clean-source and visual checks are
recorded in `ICAART2027_MANIFEST.json`.

```sh
MPLCONFIGDIR=/private/tmp/qrc-mpl-cache PYTHONPATH=pra_extension/code:src .venv/bin/python pra_extension/code/render_narma_additional.py
.venv/bin/python paper/icaart2027/prepare_assets.py
.venv/bin/python paper/icaart2027/build_submission.py
```

This is an editorial revision, not policy approval or a portal submission.
All entries below describe earlier revisions.

## Previous: direct task prediction and fresh-root memory controls (11 September 2026)

Figure 3 now contains actual next-value prediction on measured laser data (a),
driven synthetic NARMA10 targets (b), and reset/refit history restrictions (c).
Neither quantum model learns the other's output in this figure. All 24 fresh
roots per study are retained; the earlier 12-root studies remain separate.
The first fresh root and first 96 scored test rows were fixed before running.
Traces are not smoothed or averaged across roots. Slow-loss laser results and
the strong longer-window classical NARMA control remain visible.

Independent verification recomputed all 1,896 score rows and checked 7,920
saved-readout predictions against reconstructed channels. All four prospective
direction checks pass in 24/24 roots. This supports useful history beyond four
inputs in the selected tasks, not a uniquely quantum requirement or failure of
every finite window. Laser variation is conditional on one measured record.
Full results, limitations and reproduction commands are in
`../../pra_extension/TASK_MEMORY_REPLICATION_README.md`.

The active asset is `figures/fig04_task_memory.pdf`. The former emulation,
paired-prefix, and sharpness artwork and all source rows are preserved.
Exact sharpness results remain in Section 5.5 and Appendix C; their standalone
curve is no longer in the main figure. Abstract, introduction, model scope,
evaluation, conclusion and numerical methods now distinguish emulation from
the new direct-task comparison. Repeated prose was condensed to retain twelve
pages and a 184-word abstract. Figure 1/2 artwork, proof equations, publisher
fonts and margins are unchanged by this update.

```sh
MPLCONFIGDIR=/private/tmp/qrc-mpl-cache PYTHONPATH=pra_extension/code:src .venv/bin/python pra_extension/code/render_task_memory.py --source results_task_memory_replication_v1 --output paper/figures/task_memory
.venv/bin/python paper/icaart2027/prepare_assets.py
.venv/bin/python paper/icaart2027/build_submission.py
```

This is a scientific/editorial update, not policy approval or a portal submission.
All entries below describe earlier revisions.

## Previous: resource atlas and measured-input memory control (11 September 2026)

Figure 2 now shows the actual window/loss-rate history map, aligned feature
curves, and an error matrix with validation-choice frequencies. Percentile
ranges remain visible. It is built from all 198 history cells, 288 feature rows,
48 shot-root rows and 768 validation choices; later plateau budgets remain in
the source export. Its publication source is `paper/figures/resource_atlas`.

Figure 3 retains the same laser excerpt and analytic reference. Its new panel b
shows the existing paired-control output gaps, with medians and 10th–90th
percentiles across all 12 seeds for each rate. The first 48 shared inputs match
the original control figure. A framed mini-plot, lightly overlapping the
upper edge without hiding data, displays the actual two assembled input
histories. This is a controlled splice of measured inputs, not two naturally
repeated records, a global diameter estimate, or proof of permanent memory.
The N5 results remain in the text, original figures and complete source ledger.

At the author's request, Figure 3(c)'s fixed floor is now bright red and
labeled "Unavoidable window error". Its value and all curve data are unchanged;
the dashed style also distinguishes it in grayscale.

The Figure 3(b) input-history strip now reads "Different earlier inputs" and
"Same inputs from here on". Its frame is slightly wider to fit the wording;
the fast/slow-loss legend, data and remaining figure layout are unchanged.

Only Figure 2/3 presentation, their supporting descriptions, and duplicate
prose changed. The abstract, introduction, conclusion, proof equations,
publisher fonts/margins, numerical runner and frozen protocols are preserved.
The PDF remains twelve pages. Earlier entries below describe prior revisions.

```sh
MPLCONFIGDIR=/private/tmp/qrc-mpl-cache .venv/bin/python paper/figure2_candidates/render_resource_atlas.py --publication --output paper/figures/resource_atlas
MPLCONFIGDIR=/private/tmp/qrc-mpl-cache PYTHONPATH=pra_extension/code:src .venv/bin/python pra_extension/code/render_figure3_candidate.py --publication --output paper/figures/emulation_evidence
.venv/bin/python paper/icaart2027/prepare_assets.py
.venv/bin/python paper/icaart2027/build_submission.py
```

## Integrated measured-input Figure 3 (10 September 2026)

The author-selected combined plot is now part of the twelve-page ICAART PDF.
Figure 3(a) shows the fixed Santa Fe laser excerpt with trained four-input QELM
emulation of fast- and slow-loss QRC outputs; (b) combines both N5 loss rates
without dropping any of the 360 window-study observations; (c) retains every
analytic encoding-copy setting. The archived N5 feature-budget panel is no
longer displayed, but its data, original artwork and main-text result remain.
Figures 1 and 2 are unchanged. The new asset is
`figures/fig04_emulation_evidence.pdf`; prior figure-selection descriptions
below record earlier revisions.

Section 5.1 adds the separate demonstration and its 12-seed median errors.
Appendix E.5 specifies chronological splits, training-only scaling, 33-level
inputs, matched four-qubit dynamics and 48 features, ridge selection and the
same-window control. The observable target is not a forecast of future laser
intensity; the fitted errors do not establish global worst-case bounds or
permanent memory. The data source is cited to the Santa Fe competition volume.

Repeated evaluation/methods prose was condensed to keep twelve pages.
Abstract, introduction, theory, conclusion and Appendix A--D are unchanged;
equations (12)--(26), publisher fonts and margins are preserved. The original
numerical runner and its pre-simulation protocol hashes are unchanged.
Reproduction and figure provenance are in
`../../pra_extension/LASER_DEMONSTRATION_README.md`.

```sh
PYTHONPATH=pra_extension/code:. .venv/bin/python pra_extension/code/render_figure3_candidate.py --publication --output paper/figures/emulation_evidence
.venv/bin/python paper/icaart2027/prepare_assets.py
.venv/bin/python paper/icaart2027/build_submission.py
```

## Finite-resource bridge (10 September 2026)

Section 4.4 now brings Appendix C's midpoint resource guarantee into the main
theoretical narrative, immediately before the experiments. Its unnumbered
summary names missing history, representation, and measurement contributions,
with explicit stronger contraction/Lipschitz assumptions, fixed-history
probability, copy/shot rates, construction costs, and the need for reachable
output extrema. It is a sufficient construction, not an efficient learning
algorithm or an exact decomposition of fitted-model error.

Repeated resource prose in Section 3 and Appendix C, the repeated Figure 1
description in Section 2, and the duplicated threshold-sensitivity paragraph
in Appendix E are condensed; the latter remains in Section 5.2. Equations
(21)--(22) retain both bounds and their numbering, with sampling probabilities
stated directly underneath instead of nested probability brackets. All proofs,
data, figures, captions, and publisher formatting are preserved. The paper
remains twelve pages with a 187-word abstract.

## Figure-caption readability (10 September 2026)

The Abstract now uses the author's selected explanation: accurate emulation
does not require all stored information to disappear, only the influence of
inputs outside the window on the chosen outputs. All three active captions
lead with what the reader should see and understand. Detailed settings remain
in Sections 2 and 5 and the appendix. The captions retain the distinctions
between schematic illustrations, sampled errors, and exact constructed
readouts. No plot artwork, data, comparison conditions, fonts, or margins
changed. The rebuilt paper remains twelve pages, with a 187-word abstract.

## Exact sharpness integration (10 September 2026)

The active ICAART draft adds Section 5.5 and incorporates its Cobalt convergence
panel into Figure 3(d). Figure 3 now groups the three original long-context
panels and the analytical construction in a compact 2-by-2 layout; there is
no separate Figure 4. Figures 1 and 2 are unchanged, and no numerical series
are dropped. Original standalone figure artifacts remain archived.
The new post hoc checks demonstrate exact quadratic midpoint attainment and
convergence of a specified nonpolynomial Bernstein readout, not trained-model
performance or an optimal error at each finite capacity. The target is a
classical leaky integrator embedded in a one-qubit channel. Large-copy points
use factorized exact expectations, not full-register statevectors.

Appendix C gives the analytic identities behind the benchmark; Appendix E.1
uses a short biased-history example to distinguish sampled variance from
worst-case error. No extra variance figure or redundant protected-memory rerun
is included. The author's new Abstract, Introduction, and Conclusion framing
is integrated. Only repeated explanations are shortened to retain twelve
pages; the publisher template, existing experimental data, proofs, and
equation numbers (12)--(26) are preserved. The inactive journal is untouched.

The auditable implementation, checks, resource caveats, and supplied-archive
provenance are documented in `../../pra_extension/SHARPNESS_BENCHMARKS.md`.
Regenerate the new plot from the repository root with:

```bash
PYTHONPATH=pra_extension/code:. .venv/bin/python pra_extension/code/render_long_context_sharpness.py
.venv/bin/python paper/icaart2027/prepare_assets.py
.venv/bin/python paper/icaart2027/build_submission.py
```

The notes below retain earlier revision history; where they describe a standalone
construction figure, four evaluation subsections, or an earlier conclusion, this update
supersedes those descriptions. Submission policy and human approval remain
separate from these scientific and formatting changes.

## Current appendix revision

The author's complete A--E readability rewrite is integrated into the active
ICAART sources. The supplied standalone PDF contained no embedded LaTeX source;
its replacement text was reconstructed in the existing template, with no preview
heading or running header. All proofs, resource bounds, protected-memory controls,
numerical settings and experimental extensions are retained. Repeated explanations
are condensed and short italic paragraph leads replace several stand-alone
subheadings, keeping the entire paper at twelve pages without changing fonts,
margins, figures or numerical data.

Equations (12)--(26) retain their order and numbers. The protected implementation
is now D.2, and the common-suffix diagnostic is E.1. Cohorts and fitting are grouped
with the models in E.2; finite-shot and long-context extensions remain separately
identified in E.3 and E.4. Main-text links use automatic labels, and the newer
intermediate-feature experiment is explicitly distinguished from the finite-shot
study. The scalar infimum, fixed-history shot guarantees and existence-versus-
efficient-training distinction remain unchanged. Only the appendix and its
protected-example cross-reference change; the inactive journal is untouched.

## Previous color system: Cobalt

The author selected palette B: cobalt #3868D9, coral #D66A56 and violet
#7C5A9F, with charcoal #20252B and gray #8B9198. All three current figures
and the hyperlink color use this system. Pale architecture/access fills
are tints of the same roots, except the author's restored original yellow
observable/readout accent (#FFC943, with #FFDD92 background blocks).
Target/reference curves remain charcoal;
QELM predictions and explicit-window access are coral. Existing markers,
samples, uncertainty intervals, scales and figure sizes are unchanged.
Only the caption/model color words are adjusted. Earlier color descriptions
and paired-version build notes below are historical, not current instructions.

Section 2 uses the revised accessible explanation, three unnumbered equation
displays and normal-weight prose. Bold emphasis is removed from body text,
theorem labels and caption prose; section headings, panel letters and mathematical
vector notation are preserved. Section 4 now separates observable
forgetting, required window length and protected memory, retaining equations
(4)--(11), symbol definitions and automatic appendix references.
Section 5 follows the author's full
four-subsection replacement. Section 6 uses the author's latest four-paragraph
conclusion, explaining the scalar optimum as half the largest compatible output
difference and retaining an automatic link to the Section 3 assumptions.
This conclusion-only update changes no other section, figure or reference.
Main-text notation, results and equation numbers are preserved.
Both drafts now fit within twelve pages, including references and appendices.
Repeated appendix explanations and methods prose are condensed, and the unused
imperfect-protection aside is removed. Vector bounds and balanced-pair risk are
stated directly without extra named quantities or standalone displays.
The main proofs, assumptions, experimental settings and controls remain.
Fonts, margins and figure sizes are unchanged. The current anonymous source ZIP
rebuilds to a byte-identical PDF; automated checks enforce the twelve-page limit
for both delivered versions.

The subsequent citation-only refresh adds five directly relevant primary sources,
bringing both bibliographies from 18 to 23 cited works: [Chen et al. (2020)](https://doi.org/10.1103/PhysRevApplied.14.024065)
for quantum-state contraction, [Sannia et al. (2024)](https://doi.org/10.22331/q-2024-03-20-1291)
for dissipative forgetting, [Čindrak et al. (2024)](https://doi.org/10.1103/PhysRevResearch.6.013051)
for restricted recent-input replay, [Zanardi and Rasetti (1997)](https://doi.org/10.1103/PhysRevLett.79.3306)
for collective-symmetry singlet protection, and [Hoeffding (1963)](https://doi.org/10.1080/01621459.1963.10500830)
for bounded-readout concentration. Chen's first name is corrected to Jiayin.
All section wording and formulas are unchanged in this refresh, verified by
comparison with the pre-refresh sources after removing citation commands and
whitespace. The citations support these existing premises and methods, not
an attribution of the paper's window-emulation theorem. Both PDFs stayed within the twelve-page limit.

**Submission status: blocked.** The author-approved short disclosure now names
OpenAI Codex and its assistance with ideation, wording, code and figures,
including debugging, and states author review and responsibility.
Affected-section citations and conceptual-figure tool/prompt credit remain absent. Its presentation can
pass format checks, but it does not satisfy the venue's currently published
disclosure requirements. This is an editorial draft, not a submission-ready
paper. The required policy decision and known preparation history are retained
in `SUBMISSION_CHECKLIST.md`, outside the paper and anonymous source ZIP.

The current main-paper selection contains three figures: the architecture/theory
overview (Figure 1), resource diagnosis (Figure 2), and the combined
longer-context and exact-construction figure (Figure 3).
Artifact filenames stay `fig01_information_interface.pdf`,
`fig03_resource_diagnosis.pdf`, and `fig04_long_context_sharpness.pdf`.
The former Figure 2 phase map is omitted as a standalone main-paper float,
not deleted: its `fig02_information_frontier` PDF, PNG, and source CSV remain
in `../figures/information_interface/`; its renderer and upstream results
remain unchanged in their existing repository locations.
Its three size panels share the same tested primary minimum-window frontier
and visually repeat the decay story; the retained figures communicate the
resource separation and longer-context comparison more compactly. This
editorial rationale belongs in these notes, not in the manuscript narrative.
The phase-map results, root-level success rule, independent four-qubit cohort,
and qualifications remain in the paper text. No cohorts or plots are merged.
After the notation and section revisions, redundant appendix material was
compressed to meet twelve pages in both builds. Figures 1, 2 and 3 remain on
pages 2, 5 and 7 in both builds.
The resource float is queued early enough to avoid an empty column before
the conference discussion. All pages were visually reviewed after this revision;
`visual_review.json` records the current PDF hash and review scope.

The notation audit in `../NOTATION_AUDIT.md` records literature conventions,
every symbol family, retired aliases and the retained proof-only distinctions.
Both versions use target y, prediction widehat y and window predictor f;
plain Q means shots. General inputs use a, physical evolution time is tau,
and the target/QELM observable banks use O_j/P_j with counts m/B.
Definitions are local to the paper; no long notation table was added to it.
Concise rewording preserves all theorem assumptions, proof steps, numerical
values, seed chronology and error-interpretation qualifications.

The scalar theorem now gives the exact infimum D_d/2 over the growing,
uniformly dense exact-expectation QELM family at every fixed d, without
asserting attainment by one finite model. Arbitrary-accuracy emulation iff
D_d tends to zero is retained as a consequence; the vector consequence uses
replay. The observable-memory criterion explicitly credits Petreczky et al.
(2026, Theorem III.1) and the realization principles of Grigoryeva and Ortega
(2021). A separate midpoint-coefficient Bernstein construction on the full
scalar alphabet [0,1] retains the representation and per-prediction shot
bounds, with optimal history term D_d/2. It needs reachable-output extrema;
the existing replay implementation and its numerical validation are unchanged.

## Current Figure 1: fading versus persistent memory

The two stacked right-hand frames have quiet external labels: "Fading memory /
Window sufficient" above and "Persistent memory / Window insufficient" below.
One shared charcoal-filled QRC / coral-open QELM key remains between them.
Each case uses a known bit-independent baseline and a balanced old bit,
at fixed d=40 and evolving time t. Both charcoal targets and the entire lower
example are unchanged. The upper bit coefficient is now
0.25(1-clip(t/40,0,1))^2: it vanishes exactly at a fixed physical toy horizon
of 40 steps, independent of window length. Its shared baseline is the retained
target minus this coefficient; the ideal prediction therefore joins exactly
at d and remains coincident. This revises an illustrative memory mechanism,
not measured performance. The retained exponential shape is part of the known
baseline only. Exact finite-time forgetting of this bit is a special case,
not a general claim about fading memory or a proof of global D_d=0. In the lower
case, shared future controls gradually reveal the stored bit with coefficient
0.25 s^2(3-2s), s=clip((t-32)/64,0,1), independently of the window length.
At t=40 both ideal window predictors lose access to the bit. The lower gap
starts at 0.0107421875 and grows to a persistent 0.25 from t=96. Its baseline
and ideal prediction are updated together, never blending missing-bit values
into predictions after expiry. The upper gap is exactly zero. These are
analytic illustrations, not fitted or measured QRC/QELM performance.

The old lower panel b and a/b letters are removed. The 17.9197-by-9.60-cm
canvas retains the input and architecture at their existing coordinates and
font sizes. Each output frame is 4.15 by 3.20 cm, with identical x/y scales,
matched grids and common time ticks. The shared legend slightly overlaps
their facing borders. The redundant
input-step and time axis titles are omitted. The left schematic uses dt=0.5
and 2000 burn-in samples for a less oscillatory raw Mackey--Glass trace,
without smoothing or normalization. Its causal 40-input window and
experimental Figures 2/3 are unchanged.
The caption and the Figure 1 paragraph in Section 2 distinguish elapsed-time
decay at a fixed window from uniform accuracy as d grows. Section 3 and its
D_d/2 theorem are unchanged, and no trajectory gap bears that general label.
The active ICAART version contains 12 pages; the journal is not rebuilt.

### Earlier Figure 1 revisions (historical, superseded above)

Figure 1 was previously simplified in the shared native TikZ source at
`../figures/information_interface/fig01_information_interface.tex`: aligned
model lanes, grouped measurement interfaces, and one centered sampled
Mackey--Glass input plot. It contains 201 illustrative samples with current
time t=180; a bracket sends exactly the green indices 141--180 (40 values)
to the reset encoder. There is no duplicate lower window plot. The unchanged
input amplitude domain is [0.3,1.3].
Grey samples 181--200 are visual continuation only and enter neither model.
The first 120 values retain their original checksum. A purple dot and dashed
time guide identify the interior current input; a horizontal leader connects
its actual ordinate to the QRC channel through a rounded elbow. The window
bracket connects only the selected causal interval to reset encoding. The anonymous PDF copy preserves
the vector content. Figure 1 now uses 11-pt native labels, 12-pt panel letters and larger
channel, measurement and sample symbols. Its float is declared early in the
Introduction for page-two placement. All retained and archival figure data are
unchanged. The earlier reference-style revision merged the now-archival phase
map's separate history and feature grids into three square curve panels:
color identifies window length and marker shape identifies the unchanged
root-level success category. The resource figure, now Figure 2, has
larger square axes, inset legends, and less unused canvas. Faint individual
root lines and bands are replaced by capped 10th--90th percentile whiskers,
which describe root variability, not confidence intervals. All figures share
Helvetica-family text and actual blue/green/red/purple/gold reference colors;
ordered context/feature budgets use the reference viridis sequence. Bold axis
labels, fine four-sided data axes, and compact white legend boxes follow the
supplied reference while retaining larger minimum text for readability.
Figure 1 contains two architecture lanes and one shared right-hand output plot. Target y_t and prediction widehat y_t feed the same comparison brace. Compact gold readout triangles have no text labels. Target observables O_1,...,O_m and QELM features P_1,...,P_B have distinct notation, matching the text; no identical-operator or equal-budget claim is implied.

The single input and output axes are matching 4.15-cm squares centered
between the architecture lanes on the unchanged
17.9197-by-13.60-cm canvas, with 0.60 cm of extra separation above panel b. Fine black lines with dense filled markers,
green hollow prediction markers, sans-serif ticks and explicit endpoint
padding follow Figure 1 of the supplied `2608.25511v1 (7).pdf`.
The output plot uses a separate conditional toy: histories differ only in an
initial bit b=+1 or -1, with retained output g(t)+0.25b. A 40-input window
reads the bit before t=40; afterward its balanced conditional mean is g(t),
also the scalar midpoint of the two outputs. Only b=+1 is drawn. An independent
101-sample Mackey-Glass shape supplies the normalized baseline, not simulated
QRC dynamics or predictions fitted from the displayed input. Thus the plot
illustrates lost access, not measured QELM performance or a universal failure
instant. The caption discloses this scope without an in-art schematic tag.
No training/test shading or uncertainty bands are invented. Current provenance
records all values, formulas, scopes and hashes.

Separate panel c is removed, not relabeled: its former state-visibility meaning differs from the conditional toy. The reachable/observable-memory theory remains in the text. Panel b stays below a, left-aligned with the input frame, with its smooth analytic exponential unchanged. The Window QELM heading and y-axis identify the history limit D_d/2; epsilon is the desired tolerance. The scalar interval inset is removed at the author's request; the half-gap argument remains in the text. Caption and model text retain the schematic and exact-expectation-family scope, not fixed-QELM performance.

Native axes and architecture labels remain 11 pt, panel letters 12 pt; the compact legend and event labels use 10 pt. Historical full-figure alternatives and the B/C gallery remain unmodified design records. Final page-scale QA is recorded in visual_review.json and bound to the exact delivered PDF.

### Retained experimental Figure 3

Figure 3 retains the separately frozen five-qubit, longer-context study: twelve
fresh roots, a ten-qubit emulator, both specified losses and all windows
2/4/6/8/10. The 1×3 layout matches Figure 2's square panels and gutters.
Panels a/b retain the original logarithmic window comparison; panel c shows
total common-suffix QELM error at fixed d=10 for all tested budgets 30/60/120,
on a linear percentage scale. Green filled squares and open circles distinguish
the two losses, with root medians and 10th–90th-percentile whiskers. Its separate
72-row source CSV retains all twelve roots at every budget and records provenance;
the original 360-row source and frozen numerical results are unchanged.
History error falls, but the fixed emulator does not always
improve; the quadratic control beats the 120-feature QELM in 118 of 120
streaming root-condition comparisons. The new cohort is not pooled with the
earlier evidence. All three selected figures use the same reference-derived
style; their numerical source data and the archival phase-map source CSV
are unchanged. In the preceding four-figure build, Figures 1-4 appeared
separately on pages 2, 4, 6 and 7. The retained figures' declaration
points are distributed through the main text, ordinary top placement respects
the one-full-width-figure-per-page limit, and a barrier clears the remaining
figure before the Discussion and Conclusions. Float distribution preserves
figure widths and scientific evidence. The earlier Figure 1 used
the B4 tolerance diagram as a compact panel b; this is now removed in favor
of the paired memory timelines described above. The preceding four-figure journal rendition used
pages 2, 4, 5 and 6, and both versions had twelve pages. Full methods and
proofs remain in a single PDF. Build reports record the actual placement
and page count after the three-figure selection; no new positions or length
are asserted before those builds are checked.

## Deliverables

- `submission.pdf`: anonymous editorial draft (legacy filename retained).
  **Do not upload the present incompletely credited draft to PRIMORIS.**
- `ICAART2027_anonymous_source.zip`: self-contained anonymous draft LaTeX sources.
  No author handoff, repository metadata, earlier drafts, or local paths enter
  this archive. The review guidelines request a PDF; the ZIP is a reproducible
  source handoff, not an instruction to upload additional material.
- `submission_qa.json`: actual page/character counts, format/anonymity checks,
  separate required-disclosure check and explicit submission blockers.
- `ICAART2027_MANIFEST.json`: editorial-draft status, content hashes,
  policy status and clean-extraction build result.
- `template_manifest.json`: provenance and hashes of the unmodified publisher
  formatting files, downloaded from the official conference template link.
- `SUBMISSION_CHECKLIST.md`: official requirements, portal details and unresolved
  human decisions. It is not part of the anonymous source ZIP.

## Build

From the repository root, with its Python environment and a TeX Live installation:

```sh
.venv/bin/python paper/icaart2027/prepare_assets.py
.venv/bin/python paper/icaart2027/build_submission.py
.venv/bin/pytest -q paper/icaart2027/test_validate_submission.py
```

The build uses `latexmk`, pdfLaTeX, BibTeX, pypdf and Poppler (`pdftotext`,
`pdffonts`). It enforces formatting, writes a deterministic source ZIP, and
recompiles that ZIP in a fresh temporary directory. The two PDFs must match
byte-for-byte. A successful editorial build does not bypass policy checks:
`format_passed` can be true while `policy_passed`, overall QA `passed`, and
`submission_ready` remain false. The manifest records
`EDITORIAL_DRAFT_NOT_SUBMISSION_READY`. Rendered-page visual review remains a
separate required check. Running `validate_submission.py` directly returns a
nonzero exit status for the unresolved policy blocker; `--format-only` explicitly
selects the formatting exit gate without hiding policy results from its report.

To compile only the unpacked source ZIP:

```sh
latexmk -pdf -jobname=submission -interaction=nonstopmode -halt-on-error main.tex
```

The official `article.cls`, `SCITEPRESS.sty`, `apalike.sty`, and `apalike.bst`
are unmodified. SCITEPRESS redefines `\sup` for author superscripts; `main.tex`
preserves its author version as `\authorsup` and restores the mathematical
supremum operator. DOI/URL notes in the bibliography prevent the original
apalike style from silently dropping electronic identifiers. Citation keys
contain no embedded whitespace because the supplied apalike style is old.

## Feature-density update

Figure 2 now includes the existing independent four-qubit history cohort in
panel a and a separate, locally frozen fresh-root feature sweep in panel b.
The new sweep tests 18, 36, 54, 72, 108 and 144 features at fixed window,
six-qubit emulator and evolution duration: 216 QELM settings plus 36 quadratic
controls across 12 roots per target size. Panel c is unchanged. Protocol,
freeze, raw results and analysis are identified in
`pra_extension/FEATURE_DENSITY_RESULTS.md`. The caption and Section 5 distinguish
the cohorts; historical update notes above describe earlier revisions.
Both rendered manuscript versions remain twelve pages.

The subsequent spacing-only update moves Figure 2a's legend clear of the
curves, reserves legend headroom in c, and focuses b on 18/36/54/72 features.
The caption discloses the two higher budgets retained in the full results.
The complete 912-row source export is byte-identical; only page 5 changes in
pixel comparisons of both twelve-page PDFs.

The subsequent author-requested intermediate-budget experiment adds actual
B=24/30 fits on the same twelve feature-follow-up roots per target size.
Balanced partial midpoint banks resolve the 18-to-36 transition; selection
does not use outcomes. All 72 endpoint controls reproduce the parent results,
and all 36 reconstructed scientific-array sets match their parent hashes.
The current figure export has 984 rows, including 72 new intermediate fits,
and panel b shows 18/24/30/36/54/72 with non-overlapping tick labels. Higher
budgets remain in the source. Captions and methods explicitly identify this
paired exploratory extension, not independent confirmation. See
`pra_extension/FEATURE_BRIDGE_PROTOCOL.md` and `FEATURE_BRIDGE_RESULTS.md`.
No figures, fonts or proofs were reduced to retain the twelve-page limit.

The following shot-panel annotation edit moves "10% target" slightly left and
below its line, with a small lower-axis margin to avoid the curves and whiskers.
Its font size is unchanged. No new feature curves or experiments are added;
the 984-row numerical source remains byte-identical. Only page 5 changes in
each twelve-page PDF, with the exact current review recorded in `visual_review.json`.

Panel a then receives minimal extra upper-axis headroom and a slightly higher
legend, leaving a checked gap above its 10% line. The legend stays inside the
frame; typography, b/c and the numerical source remain unchanged. Both PDFs
retain twelve pages, with changes again confined to page 5.

## Review versus publication

The active build is an anonymous editorial draft. No author identity is inferred into the review
PDF. Camera-ready author details and acknowledgements require confirmation
after acceptance; no copyright, conflict-of-interest or funding declaration
has been invented. Nothing has been uploaded, submitted, registered or pushed
publicly by this formatting task.

The short disclosure does not by itself fulfill every credit requirement. The live
venue guidance requires disclosure of generated content and citations to the
system in affected text; conceptual illustrations also have tool/prompt-credit
requirements. Human authors must resolve that compliance blocker before any
submission. A private editing preference is not an organizer exemption.

Official rules were checked on 6 September 2026. Recheck them before submission:
[guidelines](https://icaart.scitevents.org/Guidelines.aspx?y=2027),
[template](https://icaart.scitevents.org/Templates.aspx?y=2027),
[AI policy](https://icaart.scitevents.org/AiTools.aspx?y=2027),
[deadlines](https://icaart.scitevents.org/ImportantDates.aspx?y=2027).
