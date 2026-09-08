# ICAART 2027 anonymous editorial draft

This directory contains the conference-format version of the current paper.
Only this ICAART version is now maintained. The journal-format draft in
`../canonical_emulation/` is retained unchanged as an inactive version.
Original results are preserved. Figure 2 adds a separately frozen feature-density
follow-up and the existing independent N4 history cohort. Panel c is unchanged.
There is one paper, including its proofs and numerical methods, not a separate
supplement.

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

## Current color system: Cobalt

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

**Submission status: blocked.** The current PDF omits preparation-tool mentions,
citations and disclosure at the author's editing request. Its presentation can
pass format checks, but it does not satisfy the venue's currently published
disclosure requirements. This is an editorial draft, not a submission-ready
paper. The required policy decision and known preparation history are retained
in `SUBMISSION_CHECKLIST.md`, outside the paper and anonymous source ZIP.

The current main-paper selection contains three figures: the architecture/theory
overview (Figure 1), resource diagnosis (Figure 2, formerly Figure 3), and the
separate five-qubit longer-context study (Figure 3, formerly Figure 4).
Artifact filenames stay `fig01_information_interface.pdf`,
`fig03_resource_diagnosis.pdf`, and `fig04_long_context_n5.pdf`.
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

The two stacked right-hand frames now show Fading memory and Persistent
memory, with one shared charcoal-filled QRC / coral-open QELM key and no
case labels inside the artwork. Each case uses a known bit-independent
baseline and a balanced old bit, at fixed d=40 and evolving time t.
The upper example and lower charcoal target are unchanged. In the lower
case, shared future controls gradually reveal the stored bit with coefficient
0.25 s^2(3-2s), s=clip((t-32)/64,0,1), independently of the window length.
At t=40 both ideal window predictors lose access to the bit. The lower gap
starts at 0.0107421875 and grows to a persistent 0.25 from t=96. Its baseline
and ideal prediction are updated together, never blending missing-bit values
into predictions after expiry. The fading gap tends to zero. These are
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
  **Do not upload the present disclosure-free draft to PRIMORIS.**
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

Deleting disclosure does not change the known preparation history. The live
venue guidance requires disclosure of generated content and citations to the
system in affected text; conceptual illustrations also have tool/prompt-credit
requirements. Human authors must resolve that compliance blocker before any
submission. A private editing preference is not an organizer exemption.

Official rules were checked on 6 September 2026. Recheck them before submission:
[guidelines](https://icaart.scitevents.org/Guidelines.aspx?y=2027),
[template](https://icaart.scitevents.org/Templates.aspx?y=2027),
[AI policy](https://icaart.scitevents.org/AiTools.aspx?y=2027),
[deadlines](https://icaart.scitevents.org/ImportantDates.aspx?y=2027).
