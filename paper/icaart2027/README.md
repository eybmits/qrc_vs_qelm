# ICAART 2027 anonymous editorial draft

This directory contains the conference-format version of the current paper.
The journal-format draft in `../canonical_emulation/` is preserved. Scientific
results are preserved; figure layouts and repeated prose are revised for readability.
There is one paper, including its proofs and numerical methods, not a separate
supplement.

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
The verified three-figure conference and journal builds each have twelve
pages. In both, Figures 1, 2 and 3 appear on pages 2, 5 and 6.
The resource float is queued early enough to avoid an empty column before
the conference discussion. All pages were visually reviewed after the omission;
`visual_review.json` records the current PDF hash and review scope.

Figure 1 has since been simplified in the shared native TikZ source at
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
Figure 1 contains two architecture lanes and one shared right-hand output plot. Both model outputs feed the same comparison brace. Compact gold readout triangles have no text labels. The same model-specific O_1,...,O_m notation is used in the two observable banks; it does not assert identical operators or budgets.

The single input and output axes are matching 4.40-cm squares centered
between the architecture lanes on the unchanged
17.9197-by-13.0-cm canvas. Fine black lines with dense filled markers,
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

Separate panel c is removed, not relabeled: its former state-visibility meaning differs from the conditional toy. The reachable/observable-memory theory remains in the text. The selected B4 tolerance panel b stays below a, centered by an x translation, with its smooth analytic exponential and unrestricted scalar-optimum meaning unchanged. Captions explicitly identify the curves and crossing as schematic and retain the fixed target readout.

Native axes and architecture labels remain 11 pt, panel letters 12 pt; the compact legend and event labels use 10 pt. Historical full-figure alternatives and the B/C gallery remain unmodified design records. Final page-scale QA is recorded in visual_review.json and bound to the exact delivered PDF.

Current Figure 3 retains the separately frozen five-qubit, longer-context study: twelve
fresh roots, a ten-qubit emulator, both specified losses and all windows
2/4/6/8/10. History error falls, but the fixed emulator does not always
improve; the quadratic control beats the 120-feature QELM in 118 of 120
streaming root-condition comparisons. The new cohort is not pooled with the
earlier evidence. All three selected figures use the same reference-derived
style; their numerical source data and the archival phase-map source CSV
are unchanged. In the preceding four-figure build, Figures 1-4 appeared
separately on pages 2, 4, 6 and 7. The retained figures' declaration
points are distributed through the main text, ordinary top placement respects
the one-full-width-figure-per-page limit, and a barrier clears the remaining
figure before the Discussion and Conclusions. Float distribution preserves
figure widths and scientific evidence. Figure 1 retains
the selected B4 tolerance diagram as a compact panel b; the new prediction
reference belongs to panel a, and the short caption matches both panels. The preceding four-figure journal rendition used
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
