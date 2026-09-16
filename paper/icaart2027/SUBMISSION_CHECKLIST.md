# ICAART 2027 submission handoff

Updated on 16 September 2026. This is an anonymous **editorial draft** in the
Regular Paper format, not a submission-ready package or a final publication.

## Clearer benchmark explanation (16 September 2026)

Section 5.5 explains why the theoretical floor can be reached: the leaky-memory
update leaves an interval of compatible outputs, and its midpoint minimizes
worst-case error. The exact quadratic example and the increasing-capacity
example retain all reported values and their construction-only scope. Both
versions remain twelve pages; the revised appendix and figures are unchanged.

## Revised appendix (16 September 2026)

The author's `qrc_qelm_appendix_revised.pdf` replaces Appendices A--E in both
editions. Shared LaTeX sources preserve the revised proofs, construction,
controls and all five numerical-method subsections. The main resource bound
uses the same midpoint-predictor notation; section references and equations
(12)--(15) resolve automatically. Both PDFs remain twelve pages, with no cuts
beyond the supplied revision and no changes to figures, fonts or margins.
The rendered appendix pages were checked in both editions; the anonymous
code note and authored repository link remain edition-specific. Historical
appendix details and equation inventories below describe earlier versions.

## Author editions (14 September 2026)

The author confirmed the paper author order as **Markus Baumann, Jonas Stein,
Claudia Linnhoff-Popien**. The separate identified preprint and its verified
LMU/QAR-Lab links are in `../arxiv/`; the review PDF and anonymous source ZIP
remain unchanged. Both PDFs have twelve pages. Author metadata and source
citations are recorded in `../arxiv/authors.json`. All three ORCIDs are verified;
Markus's ID was recovered from his existing manuscript author blocks and CFF.
This author-list instruction does not declare exclusive submission or perform
an upload. The author-approved AI disclosure is preserved in both editions.

## Current abstract update (14 September 2026)

The author-supplied Introduction and Discussion and Conclusions are now
included in full, with the original Figure 1 and caption preserved. Both
editions remain twelve pages without appendix cuts. The arXiv title-author
gap is enlarged at unchanged publisher font sizes; this spacing change is
confined to the authored edition.

The revised abstract contains 164 words (limit: 200). It emphasizes persistent
observable memory and the fixed-window QELM error floor. The scalar formula,
explicit assumption phrase, qubit count and quantum/classical contrast are
omitted from the abstract; the mathematical scope remains in the body.
The verified PDF remains 12 pages, with body text, equations, figures and
template settings intact.

## Current layout update (16 September 2026)

Figure 3(b) displays shared inputs 1--32, preserving all 48 source points.
Laser and NARMA10 predictions use consistent solid 0.9-point lines and
filled QRC dots / open QELM circles. Figures 2 and 3 have no descriptive
plot titles; panel letters, legends, axes and explanatory annotations remain.
The nine Figure 3 frames keep their dimensions and alignment in a shorter
6.221-by-3.97-inch canvas. These display choices remain unchanged in the
revised-appendix editions; captions and scientific results are unchanged.

## Previous layout update (14 September 2026)

Figure 3 spans the full text width on a 6.221 by 4.17 inch native canvas.
All nine plot frames measure 1.51 by 0.78 inches; the bottom row now matches
the upper rows. Compact white legend boxes sit entirely inside the lower plots,
clear of curves and uncertainty bands. Removing the external legend space saves
0.30 inch of figure height. History y-limits are 40--105% with ticks through 100%;
the ten-input guide label stays below the dip. Axis labels remain aligned.
Targets stay opaque black, QRC orange and QELM blue. The caption retains
all scientific qualifications in fewer words. Equivalent formula layouts and
same-topic paragraph joins save space; publisher fonts and margins are unchanged.
See the current build manifest and QA report for the verified page count.

The verified PDF has 12 pages, including references and all appendices.

## Previous scientific update

Figure 3(c)'s history plot focuses on windows 1--20 and 40--90% test NRMSE,
with a ten-input guide and the legend below; all 48 windows remain in the source.
The text reports the verified 77.6% to 53.1% median error decrease from nine to
ten inputs, with improvement in all 24 roots and a qualified delayed-input
interpretation. The upper prediction traces retain their 48-test-step display.

Both NARMA10 prediction traces in Figure 3(c) display the first 48 of the
existing 96 excerpt samples. All 96 remain in the source; vertical limits,
ground-truth opacity, full-test statistics and the history curve are preserved.
This changes the displayed time range only, without choosing a new excerpt.

The black ground-truth lines in both NARMA10 prediction traces of Figure 3(c)
use 50% opacity (alpha 0.5). This opacity-only update makes the model predictions
easier to distinguish while preserving all numerical results and other panels.

Figure 3(b) uses orange for fast loss and purple for slow loss, including their
variability bands. Earlier input prefixes and their label are red; the shared
continuation is black. This changes color only, preserving all source data.

Figure 3(c) extends the selected-QRC history grid to every integer d=1..48,
using the same 24 roots, splits, selected dynamics and per-window readout fits.
Each plotted point is computed; the full-history reference is red and dashed.
This descriptive grid extension preserves the original six-window evidence
and replication comparisons in their archived ledger.

Figure 2's staircase starts at loss rate 1, without in-cell numbers or a lower-rate
annotation. Only integer rates are labeled. Omitted rows below 1 fail the sampled
criterion; all source data remain in exports. Its feature-choice panel
now shows all twelve rounded percentages in labeled rows and columns, replacing
the stacked bars; the denominator is 192 validation choices per shot tier.
Figure 3(c) labels available history directly (last d inputs, full history,
last four inputs), while reset/replay and fitting details remain in the caption
and methods. Data, intervals, protocols and the twelve-page target are unchanged.

## Previous model-color and panel-order update (14 September 2026)

Figure 3 is reordered as laser emulation (a), paired memory plus analytical
limit (b), and synthetic NARMA10 prediction (c). The replay chart explicitly
shows resetting the same QRC and processing only the latest d inputs, with a
refitted readout. Quadratic curves are omitted from the display; their numerical
results and all source rows remain. The actual nonmonotone replay curve is
retained, and its continuous-QRC reference is not an error floor.

The active figures now use orange QRC, blue QELM, and black external targets
and input traces; QELM capacity/budget variants use blue shades. The analytical
floor remains red. Captions, model color descriptions and panel references
match the new order. Data, protocols, proofs and the selected title are unchanged.

## Previous graphic update (14 September 2026)

Figure 2(c) now displays test-error curves and percentile whiskers in place of
the numerical matrix, with larger validation-choice bars below. All source
values, cohorts, medians and intervals are unchanged. The caption explains the
within-tier point offsets and the 10% reference. The title, framing, other panels
and scientific results remain as in the preceding revision. Current PDF, source
archive and visual checks are identified in `ICAART2027_MANIFEST.json`.

## Previous editorial update (13 September 2026)

The title is now *When Can Finite-Window QELMs Emulate Quantum Reservoirs?*
The framing explicitly connects the protected-memory construction to a recall
task over unbounded delays. This follows from the existing bound and retains
the exact-expectation and stateless-window scope. The positive emulation result,
all proofs, numerical evidence, figures and captions remain in place. NARMA10
remains a separate practical check, not a test of impossibility for every window.
The rebuilt draft has twelve pages and a 162-word abstract; the source ZIP
reproduces the PDF. See the manifest for current artifact and QA identities.
This revision does not resolve the disclosure and author gates below.

## Previous scientific update (11 September 2026)

Figure 3 now groups the earlier laser emulation (a), the NARMA addition stacked
vertically with its history control (b), and paired-input memory plus the
red-floor analytical benchmark (c). This is a layout-only consolidation in
the same figure footprint. All traces, seeds, budgets and source values remain
unchanged; the older two-column Figure 3 and standalone NARMA plot are retained.
The caption and Appendix E.5 distinguish the cohorts, metrics and replay cost.
The entries below describe the preceding editorial selection.

The author-selected earlier Figure 3 is restored unchanged: laser-driven
QRC-output emulation, the paired-prefix memory test, and analytical convergence
with its red "Unavoidable window error" reference. It is not laser forecasting,
and the analytical limit is not a laser error bound. The original emulation
cohort remains separate from the newer forecasting studies.

The 24-fresh-root NARMA10 result remains a short additional prediction check
in Section 5.1, with methods in Appendix E.5. All 24 roots favor the QRC over
both four-input restrictions. The stronger twelve-input quadratic control is
reported too. This supports useful older inputs on a synthetic task, not a
uniquely quantum requirement or failure of every finite window. An optional
standalone plot is outside the manuscript at
`../figures/narma_additional/narma_additional.pdf`.

The direct-laser forecasting results, task-memory composite, all source rows
and verification records remain archived; no experiment was rerun or retuned.
The PDF has twelve pages and a 183-word abstract. Publisher fonts, margins
and proof equations are unchanged. Reproduction and
full scope: `../../pra_extension/TASK_MEMORY_REPLICATION_README.md`.
Live artifact identities and visual/clean-build checks are in
`ICAART2027_MANIFEST.json`. This does not resolve the policy/author gates below.

## Previous scientific updates

The approved Figure 2 resource atlas and Figure 3 paired-history control are
integrated. The latter retains all 12 seeds per loss rate and shows the first
48 shared inputs, alongside the unchanged natural excerpt and analytic
reference. Its framed history inset uses the actual assembled laser inputs.
The old N5 plot is archived with all source rows; its results remain in the
main text. Figure 2 keeps uncertainty in bands/ranges and adds the existing
validation-choice frequencies. No experiments were rerun or retuned.
Redundant descriptions were shortened to retain twelve pages without changing
the 177-word abstract, proof equations, fonts, or margins. The following
paragraphs record earlier figure revisions; live counts and artifact identities
are in `ICAART2027_MANIFEST.json`.

The approved combined Figure 3 is now integrated: measured-input laser
demonstration (a), both N5 window sweeps (b), and analytic convergence (c).
Section 5.1 reports the separate 12-seed results and Appendix E.5 documents
the protocol. The older N5 feature-budget panel remains archived and its
result remains in the text. No simulations or frozen comparison conditions
changed. Repeated prose was condensed to retain twelve pages with the same
177-word abstract, fonts, margins, and proof equations. Figures 1 and 2 and
Appendix A--D are unchanged. The following paragraphs record earlier revisions.

Section 5.5, Figure 3(d), and the linked Appendix C/E.1 passages add the reviewed
exact sharpness checks. Analytically specified readouts and factorized
expectations are distinguished from fitting, measurements, and hardware.
Figures 1 and 2 and all frozen comparisons remain unchanged. Figure 3 combines
all original long-context panels with the construction curve in a 2-by-2 layout;
there is no separate construction figure. No numerical series are dropped.
The author's replacement framing is integrated in the Abstract, Introduction,
and Conclusion, with repeated prose condensed for the twelve-page budget.
Actual counts, clean-source rebuild, and visual-review hash are recorded in
`ICAART2027_MANIFEST.json`; this update is not a portal submission.

The subsequent readability pass applies the author's selected Abstract wording
and concise, explanation-first captions to all three figures. Plot artwork and
data are unchanged; methodological detail remains in the main text and appendix.
The rebuilt paper has twelve pages and a 187-word abstract.

Section 4.4 now summarizes the constructed finite-resource error bound before
the experiments, retaining assumptions, pointwise probability, and resource
costs. Duplicate explanations are shortened; Appendix C keeps the derivation
and equations (12)--(26) keep their numbering. The updated build still has
twelve pages, without changing figures, fonts, or margins.

## Remaining disclosure requirements

The author approved a short AI-use disclosure, now placed after the references
and before the appendix, as requested.
It names OpenAI Codex, describes ideation, wording, code and figure assistance,
including debugging, and states author review and responsibility.
Affected-section tool citations and conceptual-figure tool/prompt credit remain
absent. Known preparation
history is unchanged: OpenAI Codex assisted research development, mathematical
exposition, drafting/revision, simulation and analysis code, and figure
preparation. It is not an author; human authors remain responsible.

The live [ICAART submission guidelines](https://icaart.scitevents.org/Guidelines.aspx?y=2027)
require disclosure and affected-section system citations. The
[dedicated policy](https://icaart.scitevents.org/AiTools.aspx?y=2027) additionally
specifies figure-credit requirements. Rechecked on 8 September 2026.
**Do not submit the current draft under those rules.** Resolve the disclosure
requirement with the authors and, if placement is uncertain, the organizers.
No organizer exception has been obtained. These notes remain outside the PDF
and source ZIP; they are not a substitute for required in-paper disclosure.

Automated QA now records separate `format_passed` and `policy_passed` values.
The complete-disclosure gate remains false, `submission_ready` remains false,
and the build manifest status is `EDITORIAL_DRAFT_NOT_SUBMISSION_READY`.

## Portal fields

- Conference: ICAART 2027, 19th International Conference on Agents and Artificial Intelligence.
- Submission type: Regular Paper, first stage.
- Area: Artificial Intelligence.
- Primary topic: Quantum Artificial Intelligence; secondary: Machine Learning.
- Title: When Can Finite-Window QELMs Emulate Quantum Reservoirs?
- Abstract: use the plain text from `sections/abstract.tex`.
- Keywords: Quantum Reservoir Computing; Quantum Extreme Learning Machines;
  Temporal Learning; Fading Memory; Finite Context.
- Eventual review attachment: the approved, policy-compliant PDF, not the current
  incompletely credited `submission.pdf`, full repository or this handoff.
- First-stage submission deadline: **29 September 2026, Anywhere on Earth (extended)**,
  rechecked on the official page on 14 September 2026.
- First-stage notification: 13 November 2026; camera-ready and registration: 27 November 2026.
- Second-stage regular submission: 22 October 2026; notification 4 December;
  camera-ready and registration 18 December 2026.

Dates: [official Important Dates](https://icaart.scitevents.org/ImportantDates.aspx?y=2027).

## Formatting and anonymity completed

- Official SCITEPRESS A4, two-column template; unchanged publisher files.
- Template margins and 10-point body type; 9-point abstract/captions/references.
- No printed page numbers, running heads, dated author block or identity in PDF metadata.
- Abstract within the 70–200 word range; title-cased comma-separated keywords.
- Author–year citations; alphabetical bibliography; existing DOI identifiers retained as links.
- Plain figure captions, vector figures within the text area; no added plot titles.
- Unnumbered Appendix follows references and contains proofs and numerical methods.
- Identifying source-repository link omitted; copied figure metadata anonymized.
- Automated report records actual counts and checks. Initial regular submissions
  require 10,000–50,000 characters excluding whitespace, including references,
  figures, tables and appendices. Twelve pages is the accepted-full-paper budget,
  not an initial hard page rule. The draft fits both limits without paid extras.

Sources: [guidelines](https://icaart.scitevents.org/Guidelines.aspx?y=2027),
[official template](https://icaart.scitevents.org/Templates.aspx?y=2027),
[citation guidance](https://icaart.scitevents.org/ReferenceCitationGuidelines.aspx).
Counts are extracted from the PDF, including vector-figure labels. They are a
local preflight, not a claimed PRIMORIS character-count result.

## Human confirmations still required before submission

- [ ] Supply an inspectable, venue-compatible anonymous route to the scientific
  artifacts. The repository's `REPRODUCIBILITY.md` lists the evidence and the
  evidence files; the anonymous LaTeX ZIP reproduces the PDF,
  not the experiments. A withheld identifying URL does not resolve reviewer access.
- [ ] Resolve the required-disclosure blocker above; a passing formatting check
  is not venue-policy clearance.
- [ ] Approve every proof, result, figure, reference, and the factual preparation description.
  This formatting pass is not an independent scientific correctness/novelty review.
- [x] Author list/order supplied: Markus Baumann, Jonas Stein, Claudia Linnhoff-Popien.
  Verified LMU affiliations, contact details and available ORCIDs are in
  `../arxiv/authors.json`, including Markus's verified ORCID 0009-0007-3575-1006.
  Enter those author details in the portal even though the review PDF omits them.
- [ ] Confirm originality and no simultaneous submission to a journal or another
  proceedings conference. Do not infer permission from earlier publication plans.
- [ ] Declare existing preprints/public manuscript versions, including any public
  repository copy, to resolve eligibility if needed. The guidelines prohibit
  posting the submitted paper publicly during review; their wording does not
  clearly settle papers already public before submission. Do not delete or
  change repository visibility without an explicit decision.
- [ ] Resolve the disclosure-placement ambiguity if needed: general guidelines
  remove acknowledgements for double-blind review, but require AI disclosure
  there; the dedicated AI policy also permits an appropriate section. This
  draft uses a non-identifying inline disclosure after the references. Organizer confirmation
  on placement does not itself waive the underlying disclosure requirement.
- [ ] Upload the final PDF in PRIMORIS and retain the submission confirmation.

No portal action, external message, registration, payment, consent or submission
has been performed.

## Repository verification

The previously reported CI defects were repaired in commit `574de3f`.
GitHub run [34252128574](https://github.com/eybmits/qrc_vs_qelm/actions/runs/34252128574)
passed for Python 3.11 and 3.12 when checked on 10 September 2026.
That result applies to the committed baseline, not the subsequent local
sharpness integration. Its local checks are a separate verification; no push
or new remote CI result is claimed for this update.

## Current author data

Paper authors, confirmed in order: **Markus Baumann**, **Jonas Stein**,
**Claudia Linnhoff-Popien**. Shared affiliation: QAR-Lab, Department of Computer
Science, Ludwig-Maximilians-Universität München (LMU Munich), Munich, Germany.
Correspondence: **markus.baumann@campus.lmu.de**, consistent with Markus's
existing paper author blocks. Jonas: **jonas.stein@ifi.lmu.de**, ORCID
**0000-0001-5727-9151**. Claudia: **linnhoff@ifi.lmu.de**, ORCID
**0000-0001-6284-9286**. Markus: ORCID **0009-0007-3575-1006**, verified
in multiple existing manuscripts and the operating-band `CITATION.cff`.
Sources and clickable profiles are in `../arxiv/authors.json` and its README.
These details are excluded from the anonymous source ZIP and review PDF.

The venue permits at most nine authors in its general guidance, and the template
caps self-citations at 20%. Recheck the latter against the approved author list.

## Camera-ready steps after acceptance

- Apply the reviews; use the acceptance category's limit: 12 pages for Full,
  8 for Short. Up to four extra pages incur fees; none are assumed here.
- Insert approved author names, affiliations, emails and applicable ORCIDs using
  `\authorname`, `\affiliation`, `\email` and the supplied template conventions.
- Add factual acknowledgements: conflicts, funding/support, contributions, data
  sharing and AI use. Do not copy example claims of no funding/no conflicts.
- Restore the appropriate public code link, update PDF metadata and rerun the
  entire build, reference, page-count and visual audit. Author blocks consume space.
- Complete copyright consent, registration and presentation requirements. Only
  acceptance plus the publisher's publication conditions authorize a final paper.

## Optional clarification email draft — not sent

To: icaart.secretariat@insticc.org

Subject: ICAART 2027 anonymous submission: AI-use disclosure placement

We are preparing a regular-paper submission. The guidelines ask for AI-use
disclosure in acknowledgements while also requiring acknowledgements to be
removed for double-blind review. The dedicated AI policy permits an appropriate
section. May we include a non-identifying AI-use disclosure before the references,
with tool citations in affected sections and a tool/prompt disclosure in the
conceptual-figure caption, and move the disclosure to acknowledgements after
acceptance? Please also clarify how the review-period posting restriction applies
if a manuscript version was already publicly accessible before submission.
