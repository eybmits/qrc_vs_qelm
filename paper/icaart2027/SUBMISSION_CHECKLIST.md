# ICAART 2027 submission handoff

Prepared on 6 September 2026. This is an anonymous **editorial draft** in the
Regular Paper format, not a submission-ready package or a final publication.

## Blocking issue: required preparation disclosure removed

The author requested removing all preparation-tool mentions from the paper.
The active manuscript therefore has no AI-use disclosure, affected-section
tool citations or conceptual-figure tool/prompt credit. Known preparation
history is unchanged: OpenAI Codex assisted research development, mathematical
exposition, drafting/revision, simulation and analysis code, and figure
preparation. It is not an author; human authors remain responsible.

The live [ICAART submission guidelines](https://icaart.scitevents.org/Guidelines.aspx?y=2027)
require disclosure and affected-section system citations. The
[dedicated policy](https://icaart.scitevents.org/AiTools.aspx?y=2027) additionally
specifies figure-credit requirements. Checked on 6 September 2026.
**Do not submit the current draft under those rules.** Resolve the disclosure
requirement with the authors and, if placement is uncertain, the organizers.
No organizer exception has been obtained. These notes remain outside the PDF
and source ZIP; they are not a substitute for required in-paper disclosure.

Automated QA now records separate `format_passed` and `policy_passed` values.
The missing-disclosure gate remains false, `submission_ready` remains false,
and the build manifest status is `EDITORIAL_DRAFT_NOT_SUBMISSION_READY`.

## Portal fields

- Conference: ICAART 2027, 19th International Conference on Agents and Artificial Intelligence.
- Submission type: Regular Paper, first stage.
- Area: Artificial Intelligence.
- Primary topic: Quantum Artificial Intelligence; secondary: Machine Learning.
- Title: When Can Finite-Context QELMs Replace Quantum Reservoir Memory?
- Abstract: use the plain text from `sections/abstract.tex`.
- Keywords: Quantum Reservoir Computing; Quantum Extreme Learning Machines;
  Temporal Learning; Fading Memory; Finite Context.
- Eventual review attachment: the approved, policy-compliant PDF, not the current
  disclosure-free `submission.pdf`, full repository or this handoff.
- First-stage submission deadline: **15 September 2026, Anywhere on Earth**.
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
  remaining locally held N5 files; the anonymous LaTeX ZIP reproduces the PDF,
  not the experiments. A withheld identifying URL does not resolve reviewer access.
- [ ] Resolve the required-disclosure blocker above; a passing formatting check
  is not venue-policy clearance.
- [ ] Approve every proof, result, figure, reference, and the factual preparation description.
  This formatting pass is not an independent scientific correctness/novelty review.
- [ ] Confirm final author list/order, affiliations, corresponding author and email;
  supply any ORCIDs. The portal stores author details even though the review PDF omits them.
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
  current editorial draft has removed disclosure entirely. Organizer confirmation
  on placement does not itself waive the underlying disclosure requirement.
- [ ] Upload the final PDF in PRIMORIS and retain the submission confirmation.

No portal action, external message, registration, payment, consent or submission
has been performed.

## Author data found in the project, not yet approved as the final list

The current `CITATION.cff` records **Markus Baumann**,
**Ludwig-Maximilians-Universität München**. `pyproject.toml` records
**markus.baumann@campus.lmu.de**. No ORCID or additional author was found in
these records. This is project metadata, not an assertion that the conference
author list is complete. These details are not included in the review package.

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
