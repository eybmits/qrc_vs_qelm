# Author-identified preprint and anonymous review copy

The author-identified edition lists **Markus Baumann, Jonas Stein, and Claudia
Linnhoff-Popien**, in that order, as instructed on 14 September 2026. All three
share QAR-Lab, Department of Computer Science, Ludwig-Maximilians-Universität
München (LMU Munich), Munich, Germany. Markus is the correspondence contact at
markus.baumann@campus.lmu.de, consistent with the existing manuscript metadata.

| Edition | PDF | Source ZIP |
| --- | --- | --- |
| arXiv preprint, with authors | `preprint.pdf` (12 pages) | `arxiv_source.zip` |
| Anonymous ICAART copy | `../icaart2027/submission.pdf` (12 pages) | `../icaart2027/ICAART2027_anonymous_source.zip` |

Both editions have twelve pages. The authored title block uses explicit
12 pt spacing after the title and neutralizes the inherited negative
title-author skip, increasing their baseline separation from 13.89 to 24.91
PDF points. Publisher font sizes are unchanged: title 14.4 pt, authors 11 pt,
affiliation/email 9 pt. Body fonts and page margins match the anonymous edition.
The authored build preserves the anonymous scientific content and the
author-approved AI disclosure. In the identified
edition, the anonymity-only code note becomes a repository link labeled as
currently access-restricted. The repository is private as verified on 14
September 2026. No repository visibility change or public upload was made.

The anonymous edition uses the official ICAART-linked SCITEPRESS template
files unchanged. The arXiv edition retains its typography and page dimensions
but adjusts title-page spacing for the author block. Its linked green ORCID
icons use `orcidlink`, as in the authors' other manuscripts. This is distinct
from the publisher's `\orcidAuthor` macro, which also adds ORCID footnotes.

The latest Introduction and Discussion and Conclusions are the author's
supplied replacement text in both editions. All appendix material, including
Time-Series Demonstrations, remains intact within the twelve-page budget.
Both editions also share the 16 September figure refinement: no descriptive
plot titles, a shared-input viewport of 1--32 in the memory-gap plot, and
consistent, slightly thicker laser/NARMA trace lines and model markers.
Appendices A--E now use the author's revised appendix PDF. Semantic references
and the main-text midpoint notation are aligned with the replacement; equation
numbers are generated from the revised source. The anonymous and authored
appendices differ only in the repository-access sentence.

## Identity sources

- Markus Baumann: [QAR-Lab team](https://qarlab.de/en/our-team/) and
  [his existing paper's author block](https://arxiv.org/html/2607.01187v1)
  establish the affiliation and correspondence email. ORCID
  [0009-0007-3575-1006](https://orcid.org/0009-0007-3575-1006) is verified in
  his existing operating-band, VQML-symmetry, and IQP-generative-modeling
  manuscript author blocks and the operating-band `CITATION.cff`.
  `authors.json` records repository-relative source paths, line numbers,
  source hashes and the matching author definitions.
- Jonas Stein: the [official LMU profile](https://www.ifi.lmu.de/mvs/en/team/contact-page/jonas-stein-ef892b38.html)
  gives jonas.stein@ifi.lmu.de and links to ORCID
  [0000-0001-5727-9151](https://orcid.org/0000-0001-5727-9151).
- Claudia Linnhoff-Popien: the [official LMU profile](https://www.ifi.lmu.de/mvs/en/team/contact-page/claudia-linnhoff-popien-5eb3fc37.html)
  gives linnhoff@ifi.lmu.de; an [official LMU publication record](https://epub.ub.uni-muenchen.de/128888/)
  identifies ORCID [0000-0001-6284-9286](https://orcid.org/0000-0001-6284-9286).

All three ORCID check digits are validated. Each author has a clickable green
ORCID icon beside their name; author names themselves link to official profiles.

## Rebuild

From the repository root, with Python/pypdf, a complete TeX Live installation,
`latexmk`, BibTeX, `orcidlink` (with TikZ/hyperref), and Poppler available:

```bash
python paper/arxiv/build_arxiv.py
```

The builder starts from the current anonymous source ZIP and checks it against
the anonymous manifest and local sources. It changes only the author header
and its spacing, PDF metadata, and the administrative code-access sentence. Authorship stays
outside the anonymous ZIP. The arXiv ZIP includes the common sources and
figures, required publisher style files, and generated `main.bbl`, with one
top-level `main.tex`. Extract the ZIP and run `latexmk -pdf main.tex`.

`ARXIV_MANIFEST.json` records the parent source, administrative changes and file
hashes. `arxiv_qa.json` records author order, hyperlinks, metadata, embedded
fonts, page geometry, preserved scientific content, unchanged anonymous PDF,
and a byte-identical clean source rebuild. Visual inspection is recorded
separately against the final PDF hash in `visual_review.json`.

The package has not been uploaded, so it has no arXiv ID, publication date or
chosen arXiv license. Source preparation follows the [arXiv TeX guidance](https://info.arxiv.org/help/submit_tex.html).
The final PDF produced by arXiv must also be checked during an actual upload.
