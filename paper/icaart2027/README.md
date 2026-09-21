# Anonymous ICAART manuscript

`submission.pdf` is the 12-page anonymous manuscript.
`ICAART2027_anonymous_source.zip` contains its LaTeX sources, bibliography,
styles, and three figures.

## Build

From the repository root, with the Python dependencies, LaTeX, `latexmk`,
BibTeX, and Poppler installed:

```bash
.venv/bin/python paper/icaart2027/build_submission.py
```

The builder checks format, anonymity, references, embedded fonts, and a
byte-identical rebuild from the source archive. File hashes are recorded in
`ICAART2027_MANIFEST.json`; checks are recorded in `submission_qa.json` and
`visual_review.json`.

`submission_approval.json` records author-confirmed coauthor approval and AI
disclosure review for the exact PDF hash. A changed PDF requires renewed
confirmation. Automated disclosure-presence diagnostics remain visible in the
QA report; they do not override the recorded human applicability review.
The approval record is excluded from the anonymous source archive. Repository
CI and the actual portal upload are separate from these manuscript checks.

## Sources and figures

The manuscript is in `main.tex` and `sections/`; references are in
`references.bib`. The SCITEPRESS template files are supplied locally.
`figures/` contains the three figures used in the paper.

After regenerating figure sources, update the anonymous figure copies with:

```bash
.venv/bin/python paper/icaart2027/prepare_assets.py
```

The [arXiv edition](../arxiv/README.md) is generated from the same scientific
sources. [Reproducibility instructions](../../REPRODUCIBILITY.md) cover the
experiments and source data.
