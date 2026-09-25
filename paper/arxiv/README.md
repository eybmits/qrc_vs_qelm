# arXiv manuscript

`preprint.pdf` is the 12-page identified manuscript.
`arxiv_source.zip` contains the LaTeX sources, figures, styles, bibliography,
and generated `main.bbl`.
The unused alternative `sections/related-work.tex` is excluded from the upload;
the compiled manuscript is unchanged.

Authors: **Markus Baumann, Gerhard Stenzel, Jonas Stein, and Claudia Linnhoff-Popien**,
QAR-Lab, Department of Computer Science, LMU Munich.
Affiliations, profile links, ORCIDs, and correspondence details are stored in
`authors.json`.

## Build

From the repository root, with the Python dependencies, LaTeX, `latexmk`,
BibTeX, Poppler, and `orcidlink` installed:

```bash
.venv/bin/python paper/icaart2027/build_submission.py
.venv/bin/python paper/arxiv/build_arxiv.py
```

The arXiv builder starts from the anonymous source package, adds the author
block and metadata, and appends the full repository link. The anonymous
manuscript contains no repository link or omission note. The scientific
sections and figures are shared. Title-to-author spacing
uses an 8 pt addition; manuscript font sizes are preserved.

`ARXIV_MANIFEST.json` records source and file hashes. `arxiv_qa.json` checks
authors, links, fonts, page geometry, preserved content, and a byte-identical
rebuild from the source ZIP. `visual_review.json` records layout verification.
These build records do not track the separate arXiv upload or publication status.

The source archive can also be extracted and compiled with
`latexmk -pdf main.tex`.
