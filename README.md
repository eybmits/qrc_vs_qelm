# qrc_vs_qelm

Code, data, figures, and manuscript sources for **When Can Quantum Extreme
Learning Machines Replace Quantum Reservoirs?**

The paper studies when recent inputs suffice to reproduce a quantum reservoir's
outputs, separating missing history, model capacity, and measurement error.

## Paper

Accepted at [ICAART 2027](https://icaart.scitevents.org/).
Read the [preprint on arXiv](https://arxiv.org/abs/2609.31027).

| Version | PDF | LaTeX source |
| --- | --- | --- |
| Anonymous ICAART | [PDF](paper/icaart2027/submission.pdf) | [ZIP](paper/icaart2027/ICAART2027_anonymous_source.zip) |
| arXiv, with authors | [PDF](paper/arxiv/preprint.pdf) | [ZIP](paper/arxiv/arxiv_source.zip) |

Both versions have 12 pages and share the same scientific content.

## Install and test

Python 3.11 or newer:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" -r requirements-ci.txt
.venv/bin/pytest -q
```

Paper and vector-figure builds also require LaTeX, `latexmk`, BibTeX, Poppler,
and the `orcidlink` package.

## Build the paper

Run from the repository root:

```bash
.venv/bin/python paper/icaart2027/build_submission.py
.venv/bin/python paper/arxiv/build_arxiv.py
```

Each builder checks that its source archive reproduces the delivered PDF.
See the [anonymous build instructions](paper/icaart2027/README.md) and
[arXiv build instructions](paper/arxiv/README.md).

## Reproduce the results

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) maps the paper's results to protocols,
configurations, data, and commands. Use fresh output directories when rerunning
experiments; retain the recorded seeds and keep independent cohorts separate.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/qrc_vs_qelm/` | Reservoirs, QELMs, simulation, fitting, and analysis |
| `scripts/` | Study runners and analysis commands |
| `configs/`, `data/` | Configurations, seed records, and reference data |
| `results_*/` | Selected recorded results and supporting evidence |
| `pra_extension/` | Additional experiments, analytical checks, and figure tools |
| `paper/` | Manuscripts, figure sources, and build tools |
| `tests/` | Core regression tests |

Scientific protocols and preregistrations are retained alongside the evidence.
The older `paper/canonical_emulation/` manuscript and selected historical inputs
support regression checks; the two current editions are linked above.

## Citation and license

See [CITATION.cff](CITATION.cff) for citation metadata. Code is distributed under
the [MIT License](LICENSE).
