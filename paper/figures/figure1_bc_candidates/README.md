# Figure 1: choose one b and one c

These ten alternatives are a selection set, not a manuscript revision. No
candidate is selected yet. The current Figure 1 and both manuscript PDFs stay
unchanged.

Open [selection.pdf](selection.pdf) to compare five versions of b on page 1 and
five versions of c on page 2. Every diagram is shown at its native 8.6 x 5.6 cm
size with 10-point labels. The B and C numbers are independent: any B candidate
can be paired with any C candidate.

## Files

- `b1.tex` through `b5.tex`, and `c1.tex` through `c5.tex`: editable TikZ sources.
- Matching `.pdf` files: vector artwork; matching `.png` files: 1400-pixel previews.
- `overview_b.png` and `overview_c.png`: selection sheets with short German hints.
- `style.tex`: the shared typography, geometry, colors, arrows, and measurement glyph.
- `b_descriptions.json` and `c_descriptions.json`: fuller captions and scientific scope.
- `selection_guide.json`: the short captions used on the selection sheets.
- `candidate_manifest.json`: artifact checksums and the unchanged-manuscript check.
- `render_figure1_bc_candidates.py` and `test_figure1_bc_candidates.py`: the
  isolated editorial renderer and its tests, not part of the scientific release.

## Meaning and boundaries

Panel b explains missing-history ambiguity: different pasts can be compatible
with one recent input window, so a window-only predictor may have to approximate
different target outputs with one answer. The alternatives use output ranges,
two-history diagrams, interval ladders, tolerance curves, or repeated updates.

Panel c distinguishes output-relevant state differences from differences that
remain invisible under every allowed shared future continuation, including no
further input. Equal output at the current time alone is insufficient. The
alternatives use measurement pipelines, an influence graph, now/later output
comparisons, invariant invisible directions, or future-output equivalence groups.

All diagrams are conceptual, not new simulations. Ranges are enclosing output
intervals, not statistical confidence intervals; some interior values need not
be attained. A few drawn paths or pairs cannot establish uniform fading or
permanent invisibility. The relevant universal conditions are stipulated in
the illustrations and explained in each candidate's scope. Midpoint predictions
refer to optimal worst-case scalar absolute error, not average squared error.
Neither panel alone establishes efficient emulation by a fixed-feature QELM.

Presentation reference: Figure 1 of the user-supplied
*Superextensive learning in quantum reservoirs at the onset of information
scrambling*, by Freiheit and Campaioli (`2608.25511v1 (5).pdf`, page 2). These
are newly authored diagrams for this paper; no data from that reference are reused.

## Rebuild

From the repository root, with `pdflatex`, `pdfinfo`, and `pdftoppm` available:

```sh
.venv/bin/python paper/figures/figure1_bc_candidates/render_figure1_bc_candidates.py
.venv/bin/pytest -q paper/figures/figure1_bc_candidates/test_figure1_bc_candidates.py
```

The renderer compiles each native panel separately and builds two selection
pages with a two-pass LaTeX layout. Temporary build files stay outside this
directory. Its manuscript guard rejects changes to canonical Figure 1, the
selected architecture source, both paper PDFs, the frozen source provenance,
or the existing release manifest during the build. It does not modify them.
