# Previous taller full-width Figure 3 layout

This earlier layout is retained for comparison. The active Figure 3 is in
`../memory_prediction/`, with equally high plots in every row. This variant's
canvas is 6.221 × 4.69 inches. Every plot is
1.51 inches wide; the first two rows are 0.78 inches high and the bottom row is
1.00 inch high. Plot edges align across all three columns.

The renderer reuses the existing Figure 3 data and drawing code, then changes
only layout and explanatory label placement. Target traces remain black, QRC
orange, QELM blue, and the full-history QRC reference red. The source ledger
records both the parent renderer and this layout variant.

Regenerate from the repository root:

```sh
MPLCONFIGDIR=/tmp/qrc-mpl-cache .venv/bin/python pra_extension/code/render_memory_prediction_full_width.py
```

Include the PDF at `width=\textwidth`. The reviewed manuscript with this layout
is delivered separately as `QRC_vs_QELM_ICAART2027_layout_draft.pdf`, alongside
a reproducible TeX source ZIP. It has 13 pages; the submission page limit remains
12. The active paper uses the consistent-height figure instead of this variant.
