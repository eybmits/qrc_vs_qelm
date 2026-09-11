# Figure 2 resource atlas

The author approved this design on 11 September 2026. The publication version
is integrated into the ICAART manuscript from `paper/figures/resource_atlas`.
Preview rendering remains available without modifying the manuscript.

The reference paper's Figure 5 inspired the stacked, aligned panels and compact
grouping. Its scientific data, labels and conclusions have not been copied.
The author-selected Cobalt palette is reused.

- **History:** replace the product axis with the actual tested window lengths
  and loss rates. Blue requires the existing sampled criterion (at least 10 of
  12 roots at or below 10% old-history NRMSE) in every size cohort, separately.
  Gray is a failed sampled criterion, not proof of permanent memory.
- **Features:** separate the three target sizes into aligned small plots, with
  equal scales and existing 10th–90th percentile bands. The transition budgets
  18–72 are shown; 108 and 144 remain in the source export.
- **Shots:** show the median error for each feature/shot setting, including the
  validation-selected policy. Outlines use unrounded medians below 10%.
  Add the existing validation-choice frequencies, checked against all 768
  selected rows. This is not selection on the test set. Shot uncertainty is
  shown as 10th–90th percentile ranges below each median, rounded to whole
  percentage points. Full-precision values remain in the source exports and JSON.

These are separate studies, not one matched full-factorial experiment. Nothing
is fitted or simulated here. The candidate is 7.055 × 2.85 inches; at the paper's
6.221-inch width its height is about 2.51 inches (the current figure is 2.12).
It contains more information but is not vertically smaller. Redundant caption
and evaluation prose was shortened to preserve the twelve-page layout.

Regenerate from the repository root:

```sh
MPLCONFIGDIR=/private/tmp/qrc-mpl-cache .venv/bin/python paper/figure2_candidates/render_resource_atlas.py
MPLCONFIGDIR=/private/tmp/qrc-mpl-cache .venv/bin/python paper/figure2_candidates/render_resource_atlas.py --publication --output paper/figures/resource_atlas
MPLCONFIGDIR=/private/tmp/qrc-mpl-cache .venv/bin/pytest -q paper/figure2_candidates/test_resource_atlas.py
```

The renderer writes PNG/PDF and CSV/JSON provenance into the selected directory.

Verification: the PNG and rendered PDF were visually inspected. Tests check
source coverage, validation-choice reconstruction, data summaries, minimum
7-point text at manuscript width, non-overlapping axes, no clipped labels,
deterministic rendering, and no manuscript changes from preview rendering.
The current PDF identity and visual review are recorded in the ICAART manifest,
not pinned to a previous revision here.
