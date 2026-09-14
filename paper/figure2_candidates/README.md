# Figure 2 resource atlas

The author approved the atlas on 11 September 2026 and requested a more intuitive
measurement-budget panel on 14 September. The publication version is integrated
into the ICAART manuscript from `paper/figures/resource_atlas`.
Preview rendering remains available without modifying the manuscript.

The reference paper's Figure 5 inspired the stacked, aligned panels and compact
grouping. Its scientific data, labels and conclusions have not been copied.
The active paper palette uses blue shades for QELM errors and feature counts,
with marker shapes retaining distinctions between settings.

- **History:** replace the product axis with the actual tested window lengths
  and loss rates. Blue requires the existing sampled criterion (at least 10 of
  12 roots at or below 10% old-history NRMSE) in every size cohort, separately.
  Gray is a failed sampled criterion, not proof of permanent memory.
  The display starts at loss rate 1, retaining seven tested rate rows and the
  exact cell-edge staircase. In-cell numbers and the lower-rate annotation are
  omitted; only integer rates 1, 2, 3, 4 are labeled. All 198 source cells remain
  in exports. The four omitted rate rows below 1 fail the sampled criterion
  for every tested window, as noted in the caption.
  Compatible data stop at six inputs; the separate N5 study is not spliced in.
- **Features:** separate the three target sizes into aligned small plots, with
  equal scales and existing 10th–90th percentile bands. The transition budgets
  18–72 are shown; 108 and 144 remain in the source export.
- **Shots:** show all 16 medians as connected points with capped 10th–90th
  root-percentile whiskers. Light, medium and dark blue circles, squares and
  triangles distinguish 18, 36 and 72 features; navy dashed diamonds show the
  validation-selected policy. Ordered shot tiers increase fourfold. Small
  within-tier offsets separate nearly coincident values; lines are visual guides,
  not fitted curves. The horizontal 10% reference is not the 10-of-12-root success
  verdict. Direct feature-count rows and shot-tier columns show all 12 choice
  frequencies, rounded to whole percentages. Pale blue/bold cells identify the
  most frequent count. Each column represents 192 validation choices (12 roots
  and 16 repetitions), checked against all 768 rows. Choices are made before
  testing. Aggregation and all source values are unchanged; full precision
  remains in CSV/JSON exports.

These are separate studies, not one matched full-factorial experiment. Nothing
is fitted or simulated here. The current canvas is 7.055 × 3.05 inches; at the
paper's 6.221-inch width its height is about 2.69 inches. The added space enlarges
the staircase and replaces opaque stacked shares with directly labeled values.

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
