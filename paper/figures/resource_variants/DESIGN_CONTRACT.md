# Figure 2: nine presentation alternatives

Presentation-only selection board for the author. The live manuscript and its
figures are not replaced. No fits, experiments, exclusions or scientific claims
are added. The source is the current, hash-checked 912-row Figure 2 export;
the three panels retain their own cohorts and error definitions.

## Chart contract

Static PNG selection sheets and individual editable SVG/PNG panels, produced
by `pra_extension/code/render_resource_variants.py`. Every alternative retains
all tested settings and models. Values are root medians; intervals/bands are
10th–90th root percentiles, not confidence intervals. Lines join tested values,
not fitted curves. Panel c retains its logarithmic shot axis. The paper's
categorical colors are an explicit palette exception for continuity; marker
shapes, open fills, line dashes and facets supply non-color identification.
This is the author's third-party manuscript: no vendor branding is inserted.

| Panel | Comparison | Version 1 | Version 2 | Version 3 |
|---|---|---|---|---|
| a | old-history error vs scaled context; 504 root points | airy line and interval | percentile ribbons | one facet per QRC size, roots plus summaries |
| b | emulation error vs six feature budgets; 216 root points | airy line and interval | same overview plus explicitly labeled median detail | one facet per QRC size, roots plus summaries |
| c | test error vs four shot budgets; 192 root/model points | four lines, selected-policy intervals | selected-policy ribbon, fixed-budget references | one facet per feature budget, each against the same selected policy |

The history panel is log-scaled in y in every version. Feature and shot error
axes are linear. All versions within a panel share their main-axis limits.
In b2 only, the inset enlarges B>=36 and explicitly shows medians only; the
complete range and all percentile intervals remain in the main axes.
In c3 the selected policy is deliberately repeated in each facet; it is not
a different policy fitted separately for each displayed feature count.

The selection boards provide equal footprints, variant IDs a1–c3 and short
German style names. English axis labels retain the manuscript terminology.
Main and minor background grids remain quiet. Insets, legends and notes must
be inspected in the exported images before handoff. A chosen design may need
one final typography pass at actual three-column manuscript size.
