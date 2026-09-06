# Main-figure chart map

## Shared publication design

### Current Figure 1 revision contract

The current reference-driven revision uses one centered 4.40-cm-square input
frame and one 4.40-cm-square output frame. The 201-sample input has
an interior selected time t=180, a dashed guide and a purple horizontal
connection at its true ordinate. A bracket sends only the exact green causal
window 141--180 to reset-QELM; there is no duplicate lower plot. Grey future
samples 181--200 are display context. The input amplitude domain remains
[0.3,1.3]. Small horizontal margins protect endpoint dots from the frames.

The single right-hand plot uses black lines/filled QRC markers and green
lines/hollow window-reference markers. Two toy histories differ only in an
initial bit b=+1 or -1 at time 0, then share all later inputs. The retained
output is g(t)+0.25b. Before t=40, a 40-input window can read the bit; from
t=40 onward its balanced conditional-mean prediction is g(t), also the scalar
minimax midpoint of the two outputs. Only the b=+1 history is drawn. Thus the
curves agree until the bit leaves the window, then differ by its retained
0.25 contribution. A normalized independent 101-sample Mackey-Glass shape
provides g(t), not simulated QRC dynamics or a trained QELM output. This toy
is separate from the left input illustration and all experiments. The caption
discloses the schematic scope; no separate in-art schematic tag, training/test
split or uncertainty band is added. Source/provenance record values and formulas.

Compact gold readout triangles are unlabeled; the comparison brace groups
both architecture outputs. Panel b remains centered below a; separate c stays
removed. Final QA uses the vector figure and both manuscript PDFs against
page 2 of the supplied `2608.25511v1 (7).pdf` at equal rendering DPI.

### Current main-paper selection

Both manuscripts now select three figures, with unchanged artifact filenames:

| Current figure | Retained artifact | Previous number |
| --- | --- | --- |
| 1: architecture and theory overview | `fig01_information_interface.pdf` | 1 |
| 2: history, features, and shots | `fig03_resource_diagnosis.pdf` | 3 |
| 3: separate five-qubit longer-context study | `../long_context_n5/fig04_long_context_n5.pdf` | 4 |

The former Figure 2, `fig02_information_frontier.pdf`, is retained as an
archival phase-map asset, not a main-paper float. Its PNG, 198-row source CSV,
renderer, classifications, and upstream data remain unchanged. The three
size panels have the same tested primary minimum-window frontier and similar
curve shapes; the retained resource and longer-context figures carry the
main visual argument more compactly. This is an editorial omission, not
evidence rejection, cohort pooling, or a new numerical result. The phase-map
findings and qualifications remain in the text. No combined plot is created.
Rendered review confirms Figures 1, 2 and 3 on pages 2, 5 and 6 in both
manuscripts. Both versions have twelve pages. Advancing the conference
resource float removes an otherwise empty
column before the discussion, without resizing artwork or changing content.

### Current refinement contract

The user rejected the faint per-root trajectories and the substitute colors.
The next rendering uses the actual reference palette: blue `#3B86FF`, green
`#00A14B`, red-orange `#FC4D30`, gold `#FFC943`, purple `#A93BFF`, and its
viridis sequence for ordered context/feature categories. No custom lookalike
palette is retained. Marker shapes and direct labels preserve non-color cues.
The resource figure (current Figure 2) reports medians with explicit 10th--90th root-percentile whiskers,
not bundles of faint trajectories or overlapping shaded bands. Every source
root remains in the summaries and exports. These whiskers are descriptive
root variability, not a confidence interval. Figure 1 now uses two connected
native-TikZ quantum workflows rather than the earlier left/right composition.
New larger-system experiments have a separate protocol and seeds; their
results must not silently replace or be pooled with existing cohorts.

- Native width: 7.055 inches, included at the manuscript text width.
- Figure 1: 17.9197 by 13.0 cm (7.055 by 5.118 inches), a landscape
  overview; current Figure 2: 2.40 inches high; current Figure 3: 2.65 inches
  high. The archival phase-map figure remains 2.65 inches high.
- Upright text: 9--11 pt for primary labels and ticks, close to the paper's
  10 pt body text; compact legends and direct curve labels use 8--8.5 pt.
  Helvetica text and Computer Modern mathematics; vector PDF artwork throughout.
- White backgrounds, reference blue/green/red-orange with gold measurement/readout
  accents in Figure 1, panel letters and essential
  object labels only; no plot titles, number pills, or interpolated heatmaps.
- All five main-paper evaluation axes, the archival phase map's three axes,
  and Figure 1's single input-series and single output-comparison axes
  have physically square plotting boxes.
  The user-selected B4 tolerance diagram uses 6.40-by-2.40-cm
  axes centered in Figure 1's lower strip. Its input and output axes are
  4.40-cm squares within panel a. Whole multipanel canvases need not be
  square. Current Figure 2 has one row of three square axes and current
  Figure 3 one row of two; the archival phase map retains its three square axes.
  equal data-unit scaling is not imposed.
- Quantitative panels use four thin visible spines, outward ticks, bold axis
  labels, inset panel letters and compact in-axis legends. There are no
  figure-level legends or large panel titles. Reservoir-size conditions and
  direct window labels are essential identifiers, not narrative titles.
- The reference color roots are blue `#3B86FF`, green `#00A14B`, red-orange
  `#FC4D30`, purple `#A93BFF`, and gold `#FFC943`, with black ink. Ordered
  contexts use the reference's viridis colors `#7AD151`, `#44BF70`, `#1FA088`,
  `#2A788E`, `#414487`, `#482475`; their small end labels remain black.
  Feature identities retain distinct marker shapes; history-status marks use
  open versus filled symbols independently of their window color.
- Quantitative error axes show normalized RMSE as percentages: 10% is 0.10,
  not a change in the estimand or a percentage-of-target error.
- References: the user-supplied *Finite-shot access laws for quantum reservoirs*
  PDF and Figure 1 on page 2 of `2608.25511v1 (4).pdf`, plus the updated
  `2608.25511v1 (5).pdf` plot-style reference. The latest reference audit found
  roughly 8-pt axis labels at actual page size. Our primary quantitative labels
  are 10 pt at native figure width and 8.82 pt after ICAART placement; ticks
  are 9 pt native and 7.94 pt on the conference page. The reference's four-spine axes,
  outward ticks and inset keys inform the quantitative styling. Their concrete network,
  measurement, and process illustrations inform the design; their scientific
  claims and data are not incorporated.

### Same-scale rendered review

The supplied reference and the ICAART PDF were both rendered at 96 dpi,
preserving equal PDF-point-to-pixel scaling rather than fitting differently
sized crops to a preview window. High-resolution inspection supplemented
this print-size check. PDF font/placement inspection gives a 0.88182 scale
for our 7.055-inch source artwork: 6.221 inches wide on the ICAART page.
The body is 9.96 pt; Figure 1 labels are 9.66 pt, quantitative axis labels
8.82 pt, ticks 7.94 pt, inset panel letters 9.26 pt, archival phase-map
legend/end labels 7.05 pt, and resource-figure legends 7.50 pt. This follows the reference's
hierarchy without claiming identical fonts at identical physical sizes.

Iteration 1 replaced the archival phase map's separate status matrix with integrated
markers. Iteration 2 shortened its failure key, darkened the two palest
curves and added upper clearance for panel letters. The resource figure's first compact
layout was checked for xlabel clipping; a one-line scaled-context label
and a tighter canvas removed the excess whitespace. The final pass used
rounded in-axis legend boxes and placed the validation-selected policy
above overlapping fixed-budget traces. Caption checks retained all cohort,
root-level classification and percentile qualifications. The subsequent pass
replaced all resource-figure individual-root traces and bands with capped percentile
whiskers, and switched every active color to the actual reference palette.

## Figure 1 — When finite context can replace recurrent quantum memory

- **a:** One input plot, two architectures, one shared output plot. QRC receives the current scalar and carries prior state; reset-QELM encodes the entire bracketed green window from fresh states. Gold model-specific observable banks feed compact unlabeled readout triangles. The target readout remains fixed during emulation; the numerical target is unchanged.
- **Input evidence:** The first 201 seed-1 Mackey-Glass samples are displayed once. The current point is t=180 and the exact 40-value causal window is 141--180; future 181--200 is grey and excluded from model access. The input amplitude domain remains [0.3,1.3]. Values are not rescaled; the original 120-value prefix is preserved. This is an input-access illustration, not evaluation data.
- **Output reference:** One plot shows the positive-bit branch of the conditional toy above: tracking while the bit is in the window, then loss of its retained output contribution. Black filled/solid QRC and green open/solid window reference retain non-color distinction. The unseen negative-bit companion defines the balanced conditional mean and pair midpoint. The 101-sample baseline is independent of the left plot; neither curve is a fitted or simulated model result.
- **b:** The selected B4 panel retains the unrestricted scalar history optimum D_d/2 and a smooth analytic exponential crossing tolerance epsilon, with a contrasting persistent floor. The entire panel is translated +4 cm in x without changing its curve or axis geometry.
- **Scope:** No separate c remains. Theorems on reachable/observable memory remain in the body. The conditional toy is not an empirical QELM failure or a universal failure-time guarantee; its green curve is an ideal window reference. The caption states that the curves and crossing are schematic.
- **Geometry and typography:** Canvas 17.9197 by 13.0 cm; one 4.40-cm-square input plot at [0.92,6.60,5.32,11.00], one 4.40-cm-square output plot at [13.40,6.60,17.80,11.00], and 6.40-by-2.40-cm history axes. Native Helvetica 11-pt axes/architecture, 12-pt letters and deliberately smaller 10-pt key/event labels. No trained-readout words or plot titles. See the current revision contract above for palette and final-context QA.
- **Source identity:** Selected Variant A equals canonical TikZ byte-for-byte. New output coordinates and current scope are recorded in render_source.json; the historical source.json remains frozen. Numerical data, gates and other figures are not altered.

## Archival phase map — Former Figure 2

- **Selection status:** Omitted from both main-paper figure sequences, with
  the complete `fig02_information_frontier` PDF/PNG/CSV and renderer retained.
  The numerical comparisons below still define this archival asset.
- **Evidence retained in text:** The dense map covers 11 loss rates and six
  windows per size, with a separately frozen four-qubit follow-up. At the
  primary 0.10 tolerance and 10-of-12 rule, all three sizes share the tested
  minimum-window sequence 6, 5, 4, 3, 3, 2 at losses 1.25, 1.5, 2, 2.5,
  3, and 4; no tested window is history-sufficient through loss 1. This is a
  finite tested crossover, not an asymptotic size law. Retain the distinction
  between median error and robust root-level success, the feature-limited
  anchors, sensitivity qualifications, and the independent-cohort boundary.

- **Layout contract:** Three integrated, square panels in one
  7.055-by-2.65-inch static Matplotlib figure replace the separate history
  curves and categorical grids. All 198 measured history cells and their
  original robust outcomes remain visible at the same coordinates. Inset
  panel letters and reservoir-size labels identify the columns. Direct
  `$d=1$` through `$d=6$` labels identify the curves. Compact in-axis keys
  explain status markers and the target; no titles or figure-level legends.
- **a--c:** Six window-length curves per target-QRC size, at the actual
  numerical loss coordinates and root-median history NRMSE on the vertical
  log scale. Lines connect measured points, without fitting, smoothing,
  interpolation, or monotonic enforcement. Ordered green-to-purple colors
  identify the windows, not feature budgets. The horizontal dashed line
  marks the registered 0.10 target. The upper y limit leaves headroom for
  inset letters/conditions; it does not change any measured value.
- **Integrated marker outcomes:** Open circle: history target met in fewer
  than 10 of 12 roots. Cross: history is sufficient in at least 10 roots,
  but no tested QELM budget is robustly successful. Filled circle, square,
  and diamond: respectively the smallest tested 18-, 36-, or 72-feature
  budget meeting the total-error target in at least 10 roots, after robust
  history sufficiency. Markers inherit their curve's window color. This
  uses independent shape/fill and color channels rather than reusing one
  color for two meanings.
- The old lower grids are removed, not compressed into hidden insets. Their
  classifications now appear directly on every history curve, including the
  90 low-loss cells previously omitted from the categorical row. All those
  low-loss cells retain their history-unmet status.
- A median below the target is distinct from meeting it in 10 of 12 roots.
  Exactly six N=3/N=4 cells have a sub-target median but only six or eight
  history-passing roots; they retain open circles below the dashed line.
  At loss 2, window 5 succeeds and window 6 fails the QELM rule for each
  reservoir size. Those diamond/square versus cross outcomes remain; no
  monotonic success region is drawn. These statuses do not claim failure
  of all possible QELM architectures or all feature budgets.
- **Cohorts:** N=2 and N=3 are the earlier frozen dense cohort; N=4 is the
  separately frozen prospective follow-up. They are never pooled.
- **Sources:** `results_dense_phase_secondary_v1_analysis/dense_phase_cells.csv`
  and `results_dense_phase_n4_v1_analysis/dense_n4_phase_cells.csv`, validated
  against their raw suffix tables and provenance hashes. The complete
  198-row source export is byte-unchanged, representing 2,376 root-level
  history cells and 7,128 root-budget QELM cells.

## Figure 2 — Three resource interventions (formerly Figure 3)

- **Artifact identity:** `fig03_resource_diagnosis.pdf` and its PNG/source CSV
  keep their existing filenames and contents; only the manuscript number changes.

- **Layout contract:** Three aligned line-and-marker panels in one horizontal
  row at 7.055 by 2.40 inches, with three square axes and no titles. The comparison remains how context, features,
  and shots address their respective error limits. Preserve all 600 source
  rows, root aggregation, marker identities, and uncertainty
  root-percentile ranges; no new data or interpolation. Use the existing Matplotlib renderer
  and inspect both the exported vector PDF and its full-width manuscript
  placement. Legends and panel letters are inside each axis, using the shared
  thin-spine, outward-tick and unified-palette contract. Axis ranges leave
  room for those keys without removing data; no extra title strip or external
  legend row is retained.
- **a:** History error against loss times context, with size-wise root medians
  and capped 10th--90th percentile whiskers. No faint root trajectories or
  translucent bands are drawn; the full root-level data remain in the export.
- **b:** QELM total error against measured feature count, on the fixed
  history-sufficient loss-4, window-6 slice. Medians and 10th--90th root
  percentile whiskers are computed separately for each target size and budget.
- **c:** Finite-shot test error against the actual per-input shot budgets.
  Colors and shapes denote 18, 36, and 72 features; black dashed stars show
  the validation-selected policy with its black 10th--90th percentile whiskers.
  This four-qubit study is explicitly marked as a finite-shot follow-up.
- **Sources:** the validated scaling source, frozen confirmatory suffix table,
  and `finite_shots_eiv_n4_analysis_v6_final_helpers_strict/eiv_root_effects.csv`.
  The original 600-row aggregated plot export is byte-unchanged.

## Figure 3 — Separate five-qubit, longer-context study (formerly Figure 4)

- **Artifact identity:** `../long_context_n5/fig04_long_context_n5.pdf` and its
  PNG/source CSV keep their existing filenames and contents; only the manuscript
  number changes. It is not merged with the archival smaller-system phase map.

- **Layout:** Two square axes at native 7.055-inch width and 2.65-inch
  height, with common logarithmic percentage limits. Inset letters
  and loss labels identify the panels; the in-panel key in a applies to both.
  Expanded upper clearance separates the highest uncertainty whisker from the
  panel letter; removing the repeated key in b leaves its low-error whiskers
  unobscured. Every percentile remains inside the plotted limits.
- **Prespecified comparison:** Five-qubit target, ten-qubit fixed emulator,
  twelve fresh roots, losses 1 and 4, windows 2/4/6/8/10. History is black,
  open-circle and dashed; the 120-feature QELM is reference green and square;
  quadratic delay is reference purple and diamond. The displayed 120-feature
  budget was fixed before fresh outcomes, not selected by test performance.
- **Aggregation:** All 360 root-series observations appear in the source CSV.
  Each median and 10th--90th percentile is computed from all twelve roots in
  its condition. There are no faint trajectories, bands, smoothing, clipping
  or excluded unfavorable windows. These are finite-panel NRMSEs, not the
  theorem's worst-case diameter, and whiskers are not confidence intervals.
- **Interpretation:** At loss 1, the history curve continues down while QELM
  error increases after window 6. The quadratic control remains competitive
  or better. The additional register is explicitly counted; neither a universal
  architecture claim nor matched-resource advantage is inferred.
- **Provenance:** `results_n5_long_context_v1/run/suffix_decomposition.csv`,
  its frozen run audit and verified analysis manifest. The renderer validates
  all upstream hashes and exports `../long_context_n5/fig04_long_context_n5_source.csv`.
  New results remain separate from every earlier cohort. In the preceding
  four-figure ICAART build, this figure appeared on page 7 of twelve pages;
  the next three-figure build must verify actual placement and page count.

## Figure 1 design selection

The original candidates A/B/C and later B1--B5/C1--C5 gallery remain design records. Selected A and canonical Figure 1 share the current landscape layout. The user retained B4; separate c and the subsequent stacked output overlays have been replaced by one memory-access toy timeline in architecture panel a. This supersedes the earlier restoration of state-visibility panel c; those old plots must not be relabeled as QRC/QELM comparisons.

DESIGN_NOTES.md and design_manifest.json describe the current selected source and retained alternatives. Rebuild with pra_extension/code/render_figure1_variants.py. The pending Figure 3 combination is a separate, unapproved proposal and is not implemented here.
