# Main-figure chart map

## Current semantic palette: input black, QRC orange, QELM blue

The active ICAART figures use `pra_extension/code/active_paper_palette.py`.
External targets and causal input samples are charcoal #20252B, QRC outputs
and carried state are orange #D97632, and QELM predictions and explicit
windows are blue #3868D9. Figure 1 retains yellow observable/readout accents
and gray future display-only samples. Output filled dots are QRC orange;
open rings are QELM blue. All coordinates, labels, samples, and drawing order
are unchanged. Historical candidate archives and `paper_palette.py` remain
frozen; color descriptions in the earlier revision entries below are superseded.

## Current Figure 1: gradually revealed persistent memory

Remove both internal memory-case labels, retaining the shared QRC/QELM key,
native frames, Cobalt/yellow palette, grids and all architecture/input geometry.
Question: can the lower fixed-window example remain close initially and then
show a growing discrepancy? Preserve its charcoal target and the entire upper
example. In the lower analytic balanced-bit toy, a shared future control
gradually reveals a stored bit: c(t)=0.25 s^2(3-2s), with s=clip((t-32)/64,0,1).
The reveal schedule is independent of window length. For the preserved target
T(t), the known bit-independent baseline is g(t)=T(t)-c(t); the two histories
give g(t)+/-c(t). At d=40 the ideal window prediction loses the bit and becomes
g(t), initially close to T(t), with a persistent gap of 0.25 from t=96 onward.
No fitted or measured data are changed or smoothed. The 101-point output
examples remain schematics, not evidence of training performance or the
uniform D_d limit. Keep both cases identified in the caption, document their
separate baselines, and QA the native PNG and twelve-page ICAART PDF.

## Current Figure 1 readability refinement

Question: can the two fixed-window memory cases and their shared model key be
read at a glance? Keep the native TikZ paired line-and-marker output examples,
their exact 101-point values, common scales and Cobalt/yellow palette. Put gray
case labels inside the frames and one white QRC/QELM legend across their facing
borders. Filled target dots and open prediction circles remain the non-color
distinction. Remove only the redundant input-step and time axis titles; keep
the coordinate ticks. The 201-point left schematic uses a half-size integration
step and the same physical burn-in for a calmer trace, without normalization,
smoothing or changes to any experiment or either output example. Retain the
native footprint and all architecture geometry. Inspect the vector export,
PNG and active twelve-page ICAART PDF. This supersedes earlier requirements
to preserve the input samples and exterior titles/legend positions below.

## Active palette contract: Cobalt, ICAART only

The author selected preview B. Apply three chromatic roots throughout the
three current ICAART figures: cobalt #3868D9, coral #D66A56 and violet #7C5A9F,
with charcoal #20252B, gray #8B9198 and pale supporting tints. The palette
definition is `pra_extension/code/paper_palette.py`; legacy TikZ names are
retained solely to avoid changing the established drawing geometry.

Preserve the native static TikZ/Matplotlib surfaces, all line/marker and
interval displays, exact source samples, scales, labels and frame positions.
Figure 1 uses cobalt for carried-state access, coral for explicit-window
access/QELM predictions, restored yellow feature cards/readout triangles
(#FFC943, with #FFDD92 background blocks) and charcoal target curves.
The author explicitly restores this original yellow accent; every other
Cobalt color, geometry, label and sample remains unchanged.
Figure 2 repeats the selected B preview: 2/3/4-qubit targets use cobalt/violet/
coral; 18/36/72-feature budgets use cobalt/coral/violet, with unchanged shapes.
Figure 3 retains charcoal history, coral QELM and violet quadratic delay;
its feature-budget slice uses coral filled/open markers for the two losses.
No extra data, smoothing or changed uncertainties: retain 201 input samples,
two 101-point analytic output pairs, 984 resource rows and the 360+72 N5 rows.

Only the active ICAART PDF is rebuilt; archived alternatives and the journal
PDF are not active deliverables. Color words in the ICAART caption and model
paragraph must match the new artwork. Update link ink to the same cobalt.
QA checks shared root identities, existing data/geometry tests, native PNGs,
all twelve ICAART pages and byte-identical numerical source files. The entries
below document earlier revisions; their previous color wording is superseded.

## Active Figure 1 contract: two fixed-window memory cases

Supersedes the earlier Figure 1 layout/refinement entries below. Use the
native TikZ line-and-marker renderer: the unchanged 201-point input and
architecture feed two 101-point analytic output examples, Fading memory
and Persistent memory. Same known baseline and balanced old bit; the
persistent coefficient is 0.25, the fading coefficient 0.25 exp(-t/30).
The ideal window prediction matches the displayed +1 history before t=40
and uses the shared baseline afterward. These are schematics, not experiments.
One shared legend maps black filled marks to QRC and green hollow marks to
QELM; cases are separated by titles, not different model palettes. Both
4.15-by-3.20-cm frames use x=[-3,103], y=[0.45,1.35], quiet two-direction
grids and the same dashed expiry guide. Only the lower frame labels time.
The old standalone history-error panel and a/b letters are removed. The
canvas is 17.9197 by 9.60 cm, with unchanged architecture/input geometry.
Section 3, not the curve gaps, carries the exact D_d/2 statement and the
increasing-window criterion. Verify final figure/PDF readability, provenance,
identical experimental assets and at most twelve pages in both formats.

## Shared publication design

### All current figures: quiet two-direction grids

Question: can coordinates be followed equally easily across every plotted
panel? Preserve the existing trend/comparison displays and all source rows:
201 illustrative input samples, 101 illustrative output pairs, the analytic
history schematic, and the existing experimental cohorts. Use native TikZ
for Figure 1 and the shared Matplotlib axis style for Figures 2 and 3.
All nine plotting frames receive #D9D9D9 horizontal and vertical dashed
grid lines, 0.35 pt wide with 2-pt dashes and gaps, above background fills
but below traces, markers and threshold guides. Experimental grids follow
major ticks; retain minor ticks without minor grids. Figure 1 input vertical
guides occur every 50 steps and output guides every 20 steps, with existing
horizontal tick guides. Panel b remains qualitative and unnumbered, using
sparse evenly spaced guides; omit the horizontal guide adjacent to epsilon
to avoid a doubled threshold. Do not grid the architecture wiring. Preserve
all colors, access fills, axes, data, legends, labels and physical footprint.
Validate actual x/y grid artists, data hashes and final pages 2, 5 and 6 in
both manuscript PDFs; no new measurements, inference or fitted curves.

### Figures 2 and 3: matched geometry and whitespace

Question: can the resource and longer-context comparisons be read at the
same visual scale? Keep their existing line-and-marker/percentile displays,
all 600 and 360 original source rows, cohort boundaries, scales, colors and claims.
Figure 3 adds a separate 72-row feature-budget slice from the same N5 cohort.
Use the existing static Matplotlib renderers with one shared layout:
7.055 by 2.40 inches per canvas; every data frame 1.60 inches square;
baseline 0.53 inches from the bottom. Figure 2 uses 0.68-inch frame gaps;
Figure 3 now uses the same 0.68-inch gaps and three-panel grid. Panels a/b
share a y-axis description, avoiding a repeated long label beside b while
retaining its logarithmic ticks. The three-panel row starts at 0.70 inches.
Figure 2's frame gap increases from 0.5644 to 0.68
inches, leaving clear white gutters beyond adjacent y-axis labels.
Both figures share 10-pt axis labels, 9-pt ticks, 8.5-pt inset legends,
10.5-pt inset letters, 1.1-pt median lines, 0.55-pt frames and 0.7-pt
whiskers. Keep model-specific marker shapes and the larger selection star.
Unify legend padding and axis-label offsets; do not crop away outer margins
or rescale either canvas differently in LaTeX. No titles, added experiments
or interpolated data. QA checks artist geometry, label/frame clearance,
unchanged CSV hashes and both complete PDF versions at equal DPI.

### Current Figure 1 revision contract

History-limit explanation: panel b identifies the window QELM explicitly,
puts D_d/2 on the error axis, and names epsilon as the desired error tolerance.
The two colored curves remain alternative reservoir-memory behaviors, not
QRC versus QELM. Direct labels explain relevance through effects on QRC output.
The author requested removal of the scalar midpoint inset. Keep the existing
TikZ/PDF surface, 17.9197-by-13.60-cm canvas, palette, curves, axes and sample
counts unchanged. The figure explains history loss through the retained output
comparison and history-error curves; the half-gap derivation remains in the text.
Remove only the inset and its caption reference, then inspect both manuscript PDFs.

Panel separation: give the architectural input/output comparison (a) and
the window-length information-limit plot (b) a visibly separate reading row.
Translate all of b downward by 0.60 cm, including its letter, axes, labels
and marker. Expand the native canvas to [0,-0.60,17.9197,13.0] cm
(17.9197 by 13.60 cm); keep the paper width and all 4.15-cm-square plot
frames, font sizes, palette, grids and values unchanged. Use native TikZ;
no divider or extra label is needed. The frame-to-frame vertical gap becomes
2.325 cm. Inspect the native export and both full PDFs for panel separation,
unchanged plot scale and clean page flow; no manuscript wording changes.

Memory-case clarity: match the left and right access palette exactly:
blue!16 for carried-past access and green!24 for explicit-window access.
The order follows meaning, not left-to-right position. Shorten the output
key to QRC / QELM, retaining its native reference style; the caption must
still disclose the ideal, unmeasured window prediction. In b, a neutral
"Two reservoir-memory cases" heading groups the red label "old inputs still
affect QRC output" and blue label "old-input effect on output fades". Each
curve shows D_d/2 for a possible reservoir-memory behavior, maximizing
over compatible histories, not one history or a QRC/QELM training result.
Keep all trajectories, axes, source data and mathematical claims unchanged.
QA checks the native figure, both manuscript captions/model paragraphs,
palette identity, legend clearance and unchanged sample hashes.

Observable/readout styling: make both banks closer to the reference without
changing geometry or scientific content. Use 2-pt rounded feature cards and
black connector bends, and 1.8-pt rounded readout triangles within their
existing 0.38-by-0.50-cm vertex spans. Three faint grey downstream lanes
illustrate the intermediate observables omitted by the vertical ellipsis;
their count is not a feature budget. They begin at the bank's output edge,
never at the quantum circuit. Preserve the two black endpoint connections,
all labels, traces, axes and positions. Verify native and manuscript-page
renders, unchanged data, and identical styling in both architecture lanes.

Output access cue: remove the visible "output" heading and the two external
event/explanation labels with their upward arrow. Keep the 4.15-cm-square
frame, legend, all 101 paired samples, scales, ticks and dashed guide at d.
Inside the frame, use green!24 before d and blue!16 after d, split exactly at
local x=4.15*43/106 and drawn behind grid and traces. These matching tints
encode access to the same old input: still inside the window, then absent
from the window but influential through QRC state. They do not imply that
QRC memory starts at d. Keep black-filled/green-open series identification;
the caption explains the background key. Verify data hashes and page 2 of
both PDFs; no separate output title or prose below the plot remains.

Past-state cue: distinguish inputs before the explicit window with a 16%
tint of the existing carried-state blue, and the window with 24% green.
Keep these fills behind the unchanged 201 samples and grid. A fine solid
green!65!ink boundary at index 140.5 adds a non-color window-start cue,
distinct from the dashed current-time guide. The fill runs from the input
frame's left edge to index 140.5, meeting the green window without overlap.
Use native TikZ and retain the 4.15-cm-square input frame; matching tints
also appear in the output plot. Above its existing
brace, use the two-line label "past carried in / QRC state rho_(t-1)" at the
native font size. This associates old inputs with carried state; it does
not assert lossless storage of the whole past. Keep the input/output
coordinates, model geometry and paper footprint unchanged; inspect the
native vector and page 2 of both manuscripts.

Reference-style output key: use one 10-pt row, "QRC" and "QELM",
in a white, rounded black frame with a restrained down-right shadow. Anchor
it at the upper left, local bounds [0.08,4.02,3.35,4.55] cm, slightly
straddling the top frame. Deeper reference-like overlap would hide the
unchanged QRC peak at local y=3.91944 cm. Keep the 4.15-cm-square axes,
101 paired values, scales, access tints and dashed d guide. Use larger
green open markers (1.45-pt inner sep, every second sample); draw each
black target dot after its green ring so agreement remains visible.
The caption still defines QRC as the target and QELM as an ideal window
prediction. No third truth, training/test partition or uncertainty band
is introduced. Inspect the native export and both final manuscript pages.

Output-axis zoom: make the retained-memory comparison easier to see within
the same 4.15-cm-square frame. Use a shared linear y-range [0.45,1.35] instead
of [0.3,1.5], with ticks 0.6, 0.8, 1.0 and 1.2. This enlarges both traces
vertically by one third without modifying either 101-point series. Their
combined extrema are approximately 0.49862 and 1.30, leaving about 0.05
units of padding above and below. Extend the time-cut guide to follow the
new visible trace height. Keep time limits, sample markers, fonts, legends,
architecture spacing and panel b fixed; verify point clearance, unchanged
sample hashes and the final PDF page. This is axis zoom, not data rescaling.

Give the shared input plot more breathing room before both architectures:
move the two dynamics groups 0.35 cm right, keeping their native sizes.
Move both observable banks just 0.15 cm right to preserve internal spacing;
keep the readout triangles, output plot, input plot and panel b fixed.
The violet scalar connector uses horizontal/vertical segments, with a final
downward arrow into the center of the QRC block's top edge. Its route remains
separate from the state brace, quantum wires and feedback loop. Preserve
the selected sample, all data, fonts, strokes and the manuscript footprint.
Verify native geometry, label clearance and page 2 in both PDFs.

Panel b now asks the same question with fewer annotations: can increasing
window length reduce the best possible scalar history error below epsilon?
Remove the words "within tolerance" and "crossing"; retain the tolerance
line, quiet band and crossing marker. Label red and blue at their right
endpoints, outside the axes, and center the x-axis label. The native TikZ
curve equations, data, palette and font sizes stay unchanged. Its local
geometry is mapped to a 4.15-by-4.15-cm square at
[1.045,0.85,5.195,5.00], left-aligned with the input frame in a. The b
letter aligns with a. Only geometry is transformed: text, line widths and
the circular crossing marker retain their native sizes. Direct labels sit
to the right. The scalar optimum now appears on the y-axis, and the compact
main-text midpoint argument explains it; no inset or floating gold box is shown. The x-axis label stays centered below
the square. Verify the native export and page 2 in both PDFs.
The model text explicitly distinguishes advancing time at fixed d in panel a
from increasing d in panel b, with the ideal-toy and finite-resource caveats.

The spacing refinement keeps the same analytical question, takeaway, native
TikZ renderer, palette, raw values and manuscript footprint. The two square
frames become 4.15 cm, with unchanged centers and 11-pt typography, to open
horizontal gutters. The reset lane moves down 0.30 cm; the causal-window
connector follows it. All scientific content remains unchanged; the later
panel-b simplification is specified above.
QA checks both final PDFs at the same rendering scale as the preceding PDFs.

The current reference-driven revision uses one centered 4.15-cm-square input
frame and one 4.15-cm-square output frame. The 201-sample input has
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
both architecture outputs. Panel b sits left-aligned below a; separate c stays
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
- Figure 1: 17.9197 by 13.60 cm (7.055 by 5.354 inches), a landscape
  overview; current Figures 2 and 3: 2.40 inches high with identical
  1.60-inch-square data frames. The archival phase-map remains 2.65 inches high.
- Upright text: 9--11 pt for primary labels and ticks, close to the paper's
  10 pt body text; compact legends and direct curve labels use 8--8.5 pt.
  Helvetica text and Computer Modern mathematics; vector PDF artwork throughout.
- White backgrounds, reference blue/green/red-orange with gold measurement/readout
  accents in Figure 1, panel letters and essential
  object labels only; no plot titles, number pills, or interpolated heatmaps.
- All six main-paper evaluation axes, the archival phase map's three axes,
  and Figure 1's single input-series and single output-comparison axes
  have physically square plotting boxes.
  The user-selected B4 tolerance diagram uses 4.15-by-4.15-cm
  axes aligned with the left input frame in Figure 1's lower strip. Its input
  and output axes are also 4.15-cm squares. Whole multipanel canvases need not be
  square. Current Figure 2 has one row of three square axes and current
  Figure 3 one row of three; the archival phase map retains its three square axes.
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
- **b:** The selected B4 panel retains the unrestricted scalar history optimum D_d/2 and a smooth analytic exponential crossing tolerance epsilon, with a contrasting persistent floor. The Window QELM heading identifies the emulator, the y-axis names D_d/2, and a tolerance key distinguishes epsilon. Direct labels refer explicitly to effects on output. The right-side scalar interval inset is removed; the midpoint argument remains in the text. Curves, axes, crossing and experimental evidence are unchanged.
- **Scope:** No separate c remains. Theorems on reachable/observable memory remain in the body. The conditional toy is not an empirical QELM failure or a universal failure-time guarantee; its green curve is an ideal window reference. The caption states that the curves and crossing are schematic.
- **Geometry and typography:** Canvas 17.9197 by 13.60 cm, with lower y=-0.60; one 4.15-cm-square input plot at [1.045,6.725,5.195,10.875], one 4.15-cm-square output plot at [13.525,6.725,17.675,10.875], and 4.15-cm-square history axes at [1.045,0.25,5.195,4.40]. Native Helvetica 11-pt axes/architecture, 12-pt letters and deliberately smaller 10-pt key/inset labels. No trained-readout words. See the current revision contract above for palette and final-context QA.
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
  keep their filenames and unchanged source CSV; artwork uses the shared layout above.

- **Layout contract:** Three aligned line-and-marker panels in one horizontal
  row at 7.055 by 2.40 inches, with three 1.60-inch-square axes and no titles. The comparison remains how context, features,
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
  PNG/source CSV keep their filenames and unchanged source CSV; artwork uses
  the shared layout above. It is not merged with the archival smaller-system phase map.

- **Layout:** Three 1.60-inch-square axes at native 7.055-inch width and 2.40-inch
  height, matching Figure 2. Panels a/b retain common logarithmic percentage
  limits and share the y-axis description in a; c uses an explicitly stated
  linear percent scale for QELM total error. Inset letters and condition
  labels identify the panels; the in-panel model key in a applies to a/b.
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
- **Panel-c contract:** At fixed d=10, how does QELM total common-suffix error
  depend on the measured feature budget? Use a grouped dot-and-interval
  comparison of the three discrete tested budgets B=30,60,120 at both losses,
  not an interpolated trend or a selected-best-budget curve. Export all
  72 root observations separately; each point is the median over the same
  twelve roots, with 10th--90th percentile whiskers. The register remains
  ten qubits. All QELM points use the approved green, with filled squares
  for loss 1 and open circles for loss 4, slightly dodged within each budget
  category. There are no faint root paths, connecting curves, bands or new
  colors. Use a zero-based linear percent axis (0--40% for the saved data),
  the existing 10% guide and an inset d=10 label and two-entry key.
  The existing data show a large reduction from 30 to 60 features and little
  further reduction at 120. This describes feature-budget dependence of
  the realized emulator error, not isolated approximation error. No new
  simulations, training, seed choices or changes to frozen numerical files.
- **Interpretation:** At loss 1, the history curve continues down while QELM
  error increases after window 6. The quadratic control remains competitive
  or better. The additional register is explicitly counted; neither a universal
  architecture claim nor matched-resource advantage is inferred.
- **Provenance:** `results_n5_long_context_v1/run/suffix_decomposition.csv`,
  its frozen run audit and verified analysis manifest. The renderer validates
  all upstream hashes and exports `../long_context_n5/fig04_long_context_n5_source.csv`.
  Panel c is exported as `../long_context_n5/fig04_long_context_n5_features_source.csv`,
  retaining budget, snapshot count, root, errors, normalization and freeze/source hashes.
  New results remain separate from every earlier cohort. In the preceding
  four-figure ICAART build, this figure appeared on page 7 of twelve pages;
  the next three-figure build must verify actual placement and page count.

## Figure 1 design selection

The original candidates A/B/C and later B1--B5/C1--C5 gallery remain design records. Selected A and canonical Figure 1 share the current landscape layout. The user retained B4; separate c and the subsequent stacked output overlays have been replaced by one memory-access toy timeline in architecture panel a. This supersedes the earlier restoration of state-visibility panel c; those old plots must not be relabeled as QRC/QELM comparisons.

DESIGN_NOTES.md and design_manifest.json describe the current selected source and retained alternatives. Rebuild with pra_extension/code/render_figure1_variants.py. Figure 3's approved budget panel uses only the separate N5 cohort; archival smaller-system curves are not merged into it.
# Notation-only consistency pass

Keep all plot families, values, scales, colors, 4.15-cm frames and geometry.
Figure 1 uses target output y_t and emulator prediction widehat y_t; the
removed midpoint inset used endpoints y_- and y_+. Its target bank is O_1,...,O_m and its
QELM bank P_1,...,P_B, matching the model section and distinguishing output
dimension from measured-feature count. The input stays a_t, the window S_d,
and the information limit D_d/2, with epsilon only the desired tolerance.
Figures 2 and 3 retain gamma_T, d, B, Q and their existing data. This is a
notation correction in the native TikZ/PDF surface, not a chart redesign.
Verify both full-paper renderings and preserve the sample/source hashes.

## Feature-density update: displayed Figure 2

The 1x3 resource figure now uses blue open circles, red diamonds and teal
squares for 2/3/4-qubit QRCs in a/b. Frame sizes, gutters and grids remain.
Panel a adds independent archived N4 data at gamma={.2,.5,1,2,4},
d={1,2,4,6}; 504 size/root/scaled-context rows retain cohort/source hashes.
Panel b uses 216 fresh follow-up QELM rows: twelve roots per size and budgets
18/36/54/72/108/144, fixed horizon and uniformly spaced checkpoints.
It neither splices old/new points nor pools cohorts. All points and root-spread
whiskers are retained without smoothing. Panel c's 192 source rows are unchanged.
The combined 912-row source includes raw MSE, normalization and fit metadata.
Caption/methods distinguish cohorts. See pra_extension/FEATURE_DENSITY_PROTOCOL.md
and FEATURE_DENSITY_RESULTS.md. Original-cohort numerical statements remain
in the text but no longer cite the new panel as their direct source.

## Display-only spacing and feature focus

Keep the established 1x3 size, fonts, categorical colors, markers and grids.
Move a's legend to the unoccupied upper-right area and reserve enough vertical
headroom in c for its four-entry legend. Panel b shows the first four tested
budgets (18, 36, 54, 72) on an ordinary linear axis, increasing the physical
18-to-36 separation without stretching one interval selectively. The complete
six-budget results and all 912 exported source rows remain unchanged. Both
captions disclose that 108 and 144 are retained in the full results, not shown
in this focused view. No plateau or equivalence claim is inferred from the
display crop. Check legend/data clearance on the rendered axes and inspect
both rebuilt PDFs while preserving the twelve-page limit.

## Actual intermediate-budget extension (current)

The author clarified that 18-to-36 needed actual additional experimental
settings, not axis spacing. Keep the approved display style and add B=24/30
from the locally locked, exploratory partial-bank study on the same twelve
feature-density roots per size. The original 18/36 rows remain authoritative;
72 new endpoint controls reproduce them and are not appended as duplicates.
All reconstructed inputs, targets and full feature arrays match the parent.
Only 72 new intermediate rows join the source: 984 total, with 288 feature
rows and unchanged 504 history / 192 shot rows. Each feature row retains its
source hash and records its original or exploratory stage. Show
18/24/30/36/54/72 on a linear axis; 108/144 remain in the full source.
The slightly smaller 8-pt b-axis tick labels retain a checked clearance margin.
Do not smooth, interpolate, or interpret the chosen banks as all possible
feature subsets. The historical nine-design gallery retains its original
six-budget comparison and is not the current numerical figure.

## Shot-panel target annotation

Keep the four-series static line-and-marker shot comparison and its 192 source
rows unchanged: three feature budgets (18/36/72) plus validation selection,
each at four measured shot budgets. The question is measurement precision,
not additional exact-expectation feature capacity. Do not add exploratory
24/30 exact-expectation fits to this finite-shot comparison or start new runs.
Move the target label from Q=340 to Q=300 and below its horizontal 10% line,
because a horizontal move alone still intersects the selected-policy whisker.
Preserve the label's 9-point type and give it a small bottom margin by lowering
only c's y-axis bound from 0.065 to 0.055. Retain all colors, marker identities,
data, legend, frame size and other panels. Verify label/line/whisker clearance,
the unchanged 984-row source hash, both rendered PDFs and twelve-page limits.

## History-panel legend clearance

For the existing static line-and-whisker history panel a, retain all 504
root-level rows, logarithmic y scale, colors, markers, typography and square
frame. The user requests a little headroom and a higher legend so the 10%
reference line remains unobstructed. Increase only a's upper y bound from
2.5 to 3.2 and anchor its upper-right legend at axes (1, 1.01), keeping the
legend inside the frame. Require a gap of at least 3% of the frame height
between the legend bottom and the reference line. All 984 source rows,
panels b/c and manuscript text remain unchanged. Inspect the figure and
both rebuilt twelve-page PDFs; do not add experiments or change data.
