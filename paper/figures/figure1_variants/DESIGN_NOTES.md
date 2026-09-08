# Figure 1 design study

## Active revision: two memory timelines

This contract supersedes the historical layout notes below. Question: after
an old input leaves a fixed window, does its effect on the QRC output fade
or persist? Use two stacked line-and-marker small multiples, titled Fading
memory and Persistent memory, with one shared black-filled QRC / green-open
QELM legend. Native static TikZ/PDF is the final surface. Each case has 101
deterministic analytic output pairs on the same known baseline and balanced
old bit; neither is fitted or measured emulator performance. The persistent
case is unchanged; the fading coefficient is 0.25 exp(-t/30), from t=0.
Both ideal window predictors know the bit until t=40, then use the common
baseline. A dashed guide at d=40 marks lost input access, not a training split.

The two frames share x=[-3,103], y=[0.45,1.35] and 4.15-by-3.20-cm geometry.
The bottom frame carries the shared time labels. Existing model colors,
open/filled markers and quiet horizontal/vertical grids remain. Background
washes indicate access: green before d, blue after d, in both cases. The
architecture and the 201-point input plot are unchanged. Remove the old
lower panel b, including its tolerance curve and letters; no general D_d/2
label is attached to a trajectory gap. The canvas is 17.9197 by 9.60 cm.
The output area grows from one square to two short aligned frames while
the figure becomes shorter. The exact worst-case/increasing-window theorem
stays in Section 3. Review the exported figure and both complete PDFs;
preserve all experimental figures/data and the twelve-page maximum.

## Historical design record (superseded where noted above)

## Reference analysis

The supplied PDF `2608.25511v1 (7).pdf`, Figure 1 on page 2, is the visual reference. Its input plot has a fine black polyline, dense filled sample dots, restrained grey grid, a thin complete frame, and a selected interior point with a dashed time guide and a purple horizontal connection to dynamics. Its prediction panel uses black targets and larger hollow green prediction markers. The small differences between successive samples remain visible rather than being replaced by spline smoothing.

The embedded reference artwork uses Helvetica/Helvetica Neue with serif mathematical symbols. Our native TikZ figure uses Helvetica and Computer Modern mathematics, with sans-serif numeric ticks. It follows that hierarchy and palette without copying the reference's network, numerical trajectories, training/test claims, or uncertainty bands.

## Design contract

All three plotted frames use the paper's quiet two-direction grid: #D9D9D9,
0.35-pt lines, 2-pt dash/gap, layered above fills and below data. Input x
guides are every 50 steps; output x guides every 20, retaining the existing
y guides and tick labels. The qualitative panel b uses sparse unnumbered
guides, omitting the horizontal line beside epsilon to prevent a doubled
threshold. No grid is applied to architecture wiring. All data and geometry
remain unchanged, and final manuscript pages are checked at native size.

Question: when can a reset-window model reproduce the QRC input-output map?

Panel a has two architecture lanes and one common output plot. The QRC carries state through a general channel; the reset-QELM receives its entire causal window from a fresh state. The target measures O_1,...,O_m and the QELM measures P_1,...,P_B, including checkpoint features; each applies a compact gold readout triangle without text labels. A brace groups target y_t and prediction widehat y_t into the same plot.

Both observable banks use softly rounded feature cards with a restrained warm
outline, rounded black connector bends, and 1.8-pt rounded gold readout
triangles. Three faint grey lanes run only from each bank's output edge to
the readout, indicating the intermediate observables omitted by the vertical
ellipsis. Their illustrative count does not assert a feature budget or add
quantum wires. The two explicit black endpoint connections, bank positions,
0.38-by-0.50-cm readout vertex spans and all scientific content stay unchanged.

The reset circuit retains its initial-state kets and fixed-gate blocks, but the visible "fresh state" and "fixed gates" labels are omitted. This annotation-only change preserves all positions, fonts, data and the existing figure footprint; verify the native export and page 2 in both manuscript PDFs.

One 4.15-cm-square input frame at [1.045,6.725,5.195,10.875] and one matching output frame at [13.525,6.725,17.675,10.875] keep their previous centers while opening horizontal gutters. The reset lane moves down 0.30 cm to separate the model rows; its window connector follows it. There is no duplicate lower input plot or stacked output comparison. The canvas is 17.9197 by 13.60 cm, including 0.60 cm of added separation above panel b. Native labels remain 11-pt Helvetica, letters 12 pt, and compact key/event labels 10 pt: only axis geometry changes, not typography or data. Input and output endpoints retain horizontal padding and the legend has reserved clearance. There are no large plot titles, decorative shells or external logos.

The reference palette is retained: blue #3B86FF for carried state and the past-influence cue, green #00A14B for the input window/QELM, purple #A93BFF for current-input injection, gold #FFC943 and pale #FFDD92 for observations/readouts, red-orange #FC4D30 for the persistent floor, plus black traces and grey guides. Black filled versus green hollow markers preserve the output comparison without depending only on color.

The lower B4 panel keeps its analytic exponential and scalar-optimum meaning in 4.15-by-4.15-cm axes, matching both square frames in a. Its left edge is 1.045 cm, aligned with the input frame; the b and a letters also align. Fonts, strokes and the round crossing marker retain their native sizes. A Window QELM heading and error-axis formula identify the information limit, while epsilon is explicitly the desired tolerance. The floating gold badge and scalar midpoint inset are omitted. The x-axis label remains centered; no "within tolerance" or "crossing" text is restored. A clear gutter separates it from panel a. Separate panel c remains removed.

## Input provenance and causal access

The region before the green window has a 16% tint of the carried-state blue,
from the frame's left edge to index 140.5; the window has a 24% green tint.
A fine solid green!65!ink line (0.4 pt) marks its onset independently of
color, distinct from the dashed current-time guide. The S_d label uses the
same green fill as its window, avoiding a pale rectangular patch. All
samples, scales and geometry remain unchanged above these backgrounds.
Its existing brace is labeled on two lines:
"past carried in / QRC state rho_(t-1)". The label describes access through
state, not lossless storage; window membership and grey continuation are
unchanged. The output plot's fills remain untouched. Check color and
grayscale separation, curve legibility and label clearance in both PDFs.

The output comparison now uses the shared linear y-range [0.45,1.35],
with ticks 0.6, 0.8, 1.0 and 1.2. Both unchanged 101-point traces appear
one third taller than under [0.3,1.5], with roughly 0.05 units of padding
around their combined extrema. Only the y-axis mapping and time-cut guide
height change: no normalization, interpolation or cropped samples. Preserve
the square frame, time scale, typography, legend and all other panels.
Verify marker clearance and inspect the native export and both final PDFs.

Both dynamics groups move 0.35 cm right to enlarge the input-to-architecture
gutter, without scaling the input plot, gates or fonts. Both observable banks
move 0.15 cm right; compact readouts and the output frame stay fixed. The
violet scalar label feeds a horizontal segment followed by a vertical arrow
down into the center of the QRC top edge. All scalar-connector bends are
right angles, clear of the QRC heading, state brace and quantum wires. This
is a classical control connection; no data or causal access changes.

The display now contains the first 201 Mackey-Glass samples after the same declared 1000-sample transient (seed 1). The original 120 values remain an exact prefix. Current time is t=180, not the rightmost sample. Only its scalar a_t enters the QRC update; older inputs act through the incoming state.

The green window contains exactly indices 141..180 (40 samples) within the single plot. Its bracket supplies those values to the reset encoder without repeating them in a lower plot. The unchanged domains are [-6,206] in time and [0.3,1.3] in amplitude. No values are rescaled and no spline is fitted. Labels identify the current time t and its dashed guide.

Grey samples 181..200 are visual continuation only: they are excluded from both the current-input connection and the complete-window selection. The purple leader starts at the actual ordinate of sample 180 and uses orthogonal segments into the channel. The window bracket feeds only the highlighted interval to reset encoding, not the whole displayed history.

These choices concern illustration legibility only. The illustrative 40-input window is not an added numerical experiment or a change to any frozen experimental window.

## Output comparison provenance

The external "output" heading, expiry arrow and explanatory text are removed.
The same green!24 and blue!16 tints as in the input plot split at t=d inside the
unchanged square frame. Green marks the old input still in the window;
blue marks it outside the window but still influencing the QRC. This is
an input-access cue, not the onset of QRC memory. The caption supplies the
key, and the dashed d guide and filled/open curve markers retain non-color
distinctions. All trace coordinates, axis scales and fonts stay unchanged.

The reference-style legend states "QRC" and "QELM" in one 10-pt
row, in a white rounded box with a black border and quiet down-right shadow.
Its local bounds [0.08,4.02,3.35,4.55] cm place it at the upper left across
the frame's top edge while clearing the unchanged peak. QRC remains the
target itself, as explained in the caption; there is no independent third
truth series. Enlarge green open markers to 1.45-pt inner sep and display
them every second sample. Draw the black dot after the open ring to keep
coincident target points visible. All 101 paired values, both axis domains,
the access backgrounds and the square frame are unchanged. No additional
training/testing labels or uncertainty bands are inferred from the reference.
The ideal, unmeasured status remains explicit in the caption and model text,
even though the legend no longer carries the parenthetical qualifier.

Panel b is headed "Two reservoir-memory cases": red means "old inputs still
affect QRC output" and blue "old-input effect on output fades". The colors do not denote
QRC versus QELM or two individual histories. Each curve is the optimal scalar
missing-history error D_d/2 for one possible reservoir-memory behavior,
defined through the largest compatible-history output gap. It is an information
limit, not a measured training error. Curves, tolerance and geometry are unchanged.
The scalar midpoint inset is removed at the author's request. The half-gap
argument remains in the manuscript. Existing curves, labels, palette and canvas
are unchanged; only the extra interval diagram and its caption reference are removed.

The single output plot is a conditional toy reference. Two histories differ only in an initial bit b=+1 or -1 at time 0 and share their later inputs. A common input-driven baseline g(t) is combined with a retained contribution: QRC output g(t)+0.25b. A 40-input window can read b for t<40; at t=40 the bit leaves the window. Thereafter the balanced conditional-mean window prediction is g(t), also the scalar minimax midpoint of the two possible outputs. Only b=+1 is drawn, so the curves track before the marked event and then differ by 0.25. The unseen negative-bit history is part of the declared toy, not an omitted experimental result.

A normalized independent 101-sample Mackey-Glass shape supplies only the drawing baseline g(t). It is not calculated from the separate 201-sample architectural input plot, and neither output is a simulated reservoir trajectory or fitted QELM prediction. Black filled markers and green open markers distinguish the curves without color alone. The marked event says the bit leaves the window, not that the QRC forgets it. Keep the abrupt loss of access visible rather than smoothing it into a fitted transition.

This illustrates an ideal window reference, not measured QRC/QELM performance or a universal failure instant for every QELM. No training/test partition or uncertainty band is invented. Schematic scope is disclosed in the caption, without a separate in-art tag. Generator, literal tables, formulas and hashes are tested.

## Delivery and verification

Panel b is translated downward by 0.60 cm as one unit; none of its curves,
labels, callout or marker is resized. The native canvas bottom extends to
-0.60 cm, keeping its top at 13.0 cm and its width at 17.9197 cm. The two
left-aligned square frames now have 2.325 cm between them, clearly separating
the architecture/time-series row from the information-limit row. Both final
PDFs must retain native-width typography and clean pagination; text is unchanged.

Selected A and canonical Figure 1 must remain byte-identical. Keep native LaTeX, vector PDF, PNG and deterministic current provenance. The older alternatives and ten-panel gallery remain historical design records.

Current provenance is fig01_information_interface_render_source.json. The historical fig01_information_interface_source.json is bound by the N5 seed freeze and must not change. Numerical evidence and other figure artwork stay unchanged.

Inspect the vector figure, then both complete manuscripts at equal rendering DPI against the supplied reference. Check selected-time causality, exact window identity, marker/frame padding, legend clearance, captions and page flow. Rebuild release hashes after the checks.
