# Figure 1 design study

## Reference analysis

The supplied PDF `2608.25511v1 (7).pdf`, Figure 1 on page 2, is the visual reference. Its input plot has a fine black polyline, dense filled sample dots, restrained grey grid, a thin complete frame, and a selected interior point with a dashed time guide and a purple horizontal connection to dynamics. Its prediction panel uses black targets and larger hollow green prediction markers. The small differences between successive samples remain visible rather than being replaced by spline smoothing.

The embedded reference artwork uses Helvetica/Helvetica Neue with serif mathematical symbols. Our native TikZ figure uses Helvetica and Computer Modern mathematics, with sans-serif numeric ticks. It follows that hierarchy and palette without copying the reference's network, numerical trajectories, training/test claims, or uncertainty bands.

## Design contract

Question: when can a reset-window model reproduce the QRC input-output map?

Panel a has two architecture lanes and one common output plot. The QRC carries state through a general channel; the reset-QELM receives its entire causal window from a fresh state. Both measure model-specific O_1,...,O_m and apply compact gold readout triangles without text labels. A brace groups both outputs into the same plot.

One 4.40-cm-square input frame at [0.92,6.60,5.32,11.00] and one matching output frame at [13.40,6.60,17.80,11.00] are centered between the unchanged architecture lanes. There is no duplicate lower input plot or stacked output comparison. The canvas stays 17.9197 by 13.0 cm. Native labels are 11-pt Helvetica, letters 12 pt, and compact key/event labels 10 pt. Input and output endpoints retain horizontal padding and the legend has reserved clearance. There are no large plot titles, decorative shells or external logos.

The reference palette is retained: blue #3B86FF for carried state and the past-influence cue, green #00A14B for the input window/QELM, purple #A93BFF for current-input injection, gold #FFC943 and pale #FFDD92 for observations/readouts, red-orange #FC4D30 for the persistent floor, plus black traces and grey guides. Black filled versus green hollow markers preserve the output comparison without depending only on color.

The lower B4 panel keeps its analytic exponential, scalar-optimum meaning, 6.40-by-2.40-cm axes and four-centimeter x translation. A clear gutter separates it from panel a. Separate panel c remains removed.

## Input provenance and causal access

The display now contains the first 201 Mackey-Glass samples after the same declared 1000-sample transient (seed 1). The original 120 values remain an exact prefix. Current time is t=180, not the rightmost sample. Only its scalar a_t enters the QRC update; older inputs act through the incoming state.

The green window contains exactly indices 141..180 (40 samples) within the single plot. Its bracket supplies those values to the reset encoder without repeating them in a lower plot. The unchanged domains are [-6,206] in time and [0.3,1.3] in amplitude. No values are rescaled and no spline is fitted. Labels identify the current time t and its dashed guide.

Grey samples 181..200 are visual continuation only: they are excluded from both the current-input connection and the complete-window selection. The purple leader starts at the actual ordinate of sample 180 and uses a rounded elbow into the channel. The window bracket feeds only the highlighted interval to reset encoding, not the whole displayed history.

These choices concern illustration legibility only. The illustrative 40-input window is not an added numerical experiment or a change to any frozen experimental window.

## Output comparison provenance

The single output plot is a conditional toy reference. Two histories differ only in an initial bit b=+1 or -1 at time 0 and share their later inputs. A common input-driven baseline g(t) is combined with a retained contribution: QRC output g(t)+0.25b. A 40-input window can read b for t<40; at t=40 the bit leaves the window. Thereafter the balanced conditional-mean window prediction is g(t), also the scalar minimax midpoint of the two possible outputs. Only b=+1 is drawn, so the curves track before the marked event and then differ by 0.25. The unseen negative-bit history is part of the declared toy, not an omitted experimental result.

A normalized independent 101-sample Mackey-Glass shape supplies only the drawing baseline g(t). It is not calculated from the separate 201-sample architectural input plot, and neither output is a simulated reservoir trajectory or fitted QELM prediction. Black filled markers and green open markers distinguish the curves without color alone. The marked event says the bit leaves the window, not that the QRC forgets it. Keep the abrupt loss of access visible rather than smoothing it into a fitted transition.

This illustrates an ideal window reference, not measured QRC/QELM performance or a universal failure instant for every QELM. No training/test partition or uncertainty band is invented. Schematic scope is disclosed in the caption, without a separate in-art tag. Generator, literal tables, formulas and hashes are tested.

## Delivery and verification

Selected A and canonical Figure 1 must remain byte-identical. Keep native LaTeX, vector PDF, PNG and deterministic current provenance. The older alternatives and ten-panel gallery remain historical design records.

Current provenance is fig01_information_interface_render_source.json. The historical fig01_information_interface_source.json is bound by the N5 seed freeze and must not change. Numerical evidence and other figure artwork stay unchanged.

Inspect the vector figure, then both complete manuscripts at equal rendering DPI against the supplied reference. Check selected-time causality, exact window identity, marker/frame padding, legend clearance, captions and page flow. Rebuild release hashes after the checks.
