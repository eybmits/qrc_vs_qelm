# Variant A: aligned quantum-processing lanes

Two full-width lanes follow the reference paper's process grammar: sampled input, purple injection, neutral dynamics, gold measured quantities, then output. The upper lane shows the current scalar entering a general QRC channel, depicted with a schematic node network. Its blue state-carry loop leaves the channel before measurement. The lower lane supplies the complete recent window to explicit encoding, then fixed gates, measured features, and the only trained readout. It begins from fresh zero states. Gold boxes display measured quantities directly; no prediction series is invented.

The bottom strip contains a square output-gap schematic and two alternative input-written differences: visible under some continuation, or invisible under every continuation. Continuations include the empty one, so visibility now is included. These are alternatives, not a physical split of one state difference.

Reference colors are blue #3B86FF, green #00A14B, red-orange #FC4D30, gold #FFC943, purple #A93BFF, black, neutral grays, and pale gold #FFDD92. The 17.9197-by-10.6-cm canvas retains Helvetica 11-pt labels, 12-pt letters, and Computer Modern mathematics. Both sample plots and the gap plot are 2.60 cm square. Essential object labels replace prose titles and external frames.

Both sample plots read the same literal 120-value Mackey--Glass table; the lower plot selects exactly indices 100--119 with unchanged amplitude. These are the first 120 samples after the repository generator's 1,000-sample transient, seed 1, without smoothing or rescaling. This illustrative input is neither a measured trace nor the independent 33-symbol evaluation stream. No experimental prediction is added. The upper plot shares the lower "input step k" label; both retain numeric time ticks and amplitude labels.

The network and circuit are schematic: no specific qubit count, gate count, Ising topology, unitary-only QRC, or measurement-backaction claim is implied. Captions now need green rather than teal, explicit encoding E(S_d) and fixed gates rather than the old U(S_d) block, and gold quantity boxes rather than meter banks. Preserve the general-channel, declared-output, and universal-family scope.

Canonical and selected native sources remain byte-identical. Rebuild with the repository's render_figure1_variants.py. Both introductions declare the full-width figure early to target page 2; verify actual placement and readability in compiled PDFs.
