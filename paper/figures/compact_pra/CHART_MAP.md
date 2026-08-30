# Main-paper chart contracts

All main-paper figures use a white background, charcoal text, quiet gray guides,
navy (`#2F5D8C`) for protected/collective behavior, amber (`#D97706`) for
local-loss or representation-limited behavior, teal (`#2E7D5B`) for
dephasing/symmetry breaking, and neutral gray for context. Color is always
paired with labels, marker shape, line style, or hatching.

| Figure | Analytical question | One-sentence takeaway | Form and evidence grain | Non-color distinction | Source and final surface |
|---|---|---|---|---|---|
| 1 | Why can recurrence be exactly non-emulatable from any fixed suffix, and what survives imperfect protection? | A protected, excited, visible mode creates a permanent floor under exact symmetry and a certified finite horizon under bounded symmetry breaking. | Mechanism diagram, exact log trend, observed perturbed gaps, and rigorous generator-norm envelopes; 12-root validation. | Direct labels, solid observed versus dashed certified curves, horizon markers, `EXACT`/`CERTIFIED` tags. | `dark_sector_trajectories.csv`, `dark_sector_leakage.csv`; full-width static PDF/PNG. |
| 2 | Across the frozen grid, when is emulation possible and what resource is limiting? | Increasing accessible history first removes history impossibility; only then can feature budget separate emulated from representation-limited cells. | Ordered categorical heatmap; 144 system cells, each summarized over 12 independent roots. | Cell letters, disagreement hatching, direct resource arrows, compact state key. | `fig04_regime_map_source.csv`; full-width static PDF/PNG. |
| 3 | Does the prospectively frozen dissipation-window product predict the history floor? | History NRMSE collapses approximately exponentially in `x = gamma_T dt d` in both teacher families. | Two faceted log-line panels; 480 rows, 12 root fits per family. | Open root points, faint root fits, amber family medians, charcoal median fit, explicit tolerance line. | `fig07_dissipation_window_scaling_source.csv`; full-width static PDF/PNG. |
| 4 | How do dynamics and declared observables move the history frontier? | Forgetting is useful only while the target remains live, and collective loss can retain observable history hidden by local loss. | Root dot-and-interval comparison, three-point liveness trend, faceted marker/line comparison. | Dissipator color plus observable marker shape; open roots and capped 95% root-bootstrap intervals. | Observable-memory strict analysis and 3,888-row raw table; full-width static PDF/PNG. |
| 5 | How does finite measurement move the useful feature budget? | Larger feature maps become useful only when the shot budget resolves them. | Stacked validation-selection composition, paired root performance curves, root dot-and-interval noise penalty. | Feature-budget colors with direct labels; validation policy in charcoal; open root dots and 95% root-bootstrap intervals. | Strict EIV analysis and 2,304-row raw table; full-width static PDF/PNG. |

Final QA is performed in the compiled two-column REVTeX PDF, not only on the
standalone PNGs.
