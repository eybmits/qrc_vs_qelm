# Four-qubit dense phase-map extrapolation

Status: **prospective; prepared for a material-hash freeze before execution**  
Scientific status: independent teacher-size extrapolation designed after the
`N_T={2,3}` dense secondary map was opened.

## Question

Does the measured history-to-feature crossover persist for a four-qubit QRC
teacher when the reset-QELM architecture, input distribution, loss grid,
window grid, feature budgets, data splits, and decision rules are held fixed?

## Frozen material

- Teacher size: `N_T=4`.
- Teacher dissipator: local loss.
- Loss rates:
  `gamma_T={0,0.2,0.5,0.75,1,1.25,1.5,2,2.5,3,4}`.
- Reset-QELM: the same six-qubit one-lag-per-qubit parallel architecture.
- Windows: `d={1,2,3,4,5,6}`.
- Feature budgets: `B={18,36,72}`.
- Twelve new root blocks `dense_n4_01` through `dense_n4_12`. Every teacher,
  student, input, suffix, prefix, and classical-feature seed is fixed in
  `configs/canonical_emulation_dense_n4.yaml`. Before execution, a structured
  registry checks those seeds against seed-valued fields in existing YAML,
  JSON, and CSV configuration, provenance, and released-result files. The
  registry sources and their hashes are included in the freeze record.
- Train, validation, test, and common-suffix panel sizes; observable panels;
  ridge grid; input alphabet; and NRMSE tolerance are identical to the opened
  dense `N_T={2,3}` study.
- The root block is the statistical unit. Every specified cell is reported.
  No grid point may be added or removed after execution.

The already opened `N_T={2,3}` roots are not reused. Their results determine
the question and fixed design, but never enter training, model selection, or
the four-qubit decision statistics.

The frozen Cartesian grid contains exactly 132 teacher systems, all of which
must be live; 7,920 rows in each of the stream and suffix tables; 66 aggregate
loss-window cells; 792 root-level history cells; 2,376 root-budget QELM cells;
and 12 scaling fits, each using the 60 positive-loss window cells. Any count
drift fails closed.

## Frozen summaries and display

For every loss-window cell, report the median, minimum, and maximum common-
suffix history NRMSE across the 12 roots. A cell is history-sufficient only
when at least 10 of 12 roots have history NRMSE at or below `0.10`. For each
feature budget, report the root-level regimes and the number of `EMULATED`
roots. A badge gives the smallest tested budget with at least 10 `EMULATED`
roots. Ties remain mixed and all root counts remain in the source table.

The four-qubit panel uses the same discrete heatmap, logarithmic color scale,
badge semantics, and exact cell-edge boundary as the existing panels. No
smoothing, interpolation, fitted contour, or post-hoc cell selection is
permitted.

## Prospective checks

All checks below are fixed before any four-qubit result is opened.

1. **Integrity.** The result tables form the complete frozen Cartesian grid,
   all decomposition identities hold to `1e-10`, all target cells are live,
   and every embedded artifact hash matches.
2. **Fading direction.** For each root, fit ordinary least squares to

   `log(max(history_nrmse,1e-12)) = intercept + slope*(gamma_T*dt*d)`

   over all 60 positive-loss cells. At least 10 of the same 12 roots must
   jointly have a negative slope and `R^2 >= 0.85`.
3. **Measured frontier.** Define `d*(gamma_T)` as the smallest tested window
   that is history-sufficient, and set it to `7` when no tested window is
   sufficient. The sequence must be non-increasing with loss rate, must show
   at least one strict decrease, and at least four loss rates must have a
   finite threshold within `d<=6`.
4. **QELM anchor.** At the inherited point `gamma_T=4`, `d=6`, and `B=72`, at
   least 10 of 12 roots must be `EMULATED`.

The extrapolation passes only if all four checks pass. A failed check remains
a reportable result and may not be repaired by changing the grid, tolerance,
feature budgets, roots, or success counts.

## Claim boundary

Passing supports teacher-size stability of the measured finite-resource
crossover from two and three to four teacher qubits for the tested local-loss
family and fixed six-qubit reset-QELM. It does not establish an asymptotic size
law, a universal phase boundary, efficient QELM scaling, hardware validity, or
quantum advantage.
