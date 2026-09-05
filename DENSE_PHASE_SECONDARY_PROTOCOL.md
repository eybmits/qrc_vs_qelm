# Dense local-loss phase-map secondary replication

Status: **frozen before execution**  
Freeze date: 2026-08-31  
Scientific status: prospectively specified secondary replication, designed
after the original frozen confirmatory experiment.

## Question

Does the history-to-feature crossover seen on the original coarse grid persist
when the local-loss and finite-window axes are sampled more densely with root
blocks that are disjoint from every design and confirmatory root?

## Frozen material

- Teacher sizes: `N_T = {2, 3}`.
- Teacher dissipator: local loss.
- Loss rates:
  `gamma_T = {0, 0.2, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4}`.
- Reset-QELM windows: `d = {1, 2, 3, 4, 5, 6}`.
- Reset-QELM feature budgets: `B = {18, 36, 72}`.
- Twelve fresh root blocks `dense01` through `dense12`; every seed is fixed in
  `configs/canonical_emulation_dense_secondary.yaml` and disjoint from all
  roots in `configs/canonical_emulation.yaml`.
- Train, validation, test, and common-suffix panel sizes, model architectures,
  observables, input alphabet, ridge grid, and NRMSE tolerance are identical to
  the original confirmatory profile.
- The root block remains the statistical unit. Every specified cell is
  reported. No grid point may be removed or added after execution.

## Frozen summaries and display

For each teacher-size, loss-rate, and window cell, the primary continuous
quantity is the median common-suffix history NRMSE across the 12 roots. The
minimum and maximum across roots are retained. For each feature budget, the
modal regime, its agreement count, and the number of `EMULATED` roots are
computed using the existing frozen classifier. A cell badge reports the
smallest tested budget for which at least 10 of 12 roots are `EMULATED`.
Modal labels remain descriptive; a plurality alone is not called a successful
budget. Ties remain `MIXED`; unresolved and representation-limited cells are
not reclassified.

The display is a rectangular discrete heatmap. Cell color shows the median
history NRMSE on one shared logarithmic scale. Badges show the smallest
successful tested feature budget. An exact staircase follows edges between
tested cells on opposite sides of the pre-existing `0.10` history tolerance.
No smoothing, interpolation, or contour fitting is permitted.

## Primary prediction and decision

For each root and teacher size, fit ordinary least squares to

`log(max(history_nrmse, 1e-12)) = intercept + slope * (gamma_T * dt * d)`

using all positive-loss cells. The secondary replication passes its primary
directional prediction when the fitted slope is negative in at least 10 of 12
roots for each teacher size. The dense map must also contain, for each teacher
size, at least one root-median cell above and at least one at or below the
pre-existing `0.10` history tolerance. Completeness, finite-value, decomposition
identity, and seed-disjointness checks are mandatory and fail closed.

The QELM-facing anchor is inherited from the original map rather than chosen
from the dense results: at `gamma_T=4`, `d=6`, and `B=72`, at least 10 of 12
fresh roots must be `EMULATED` for each teacher size. The secondary study passes
only if both the directional history prediction and this anchor check pass.

## Claim boundary

Passing this protocol supports a stable finite-resource crossover on the tested
local-loss QRC families and the tested reset-QELM. It is not part of the
original frozen confirmation and does not establish a universal phase
boundary, arbitrary-dissipator behavior, a contraction constant, efficient
QELM scaling, hardware validity, or quantum advantage.
