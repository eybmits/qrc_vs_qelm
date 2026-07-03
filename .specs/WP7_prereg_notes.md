# WP7 pre-registration accumulator (design-stage notes, NOT yet frozen)

Running list of commitments that must land in PROTOCOL.md / PREREGISTRATION.md at the
WP7 freeze. Accumulated from reviews across WPs so nothing is lost. Each entry: the
commitment + which review raised it.

## Acknowledged, intentional, UNMATCHABLE asymmetries (must be pre-registered as such)

1. **Input-injection mechanism** (WP2 adversarial review). QRC couples the input via an
   active Hamiltonian drive term (`h_in · transverse_drive`) throughout its recurrent
   evolution; QELM injects the input ONLY via state preparation (`encode_window`) and
   then evolves under a fixed, input-independent channel. This is a *structural*
   difference, not a tunable one — memorylessness REQUIRES QELM to have no dynamic
   drive. `assert_matched` cannot and does not equalize it. Pre-register it explicitly
   as an intended asymmetry so a reader does not mistake `assert_matched` for full
   parity certification. Frame in the paper as: "the comparison isolates "recurrent
   dynamic encoding + intrinsic memory" (QRC) vs "static state-prep encoding + classical
   delay embedding" (QELM), under matched qubits / observables / readout dim / shot
   budget / ridge pipeline."

## Frozen methodological rules (operationalize at WP7, informed by the design pilot)

2. **`T_evolve` live-regime rule** (WP2 adversarial + WP3 guard). QELM's frozen
   `T_evolve` grid must sit in the dynamically live window returned by
   `matching.recommend_T_evolve` (γ>0: ≈ mixing_time; γ=0: ≈ 2π/spectral-bandwidth), so
   the V snapshots sample distinct input-dependent states rather than a collapsed
   steady state or V identical copies. The design pilot MEASURES this window per
   configuration; the frozen grid is chosen from it. `assert_features_live` enforces it
   at runtime (dead-column + effective-rank-fraction floors). Pre-register the floors
   (`min_col_var`, `min_rank_frac`) and the recommended-window rule.

3. **Effective-dimension reporting** (WP2 adversarial). Every run manifest records
   `eff_rank` and `eff_rank_frac` for BOTH models per cell, so any residual difference
   in *effective* (not just declared) readout dimension is transparent in the results,
   not hidden. Declared-D parity is asserted; effective-D difference is reported and
   discussed.
