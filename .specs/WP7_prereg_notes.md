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

## WP7 freeze checklist (verify before tagging stage1-frozen)

- **poly_delay control not starved** (WP4 adversarial 4/5): at every frozen operating
  point verify the readout width `D` is ≥ the full degree-≤2 count `2d + C(d,2)`, so the
  `poly_delay` classical control never truncates cross terms (which would weaken the
  G1c kill-rule baseline). Holds automatically under `d ≤ n_qubits` + default weight-≤2
  3-axis observables, but re-verify if the observable set is ever reduced.
- **T_evolve frozen grid inside `[t_lo, t_hi]` for both models** (see rule 4 below).

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
   `eff_rank`, the multiplex `gain`, and (as a diagnostic only) `eff_rank_frac` for
   BOTH models per cell. Declared-D parity is asserted via `assert_bundles_matched`;
   effective-D difference is reported and discussed. NOTE (WP3 review): `eff_rank_frac`
   is compressed to ~0.1-0.2 for both models at realistic D and is NOT a fairness
   discriminator — lean the discussion on multiplex `gain` and per-cell capacity.

5. **SCOPE DECISION: V=1 is the primary AND ONLY matched-D comparison; drop matched
   V>1** (WP3-fix dual evidence: Codex calibration + code-review 60-config sweep +
   physics). QRC virtual-node multiplexing (substep states within one `dt`) is strongly
   correlated → low multiplex gain → declared D=V·n_obs overstates QRC's effective
   dimension. QELM trajectory snapshots add more real rank (gain ~1.4 at N=5). So a
   matched-D comparison at V>1 pits hollow-QRC-D against live-QELM-D — the exact
   unfairness `assert_features_live` exists to catch. Resolution: run the phase diagram
   at V=1 (clean, unambiguous). The multiplexing asymmetry (multiplexing helps QELM more
   than QRC at these scales) becomes a REPORTED discussion point, not a matched axis. If
   QRC memory still wins regions at V=1, that conclusion is strengthened, not weakened,
   by QELM having had multiplexing available. Pre-register V=1 as the comparison; any
   V>1 result is exploratory and explicitly caveated (not matched-D).

4. **`T_evolve` frozen grid must sit in the recommended window** (WP3 adversarial
   concern 3). `recommend_T_evolve` is advisory; nothing in code forces the frozen grid
   into `[t_lo, t_hi]`. WP7 FREEZE CHECKLIST: verify the frozen `T_evolve` grid sits
   inside `[t_lo, t_hi]` for BOTH QRC and QELM at every operating point — not merely
   that it clears `assert_features_live`. A grid point that clears the guard but sits at
   the noisy low edge for one model and mid-window for the other is an unfair operating
   point.
