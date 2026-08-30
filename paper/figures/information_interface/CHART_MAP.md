# Main-figure chart map

## Figure 1 — Information interface and protected memory

- **Question:** Why can a reset-window learner fail even with a rich feature map?
- **Takeaway:** One older addressable control changes a protected, measurable
  singlet component; every later allowed symmetry-preserving suffix is shared,
  so the visible window is identical while the recurrent outputs remain apart.
- **Form:** Four-panel vector diagram: fixed trajectory pair, the
  protected–excitable–visible mechanism, three resource limits, and the
  practical decision flow.
- **Palette:** Navy and amber as the two non-neutral roots; teal only for the
  measurement/feasible endpoint; grey for omitted information.
- **Source:** Frozen dark-sector summary and configuration.

## Figure 2 — Minimum information and feature resources

- **Question:** At each dissipation–window point, is history still missing, or
  what is the smallest tested QELM feature budget that reaches the target?
- **Takeaway:** Increasing dissipation or window length first makes the target
  suffix-determined; only then does the smallest budget whose 12-root modal
  class is emulated become relevant.
- **Form:** Two 6-by-4 categorical heatmaps, one per teacher size. Each cell is
  derived from the frozen 12-root modal classifications across all three
  feature budgets. Hatching means that at least one tested budget has
  nonunanimous root labels.
- **Palette:** Grey for history missing, teal for unresolved, and ordered navy
  shades for feasible cells. Hatching records root disagreement.
- **Source:**
  `paper/figures/canonical_emulation/fig04_regime_map_source.csv`.

## Figure 3 — Three interventions, three diagnosed limits

- **Question:** Do window/dissipation, feature budget, and shot count move the
  error source predicted by the theory?
- **Takeaway:** Observable history decays with the dissipation–window product;
  richer features reduce normalized map error only on a history-feasible
  slice; and limited shots can make the richest feature set less useful.
- **Form:** Three panels: median history NRMSE versus `gamma*dt*d`, normalized
  representation MSE versus feature budget, and test NRMSE versus total shots.
- **Palette:** Navy and amber for the two teacher families; navy shades and
  neutral black for finite-shot budget curves. Markers and line styles preserve
  grayscale legibility.
- **Sources:**
  `paper/figures/canonical_emulation/fig07_dissipation_window_scaling_source.csv`,
  `results_canonical_confirmatory_v1/suffix_decomposition.csv`, and the separate
  12-root four-qubit local-loss shot slice at `gamma_T=4,d=6` in
  `pra_extension/outputs/finite_shots_eiv_n4_analysis_v6_final_helpers_strict/eiv_root_effects.csv`.
