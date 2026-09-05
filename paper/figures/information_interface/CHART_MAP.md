# Main-figure chart map

## Shared publication design

- Native width: 7.055 inches, included at the manuscript text width.
- Figure 1: 4.4 inches high; Figure 2: 6.1 inches; Figure 3: 3.5 inches.
- Upright text: 9--11 pt, close to the paper's 10 pt body text. Helvetica
  text and Computer Modern mathematics; vector PDF artwork throughout.
- White backgrounds, restrained blue/teal/orange, directly meaningful
  panel headings, no number pills, no interpolated heatmaps.
- Quantitative error axes show normalized RMSE as percentages: 10% is 0.10,
  not a change in the estimand or a percentage-of-target error.
- Reference: the user-supplied *Finite-shot access laws for quantum reservoirs*
  PDF. Its spacious top-wide / supporting-panels layout informs the design;
  its scientific claims and data are not incorporated.

## Figure 1 — When finite context can replace recurrent quantum memory

- **a:** One stream, two temporal architectures: the QRC carries quantum state
  forward, while the reset-QELM encodes the explicit last-d input window from
  the same reset state and applies a trained linear readout. The question is
  agreement of declared outputs, not reconstruction of internal states.
- **b:** Only state differences that past inputs can create and present or
  future declared outputs can reveal determine the emulation boundary.
  Unreachable and permanently invisible directions need not fade.
- **c:** Uniform fading of this relevant memory permits arbitrary-accuracy
  emulation with growing context and a growing universal reset-QELM family.
  Persistent visible memory instead gives a nonzero error floor at every
  finite window. Both largest-compatible-output-gap curves are schematic.
  The scalar midpoint proof remains in the manuscript, not in the overview.
- **Source:** Editable `fig01_information_interface.tex`, compiled using
  pdflatex without shell escape. Source JSON retains theorem assumptions,
  schematic status, and hashes for the TikZ and validated dark-sector source.
- **Palette:** blue for the recurrent QRC and relevant-memory criterion, teal
  for explicit context and the reset-QELM, orange for persistent visible memory.

## Figure 2 — History error first, feature adequacy second

- **Layout contract:** Keep the existing six-panel, 7.055-by-6.1-inch static
  Matplotlib figure and all 198 measured cells. Remove the two large narrative
  row headings from the artwork; the manuscript caption supplies their
  interpretation. Retain compact QRC-size labels, panel letters, axes, the
  success-rule annotation, and both legends. Preserve palette, marker shapes,
  scales, and provenance; verify the exported PDF and manuscript placement.
- **a--c:** Six window-length curves per target-QRC size, covering every
  measured loss rate. The 198 markers are root-median history errors.
  Lines connect measured points; there is no fitted or interpolated surface.
  The horizontal dashed line marks the registered 0.10 target.
- **d--f:** Discrete feature-success maps for the six tested rates above one.
  Circle: 18 features; square: 36; diamond: 72. Each is the smallest tested
  feature count meeting the total-error target in at least 10 of 12 roots,
  after the history target is also met in at least 10 roots. Orange crosses
  show sufficient history but no robustly successful tested budget. Pale dots
  show unmet history. Lower rates are omitted from this row because every
  tested window there fails the history rule; all are plotted in a--c.
- The lower rows are categorical tested rates, not an equally spaced physical
  axis. The upper curves use the actual numerical loss coordinates.
- A median below the target is distinct from meeting it in 10 of 12 roots.
  Both this distinction and the nonmonotonic feature success at loss 2,
  windows 5 and 6 are retained; no monotonic success region is drawn.
- **Cohorts:** N=2 and N=3 are the earlier frozen dense cohort; N=4 is the
  separately frozen prospective follow-up. They are never pooled.
- **Sources:** `results_dense_phase_secondary_v1_analysis/dense_phase_cells.csv`
  and `results_dense_phase_n4_v1_analysis/dense_n4_phase_cells.csv`, validated
  against their raw suffix tables and provenance hashes. The complete
  198-row source export is preserved, representing 2,376 root-level history
  cells and 7,128 root-budget QELM cells.

## Figure 3 — Three resource interventions

- **Layout contract:** Three aligned line-and-marker panels in one horizontal
  row at 7.055 by 3.5 inches. The comparison remains how context, features,
  and shots address their respective error limits. Preserve all 600 source
  rows, root aggregation, scales, colors, marker identities, and uncertainty
  bands; no new data or interpolation. Use the existing Matplotlib renderer
  and inspect both the exported vector PDF and its full-width manuscript
  placement. Move legends below the axes without reducing the 9--10.5 pt text.
- **a:** History error against loss times context, with size-wise root medians,
  faint individual root traces, and 10th--90th percentile bands.
- **b:** QELM total error against measured feature count, on the fixed
  history-sufficient loss-4, window-6 slice. Root traces and medians are shown.
- **c:** Finite-shot test error against the actual per-input shot budgets.
  Colors and shapes denote 18, 36, and 72 features; black dashed stars show
  the validation-selected policy with its 10th--90th percentile band.
  This four-qubit study is explicitly marked as a finite-shot follow-up.
- **Sources:** the validated scaling source, frozen confirmatory suffix table,
  and `finite_shots_eiv_n4_analysis_v6_final_helpers_strict/eiv_root_effects.csv`.
  The original 600-row aggregated plot export is unchanged.
