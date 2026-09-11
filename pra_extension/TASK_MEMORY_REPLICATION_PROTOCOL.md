# Task-memory replication and Figure 3 integration

Frozen before the fresh simulations. This author-requested work package is
separate from the original confirmatory emulation protocol. It prospectively
replicates two already explored forecasting studies, without changing their
settings to obtain a desired outcome. It does not amend PROTOCOL.md or
PREREGISTRATION.md.

## Fixed replication size and settings

- NARMA: 24 new roots, 921000--921023. All NARMA_MEMORY_PROTOCOL.md settings
  unchanged except root list and bootstrap seed (939991). Independent input
  streams and couplings per root; four candidate loss rates selected by
  validation only, separately for QRC and QELM; all candidates retained.
- Laser: 24 new coupling roots, 922000--922023. All
  LASER_FORECAST_PROTOCOL.md settings unchanged except root list and bootstrap
  seed (939992). Rates 4 and .2, horizons 1 and 4, all six replay windows,
  frozen-readout controls and persistence retained. Same fixed measured record
  and same train/validation/test splits. This is independent coupling sampling,
  not new measured data or a new unseen test record.
- No optional stopping, seed exclusion, smoothing of observed trajectories,
  new architecture search, or result-dependent change of displayed interval.
  Existing 12-root results stay separate; never pool them invisibly into the
  fresh 24-root replication.

## Prospective direction checks

Four primary paired effects: QELM(4)-QRC and refitted replay(4)-QRC for (i) NARMA
and (ii) fast-loss next-step laser prediction. Each direction check passes only
if at least 20 of 24 roots have positive differences and the two-sided 95%
paired-bootstrap mean interval is entirely positive. These are replication
criteria for previously observed directions, not the original paper's 10%
emulation gate and not a universal superiority claim. Report each effect's
interval separately; no simultaneous-coverage claim. Do not describe any
method as robustly better if its checks fail.

The fast-loss h=1 laser display choice was informed by the earlier study.
Retain the slow-loss h=1 negative control in the main figure and h=4 comparisons
in the appendix and source ledger. NARMA's reduced input range is [0,.2]. No
model receives past NARMA target values. Exact expectations throughout.

## Three-panel figure, fixed display data

a: measured laser input and actual future-value predictions, showing both
fast and slow loss. First fresh coupling root 922000, first 96 scored rows.
b: iid driving input and actual NARMA10 target/predictions. First fresh root
921000, first 96 scored rows. This is synthetic task data, not measured data.
c: refitted reset/replay history controls, with the same QRC circuit and
features but increasing accessible input history. Show both laser rates and
NARMA, QRC references and the four-input QELM comparison. Retain the NARMA
classical controls in the figure or adjacent numerical text. All curves use
all 24 fresh roots; display medians and 10th--90th percentile variability.

Separate axes where the units, targets or intervals differ. Do not average
NARMA raw trajectories across roots; their input and target sequences differ.
Do not score a pointwise median trace as if it were an individual model. The
figure contains actual stored predictions with no smoothing/interpolation
beyond straight connections of sampled points. Curve aggregation is only over
rootwise error scores at identical declared window settings. Explicitly note
that longer replay has greater evolution cost despite matched qubits/features.

The exact-sharpness construction is retained in text and its existing appendix
derivation; the old Figure3c curve can be replaced without deleting results or
historical assets. The theoretical result still concerns QRC emulation, while
the new figure is a separate practical task-level check. No claim that NARMA
requires a QRC or fails for every finite window is permitted.

## Validation and integration

Use new wrappers rather than changing any hashed original runner. Hash the
wrapper, this protocol, original source summary identities and dependency code
before running. Verify all fresh stored test scores, causal target alignment,
validation-only selection and independent RNG streams. Sample saved-readout
reconstruction at fixed rows. Check the complete cohort and all four direction
criteria before writing result claims. Use a read-only independent review.

Replace current Figure3 and the corresponding methods/result paragraphs;
preserve all mathematical statements, exact-reference results and the frozen
emulation comparisons. Keep ICAART formatting and the 12-page maximum. Render
and inspect every page, check figure provenance and clean-extraction rebuild,
and run the full pytest suite. No commit or push is requested by this package.
