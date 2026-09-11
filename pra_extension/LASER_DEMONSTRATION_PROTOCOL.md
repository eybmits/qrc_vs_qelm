# Measured-input time-series demonstration

Status: separate, prospectively specified demonstration requested by the author.
This does not amend the frozen primary studies, their gates, or the submission.
The existing 12-page ICAART PDF is not changed by this experiment.

## Question

Can a trained reset-window QELM reproduce the output of a standard dissipative
Ising QRC driven by measured laser inputs? Does a same-window intervention
separate missing-history ambiguity from representation/training error?

This is QRC-output emulation, not prediction of future laser intensity and not
a competition between two fitted forecasting models. The QRC output is a fixed
observable, not a fitted target selected to favor either outcome.

## Data and splitting (fixed before simulation)

- Santa Fe competition dataset A continuation: 9,093 measured laser samples.
  Use the univariate chronological record, not a shuffled regression table.
- Original dataset description: <https://www-psych.stanford.edu/~andreas/Time-Series/SantaFe.html>.
  The original host was unavailable. The academic KEEL description confirms the
  measured-laser origin: <https://sci2s.ugr.es/keel/dataset.php?cod=47>.
- Pinned mirror: <https://raw.githubusercontent.com/shubhomoydas/ad_examples/81b9fbe053b7da1454202f3e604e25009ded60f3/datasets/SantaFeCompetition/SF_Acont.dat>.
  SHA-256: `d515c7275fb6c7c56a0ebf537d8be2f87f06f3ad36f8153d759519425c746b77`.
  Original attribution: Udo Huebner, PTB; collected primarily by N. B. Abraham
  and C. O. Weiss. The mirror changes storage to comma-separated values.
- Zero-based, half-open blocks: train [0,4096), validation [4096,5120),
  test [5120,7168). Remaining samples are unused. Reset the target at each
  block boundary and exclude its first 128 samples from fitting/scoring.
- Fit input min/max on the training block only, clip later values to [0,1],
  and quantize both models' inputs to the same 33 levels. Record clipping.
  This is a disclosed finite-resolution transformation of measured data.
- Preserve repetitions: identical laser values are not duplicate records.

## Architecture and fitting

- Four qubits; all-to-all random XX couplings drawn once per seed from [-1,1];
  transverse Z field 0.5; input drive a times the sum of X operators; step 0.5.
  This is the existing repository's input-driven Lindblad/Ising architecture,
  not an invented response curve. Related architecture and input conventions:
  <https://www.nature.com/articles/s41534-023-00682-z>.
- Two local-loss rates: gamma = 4 (fast loss), gamma = 0.2 (slow loss).
  Slow loss is not a claim of indefinitely persistent memory.
- QELM: reset product encoding of the newest four inputs, one per qubit;
  input-independent Ising channel. Share the couplings, field, loss rate,
  evolution duration, observable bank, and four snapshots with its QRC.
  Both feature banks have 48 exact local X/Y/Z expectations. No shot noise.
  Runtime resource checks must pass. Their input interfaces intentionally differ.
- Predeclared target: final-snapshot expectation of Z on qubit 0. The QRC's
  corresponding fixed 48-dimensional readout selects this coordinate; only the
  QELM readout is trained. This is not a downstream forecasting comparison.
- Standardize QELM columns using training rows only; unpenalized intercept.
  Ridge grid: 1e-8, 1e-6, 1e-4, 1e-2, 1, 100. Select on validation MSE only;
  no refit on validation or test. Report test RMSE and RMSE/training-target SD.
- Twelve predeclared coupling seeds 917000 through 917011. Show seed 917000
  (not the best seed) and report all seeds. There is no target/seed/window search.

## Same-window intervention

For each seed and loss rate, start two QRCs in the same zero state. Feed them
different measured prefixes: test-block samples [0,128) and [128,256).
Then give both the same measured continuation, test samples [256,384).
These are deliberately spliced experimental histories, not two natural records
claimed to have identical suffixes. Their last four inputs agree from the fourth
common sample onwards; the fixed QELM must then produce identical predictions.

Keep two QRC trajectories and the trained QELM predictions. The half-gap between
the paired scalar QRC outputs is a witnessed lower bound on the larger error of
any common prediction, not the global D_d/2 and not an RMSE bound. Check this
inequality numerically. Report the gap at window expiry and its maximum over
the first 16 post-expiry samples; do not choose a peak for the displayed segment.

## Predetermined checks and interpretation

Engineering gates: finite data of expected length/hash, disjoint split windows,
train-only transformations, matched resources, identical-window predictions
(absolute discrepancy <= 1e-10), physical density matrices, and the pointwise
pair lower bound (numerical slack <= 1e-10). Run tests before handoff.

Descriptive demonstration checks (not tests of statistical significance):
median held-out normalized error below 10% in the fast-loss case; median paired
maximum half-gap over the first 16 post-expiry samples larger for slow than fast
loss; at least one slow-loss witnessed half-gap > 0.01 in observable units.
Report failures as well as successes. A fit failure alone is not evidence of
missing information. No global impossibility, hardware, universal finite-model,
efficient-training, or advantage claim follows from this demonstration.

## Deliverables

Reproducible code and tests; ignored local data/results; Cobalt-style static
candidate plots of the measured-input test segment and the paired control;
all-seed metrics, source/code hashes and machine-readable checks. No automatic
replacement of Figure 1 and no change to the paper's page count.
