# Direct laser forecasting and controlled history restriction

Author-requested, separately specified follow-up. This protocol is written
before running its forecasting results. It does not amend any frozen emulation
study or replace a manuscript figure automatically. All outcomes are retained.

## Questions

1. Can the matched QRC and parallel reset-window QELM forecast actual future
   measured laser values, rather than emulate a chosen reservoir observable?
2. Does preserving older state improve forecasting within the same QRC
   architecture, compared with resetting and replaying only recent inputs?

The second question is not answered by changing the loss rate alone: changing
loss also changes the feature map. It needs a within-architecture intervention.

## Frozen data and indices

- Reuse the hash-verified 9,093-sample laser record and the exact source identity
  in `laser_demonstration.py`; no additional data, excerpt search or shuffling.
- Chronological blocks: train [0,4096), validation [4096,5120), test [5120,7168).
  Each block starts from the zero state. No prediction target crosses a split.
- Input scaling uses training min/max only, with clipping and the same 33-level
  quantization for both quantum architectures. Record every clipping count.
- Targets are the actual, unquantized future laser measurements. Primary
  horizon: one step; secondary: four steps. Neither horizon is selected on test.
- Predict at indices t = 128,...,block_length-5 in every model and horizon.
  This common scoring set removes washout and excludes all boundary crossings.
  Features see inputs no later than t; the target is raw_laser[t+horizon].
- Standardize each target using that horizon's training targets only. Report
  RMSE in laser units and NRMSE relative to that training-target SD.

## Architectures and readouts

- Twelve new coupling seeds 918000 through 918011, shared across loss rates
  4 and 0.2 and across models. No seed or rate search. Report every cell.
- Reuse the demonstrated four-qubit Ising/Lindblad architecture, field 0.5,
  step duration 0.5, same couplings, local loss, and 48 exact local X/Y/Z
  expectation features from four checkpoints. Resource matching must pass.
- QRC: carry the state across all preceding inputs within a split.
- QELM: reset for each prediction, parallel-encode the latest four inputs,
  one per qubit, and apply the same input-independent channel as the previous
  demonstration. Its input interface intentionally differs from the QRC.
- Train BOTH readouts directly on future measured laser values. Use the same
  train-only feature standardization, unpenalized intercept, ridge grid
  (1e-8,1e-6,1e-4,1e-2,1,100), and validation-MSE selection. No validation/test
  refit. Reservoir/channel parameters are never trained.
- Context baseline: repeat the last AVAILABLE input, decoded from the same
  quantized input representation. Label it quantized persistence; it has no fit.
- Exact expectations only: this follow-up does not test finite-shot forecasting.

## History intervention, not an information-floor estimator

For each QRC, independently reset to its zero state before each scored
prediction and replay its last d inputs, for d = 1,2,4,8,16,32. Use the SAME
input-dependent channels, qubits, observables, 48 feature dimensions and
final-step measurement checkpoints as the full-history QRC. This is a
history-restricted QRC control, not the parallel-encoded QELM.

For each d report two analyses:

1. Refit a readout on these restricted features with the identical training
   and validation procedure. This is the primary forecasting-memory comparison.
2. Apply the already fitted full-QRC readout, including its original feature
   standardization, without refitting. This tests dependence of that PARTICULAR
   fitted predictor on older state; a degradation alone may reflect distribution
   shift and is not an information-theoretic impossibility result.

Replay starts from the last d actual inputs of each NATURAL sequence, not a
spliced counterfactual assigned an invented future. Replay requires d input
steps per prediction, versus one new update in a streaming QRC. Its extra
computation is disclosed; no resource-saving claim follows. The same observable
measurement budget applies at the final replay step only.

## Fixed summaries and interpretation

- Plot the first seed's first 128 scored test targets/predictions, not a good
  seed or hand-picked segment, with both loss settings and identical y scales.
- Retain and summarize all 12 roots per rate and all tested windows/horizons.
  Show medians and 10th-90th root percentiles, not confidence intervals for
  independent time-series datasets.
- Predefined paired contrasts: (QELM NRMSE - QRC NRMSE), and (refitted d=4
  restricted-QRC NRMSE - full-QRC NRMSE), separately for each rate/horizon.
  Positive differences favor the full QRC. Bootstrap the mean paired difference
  over the 12 coupling seeds with 10,000 draws and RNG seed 919991. These 95%
  intervals quantify architecture-seed variability conditional on this one
  chronological record, not population uncertainty over arbitrary time series.
- A reproducibly positive second contrast supports useful retained history
  for this fitted architecture and task. Slower loss alone does not establish
  that result. Report absent, reversed, or mixed effects without retuning.
- No exact unavoidable forecasting error is known from this dataset. Do NOT
  transfer the analytically solvable example's red floor to these plots.
- No claim of uniquely necessary quantum memory, quantum advantage, general
  superiority, universal forecasting ability, or practical resource efficiency.

## Verification and deliverables

Test target indexing and split boundaries, causal features, train-only scaling,
paired coupling identity, physical states, batch replay versus the existing
step-by-step implementation, complete source coverage, and readout separation.
Persist protocol/code/dependency/data hashes before simulation, all results,
readout parameters, and first-seed display indices. Results/data stay ignored.
Deliver a Cobalt-style candidate figure and an evidence-backed short readout;
leave the existing twelve-page manuscript unchanged pending interpretation.
