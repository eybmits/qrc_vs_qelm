# Direct NARMA10 target and useful-history follow-up

Specified before executing this follow-up. This is a separate exploratory
extension, not an amendment to PROTOCOL.md or PREREGISTRATION.md, and is not
automatically part of the ICAART manuscript.

## Question

Can a recurrent QRC predict the actual NARMA10 target better than a four-input
parallel reset-QELM, and does restricting that same QRC's history remove any
benefit? No outcome is required: report negative comparisons and every seed.

## Task and causal alignment

Generate three independent iid U[0,1] driving streams per root with NumPy
SeedSequence([root, 7301]). Scale the input by 0.2 inside the NARMA10 recursion:

    y[t+1] = .3*y[t] + .05*y[t]*sum(y[t-9:t+1])
             + 1.5*(.2*u[t-9])*(.2*u[t]) + .1.

This is the reduced-range iid NARMA setting used by Fujii and Nakajima,
Phys. Rev. Applied 8, 024030 (2017), Eq. (18) and Sec. IV D:
https://doi.org/10.1103/PhysRevApplied.8.024030 . NARMA is a synthetic benchmark,
not a measured real-world signal. Both models receive only u through time t;
neither receives past y values, future u values, or the other model's output.
The true target is unquantized y[t+1]. The shared hardware input is rounded
to 33 evenly spaced levels on [0,1], solely for cached exact simulation.
No clipping of divergent targets is permitted.

Train/validation/test stream lengths are 4096/1024/2048, including 128 discarded
warm-up inputs each. All models use t=128,...,length-2, hence 3967/895/1919 rows.
Models and the target recursion reset at each independent split. Target scaling
and feature standardization use training rows only. Inputs have a known range;
no full-data fitted preprocessing is used.

## Matched models and predeclared selection

Roots 920000 through 920011 give independent coupling and driving-input draws.
Use the existing four-qubit Ising/Lindblad architecture from the laser study,
dt=.5, field=.5, coupling scale 1, four checkpoints, all local X/Y/Z observables
(48 features), exact expectations, the same couplings in each paired comparison.
The QRC carries state; the parallel QELM resets and encodes the latest four
inputs on four qubits. Shared candidate loss rates are .1, .3, 1, 4.

Fit the existing standardized ridge readout at alphas
1e-8,1e-6,1e-4,1e-2,1,100. Select alpha and loss separately for each model using
validation MSE only, not the held-out test. Save every candidate's test result
for transparency, but do not use it for selection. The primary comparison is
between the two validation-selected models.

## History control and classical context check

At the selected QRC loss rate, reset the same QRC before every prediction and
replay only d=1,4,8,12,24,48 latest inputs, oldest first. Refit each readout on
training data and select its alpha on validation. Qubits, observables, couplings
and final-step checkpoints stay fixed; the input preparation and circuit depth
are different from parallel QELM encoding. Call this the reset/replay history
control, not an identical-cost parallel QELM. Compare d=4 against the full QRC.
Longer replay requires more evolution steps and is not a free resource.

Also fit quadratic classical window models at d=4 and 12 with the same ridge
grid/splits. These have 14 and 90 nonconstant features, respectively; they are
context checks, not feature-budget-matched quantum-advantage comparisons.
Include the training-mean predictor. Do not suppress strong classical results.

## Paired input intervention on the task itself

Take the first and second 128-input prefixes of the unquantized test stream and
append the same next 48 inputs to each. Generate the actual NARMA targets for
both sequences and evaluate the two selected models with their fitted readouts.
After four common inputs, the four-input QELM predictions must be identical.
The actual targets may still differ. This directly demonstrates hidden task
history; a target pair is not a global bound or an estimate of typical error.
Save all roots and show the first root, not the one with the largest gap.

## Reporting and display

Primary error: test RMSE divided by training target standard deviation (NRMSE).
Report all 12 roots, median and 10th–90th percentile spread. Paired differences
QELM minus QRC and replay-d4 minus full QRC: mean and 95% paired bootstrap
interval, 10,000 draws, seed 929991. Roots include input and coupling variation.
These exploratory intervals are not a confirmatory significance claim.

Display root 920000 and the first 96 scored test rows, fixed before running.
Show actual targets and fitted predictions, paired whole-test results, the
history-control curve, and the same-continuation target intervention. Use the
existing Cobalt palette. No curve smoothing, seed selection, excerpt search,
or imported theoretical error floor. Keep the current paper PDF unchanged.

Validation: independent recursion/index checks, no-target-leakage tests,
same-window invariance, exact replay equivalence (existing tests), physical
state sanity, complete cohort and candidate selection checks, recomputed all
scores and source hashes, and visual inspection. A QRC win alone does not
isolate memory, and no outcome establishes quantum advantage or a need for
specifically quantum recurrence.
