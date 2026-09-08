# Actual intermediate budgets between 18 and 36

This author-requested exploratory extension adds B=24 and B=30 to the existing
feature-density cohort. It does not claim new independent roots, a new public
preregistration, or untouched confirmatory data. The original frozen study,
protocol, outputs and six-budget analysis remain unchanged. This extension's
configuration, code, tests and parent artifacts are locally locked before
running the production extension. Previously observed endpoints motivate the
question; no new intermediate result selects the feature subsets or settings.

The original six-qubit QELM has 18 local XYZ expectations per checkpoint.
B=18 measures them at final evolution time 0.5; B=36 adds all 18 at time 0.25.
The new banks retain all final-time measurements and add one (B=24) or two
(B=30) measurements per qubit at time 0.25. For qubit i=0,...,5, first add axis
(i+p) mod 3 and then (i+p+1) mod 3, with axis order X,Y,Z. Root feature01 has
p=0, feature02 p=1, feature03 p=2, cycling through the twelve roots. Thus each
addition uses every qubit and balances X,Y,Z exactly; each phase occurs four
times per target size. This assignment is fixed by root identity, not outcomes.
The sets are nested 18 subset 24 subset 30 subset 36. At 36 the old canonical
column order is retained exactly. Feature identity, not only count, matters;
the plot characterizes these banks, not every possible B-feature selection.

Reuse every parent target/emulator seed, stream, 480/160/320 split, 128-step
burn-in, window d=6, gamma=4, evolution horizon, target observables and 32x4
common-suffix panel. Reconstruct exact features on the unchanged 24-point grid.
Training-only scaling, validation-selected ridge and 640-row refitting are
unchanged. Test and panel outcomes never select features or hyperparameters.
All values are exact expectations, not additional finite-shot observations.

Run all twelve parent roots at N=2,3,4, evaluating 18/24/30/36: 144 rows, of
which 72 are genuinely new intermediate-budget fits and 72 are endpoint
replication controls. Every reconstructed scientific array hash must match
the parent run; each 18/36 row must reproduce the archived parent metrics and
fit metadata within declared numerical tolerance (rtol=1e-11, atol=1e-14).
Do not append new results to the parent directory. Any mismatch fails closed;
preserve its failure record and do not combine incompatible cohorts.

Before production, compare bank columns and observable names, test nesting
and exact balance, compare endpoints to the original extraction, check
train-only scaling, complete two deterministic development smokes on the
parent's disjoint development block, and lock source hashes with passing
JUnit evidence. No new randomness is introduced; reused seeds are disclosed.

Report root medians and 10th-90th root percentiles for every tested bank;
adjacent-budget paired changes are descriptive. No new significance or
equivalence claim is made from these already inspected root systems. The
original classical controls and original six-budget inference remain available.

Figure contract: add measured 24/30 points between 18 and 36, while keeping
the recently approved line/marker style, legend clearance and linear feature
axis. Show 18,24,30,36,54,72; retain 108/144 in the complete source and report.
Only the 72 new rows join the figure export, not the replicated endpoints,
giving 984 rows (504 history, 288 features, 192 shots). Preserve the original
rows and lineage, explicitly identify the exploratory partial-bank extension
in caption/methods, and keep both manuscript versions at twelve pages.
