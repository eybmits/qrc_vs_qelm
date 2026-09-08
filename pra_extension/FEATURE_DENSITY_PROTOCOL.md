# Feature-density follow-up for Figure 2

This is a new, prior-results-informed follow-up, locally frozen before opening
fresh production roots. It does not amend any previous preregistration or
replace the archived results. It is not public preregistration.

Question: how does fitted QELM error change across six measured-feature
budgets at fixed sufficient context and fixed evolution horizon? The settings
in configs/feature_density.json are authoritative. Twelve fresh root blocks
each contain target sizes 2, 3 and 4; sizes are analyzed separately. Each
root/size fixes its target, six-qubit emulator, streams and diagnostic panel.
Across budgets these are identical. Target sizes need not share the same
random realization. No matched-qubit task-performance advantage is claimed:
the target is the declared QRC output, not an external prediction task.

The canonical XX dynamics, local loss gamma=4, dt=0.5, input alphabet of 33
uniform levels, window d=6, encoding, local XYZ emulator observables and
target single-site/same-axis two-site Pauli outputs are unchanged. The QELM
evolves for the same total time 0.5. For m=1,2,3,4,6,8 checkpoints, use
times j*0.5/m, j=1,...,m. All are computed on one 24-checkpoint grid. Each
bank is uniformly spaced and contains the final time. Banks are nested when
their checkpoint counts divide one another, not for every adjacent budget.
The old 1/2/4 schedules are preserved exactly. More checkpoints do not mean
more history, longer dynamics, more qubits or necessarily independent features.

Use 480 training, 160 validation and 320 test rows, each stream with 128
washout steps. Standardization uses training rows only; validation selects
the historical ridge penalty, then the readout is refitted on 640 rows with
the training scaler fixed. Numerical SVD cutoff remains 1e-6. Exact
expectations are used. Quadratic delay uses the same windows, rows and fit
policy. Its dimension is recorded; there is no dimension-matching claim.

Use the historical 32-group common-suffix construction: unique newest
symbols sampled without replacement, with four independent 128-step
prefixes per suffix. Suffixes are chronological for target evolution and
newest-first for QELM encoding. The original finite-panel decomposition and
panel target-variance normalization are retained. Report raw total, history
and residual MSE, NRMSE, identity error, target variance, selected penalty,
resolved feature rank, streaming test MSE and all seeds/array hashes.
No test or diagnostic outcome selects budgets, roots or hyperparameters.

Before production: validate checkpoint extraction against directly evaluated
uniform schedules; test train-only scaling and suffix orientation; run two
identical development smokes using a disjoint seed; require matching scientific
hashes and passing tests. Bind protocol, config, numerical dependencies, tests
and analysis with a source manifest. Audit base and derived seeds against the
historical registry, including the training-data follow-up. Production requires
the matching freeze and an absent/empty output directory. Preserve failures.

Analysis is root-paired within each target size. Predefined endpoint contrasts
are 18->144 and 72->144 features in normalized total common-suffix MSE. Report
paired median absolute/relative decreases, 20,000-root-bootstrap intervals,
improving-root counts and one-sided sign p-values (ties counted as failures),
with Holm correction across all six contrasts. All 252 rows are retained:
12 roots x 3 sizes x (6 QELM budgets + quadratic delay). Plot all 216 QELM
rows; no post hoc selection. Statistical evidence of a reduction is not a
claim of a large practical gain, universality or a pure approximation floor.

Figure contract: keep the existing 1x3 square-panel layout and fonts. Panel a
adds the independent archived four-qubit cohort at the same loss/window cells
as the displayed two/three-qubit scaling series; no root pooling across sizes.
Panel b shows the fresh six-point feature sweep for 2/3/4-qubit targets, medians
with 10th-90th percentile root-spread whiskers, not confidence intervals.
The colors/markers identify target size consistently across a and b. Panel c
retains the original finite-shot sources and settings. Caption and methods
distinguish the cohorts. Connecting segments are not additional measurements;
there is no smoothing. Rebuild and visually inspect both paper PDFs, preserving
the twelve-page limit and all unrelated content.
