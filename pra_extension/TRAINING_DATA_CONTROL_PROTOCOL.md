# Training-data and conditional-mean follow-up

## Status and question

This is a separately specified follow-up chosen after inspecting the existing
paper, not an amendment to any historical confirmatory result. Production roots
must remain unopened until a machine-checked freeze record binds this protocol,
the resolved configuration, runner, dependencies, tests, and analysis code.
The freeze is a local prospective record, not an independently witnessed public
preregistration. No result or protocol from the original cohorts is replaced.

The question is whether additional training data reduces fitted QELM residual
error when the window already contains sufficient history in the previously
studied regimes, and how much a four-prefix estimate of the conditional mean
differs from an exact finite-prefix reference. This is simulated emulation of
fixed QRC outputs, not a matched-qubit QRC-versus-QELM task-performance contest.

## Fixed design

The authoritative numeric settings are in `configs/training_data_control.json`.
There are twelve new root blocks, with disjoint production and development
seed namespaces. Each root contains both anchors; anchors are reported
separately and never pooled as twenty-four independent observations.

| Anchor | Target qubits | Local loss | Window | QELM qubits | Feature budgets |
|---|---:|---:|---:|---:|---|
| n4_fast | 4 | 4 | 6 | 6 | 18, 36, 72 |
| n5_long | 5 | 1 | 10 | 10 | 30, 60, 120 |

These anchors were chosen from the earlier studies: one successful fast-loss
reference and one longer-window setting with substantial fitted residual.
The choice is therefore informed by prior results, not a random sample of all
reservoir families. No additional anchors are selected after production opens.

The target Hamiltonian, local-loss convention, time step 0.5, coupling law,
33-symbol uniform input alphabet, target single-site and same-axis two-site
Pauli outputs, QELM encoding, evolution time, and four maximum checkpoints
are inherited from the canonical implementation. Checkpoint subsets retain
the same final evolution horizon. Quadratic delay receives identical windows,
training sizes, validation and test rows. Its actual readout dimension is
reported; no dimension-matched or quantum-advantage claim is made.

Within a root/anchor, one maximum-length training stream supplies nested
training sets of 480, 960, 1,920 and 3,840 rows. Separate streams provide
512 validation and 2,048 test rows. Washout and causal delay alignment follow
the existing streaming implementation with 128 washout steps. Temporal rows
are not counted as independent root replicates.

Features are exact expectations. Only training rows fit feature centering and
scaling; the existing ridge grid selects a penalty using validation alone.
The selected readout is refitted on training plus validation while retaining
training-only standardization. Thus the final fitted row count is the x-axis
training count plus 512; it is recorded explicitly. The relative 1e-6 SVD
cutoff is unchanged. No test or oracle-reference value selects a model,
penalty, budget, seed, checkpoint or training size.

## Conditional-mean reference

Let the prefix contain exactly 128 independent uniform draws from the
33-symbol alphabet, starting at the target's standard initial state. Channel
linearity implies that the average prefix state is obtained by applying the
average single-input channel 128 times. Apply the specified suffix channels
to this average state, then evaluate the declared linear output map.
This gives the conditional mean for this finite-prefix distribution, to
floating-point accuracy, without enumerating all prefix histories.

The reference evaluation uses 1,024 fresh IID windows, independent of all
streaming splits. Their repeated values, if any, are legitimate repeated
draws: no sample-specific empirical means are used for their reference
labels. Windows are chronological for target evolution and converted to
the canonical input order for QELM encoding. The reference is evaluated,
not used as a training target or model-selection signal.

A second panel uses 32 independently drawn windows and four independent
128-step prefixes per window. It records the error to the four-branch mean,
error to the exact finite-prefix mean on the same windows, within-window
variation, and the mean-sampling discrepancy. The realized empirical MSE
identity is verified numerically. The exact-mean residual does not replace
the empirical residual inside that finite-sample algebraic identity.
The empirical-minus-exact residual difference may be negative for a realized
panel because of its cross term; no sign is forced and no difference is clipped.
The four-prefix residual inflation equals a quarter of conditional variance
only in expectation over independently sampled prefixes.

As a fixed sensitivity diagnostic, repeat the reference means at prefix
length 256 on the same 1,024 windows and record raw and normalized RMS and
maximum differences. This is a finite-prefix sensitivity check, not a proof
of stationarity or uniformly fading memory. It never changes the primary
128-step reference or triggers additional production settings.

This conditional mean is not the scalar worst-case midpoint, D_d/2, a
conditional-variance calculation, the stationary streaming conditional mean,
or the optimal readout within the selected QELM feature span. Streaming test
error and reference-panel residual are distinct estimands and remain separate.

## Metrics, analysis and decision discipline

For every root/anchor, one fixed normalization is the mean across target
observables of their variance on the independent streaming test targets
(population denominator). All models, training sizes and panel/reference
errors use that same denominator. Raw MSE and the denominator are archived.
Nonfinite values or a denominator at most 1e-8 fail the numerical run; no root
is silently removed from the fixed denominator of twelve.

Primary contrasts, separately for each anchor, are the change from 480 to
3,840 training rows in exact-reference normalized residual MSE for the
largest QELM feature budget. Report the paired root median decrease, the
median relative decrease, a 20,000-resample root-block percentile interval,
and the number of roots that improve. Use the fixed analysis seed. A paired
one-sided sign calculation is reported with Holm correction across these
two primary comparisons; zero improvements count conservatively as failures
in the twelve-root calculation. Statistical significance alone does not
establish a practically important effect.

Secondary analyses report every lower budget, intermediate training size,
quadratic-delay result, streaming test error, four-prefix mean discrepancy,
and 128-versus-256 sensitivity. No primary finding is rescued by selecting
one of these secondary contrasts. Root-level intervals describe these
sampled systems; no population-wide success probability is inferred from
the number of panel rows or observables.

A decrease supports a training-data contribution under this pipeline. A
small or inconclusive decrease does not establish an irreducible feature
floor: finite training, regularization, numerical rank truncation, feature
conditioning and the difference between stream and panel distributions can
remain. No fixed-QELM universality, sample efficiency, resource savings or
quantum advantage is claimed.

## Validation, provenance and execution

1. Unit-test the mean-channel reference against exhaustive short-prefix
   enumeration, including noncommuting channels, empty suffixes and invalid
   inputs. Test train-only standardization and split/window orientation.
2. Run two same-seed reduced development smokes in separate output directories;
   compare deterministic metrics and scientific-array hashes. Development
   roots never enter production summaries.
3. Benchmark actual N5 model dimensions on a separate development seed. Inspect
   runtime, peak memory and numerical integrity, not effect sizes. Production
   runs only if projected execution is feasible locally without a resource
   expansion; otherwise stop and report the estimate. Optimize an equivalent
   backend if needed, but do not weaken numerical settings after data access.
4. Freeze the full dependency material and config, record UTC time, smoke
   identities, successful tests and the unopened-root attestation. The runner
   verifies the identity before opening each root. New runs must not overwrite
   an existing result directory. Independent roots may execute in separate
   processes with bounded BLAS threads.
5. Verify complete production coverage, array/config/material hashes, numerical
   identities, seed uniqueness and root pairing before analysis. Preserve all
   outcomes, including failures or effects opposite the expectation.

Output directories are separate `results_training_data_control_*` directories.
They and any large generated arrays are not committed. Source/protocol files
are inspectable separately. A root-level manifest stores seed entropy,
inputs/features/reference hashes, settings and per-stage runtime. This run
does not alter historical manuscripts, figures, cohorts or freeze records.

## Figure contract

The standalone follow-up figure uses two matched axes, one per anchor, showing
exact-reference residual NRMSE versus the four observed training sizes. Four
points are intentional controlled sample-size settings, not a smoothed temporal
trend. Lines only connect tested settings. The three QELM budgets and quadratic
control are all retained with marker and color distinctions; root medians and
10th--90th percentile whiskers show between-root spread, not confidence bounds.
The layout, fonts, quiet two-direction grid and legend style follow the existing
scientific figures. CSV/JSON source data and a caption disclose the oracle-panel
estimand, fixed normalization, twelve roots, refitting convention and exact
expectations. Paper insertion, if later requested, must respect twelve pages
and must not replace historical evidence without explicit attribution.
