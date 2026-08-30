# Recurrence-Fingerprint Study: Internal Frozen Preregistration

**Lock:** `recurrence_fingerprint_v1`

**Status:** internal lock, simulator-only

**Lock date:** 2026-08-28

**Machine-readable mirror:** `data/frozen/recurrence_fingerprint_v1.json`

This is a separate study addendum. It neither changes nor supersedes
`PREREGISTRATION.md`, `PROTOCOL.md`, or `data/frozen/frozen_v1_config.json`.
The source checkout was dirty when this internal lock was written. The recorded
base commit is `baa7a8ac22f1e23a11bc07b5883ff164f75bb47e`; a run manifest must record the
exact runner/config hashes and dirty-tree status used for every execution.

## 1. Question and bounded claim

The study asks whether an architecture-blind rule can distinguish an accessible,
reset-sensitive temporal tail from behavior compatible with a finite input cutoff.
The inputs to the rule are only paired-history output-distance curves under carry
and ideal-reset interventions. Device names, implementation labels, and declared
window sizes are hidden from the rule.

“Architecture-blind” is a code-path claim, not a human double-blind claim. The
fixed classifier rejects source/model labels, expected behavior, adapter type,
and declared window columns. It receives a deterministic case pseudonym plus
`physics_id` only to keep complete histories and block bootstrap units paired.
Because the pseudonym is reproducible from public study fields, it should not be
described as cryptographically secret or as masking the study from a human analyst.

If every held-out gate passes, the allowed claim is:

> In the frozen simulator family, over the registered `N`, dissipator, `gamma`,
> `dt`, observable, and shot ranges, a recurrence fingerprint calibrated and frozen
> using only the design split distinguishes the state-carrying implementation from the declared
> finite-window implementations with the registered finite-shot accuracy, and the
> fingerprint covaries with a shift toward QRC performance on longer linear-delay
> tasks.

The study does **not** establish hardware performance, quantum advantage,
non-Markovianity, microscopic architecture identity, universal QRC superiority, or
an impossibility theorem for arbitrary QELMs. A positive temporal tail is an
operational witness of accessible state carryover in this intervention, not a
complete model-identification theorem.

## 2. What is and is not scientifically nontrivial

The exact cutoff of a reset-per-row model whose output is a deterministic function
of its last `d` inputs is definitional. Likewise, an ideal forced reset eliminating
all pre-reset history is a calibration fact. These outcomes validate the assay but
are not discoveries.

The nontrivial test is simultaneous held-out generalization across fresh couplings,
input streams, system sizes, dissipators, operating points, and finite-shot draws:
the architecture-blind classifier must retain its registered performance, the QRC
tail must be reset-sensitive beyond `Kmax = 5`, and the task-link crossover must be
positive. The leaky-QELM result is a positive-control check that the assay responds
to state carryover rather than to a device name.

## 3. Split discipline and lock rules

- The eight design blocks were already explored and may be used for implementation
  debugging and for freezing the estimand, thresholds, and rule below. They never
  enter a held-out gate or confidence interval.
- The 24 held-out blocks use fresh coupling, history, and task-input seeds. No
  held-out result may be used to change preprocessing, thresholds, suffixes, model
  semantics,
  aggregation, exclusions, or gates.
- The held-out split is opened once, after the runner validates this JSON and writes
  hashes of the runner, dependencies, and config to a manifest.
- A pure implementation correction must be documented before inspecting the
  corrected outcomes. Any analysis change after held-out outcomes are visible is
  exploratory and requires a new held-out seed namespace for confirmation.
- All registered held-out blocks and device rows are retained. Numerical failure is
  reported and counts as an abstention/error; it is not silently dropped.

Implementation-audit clarification, frozen before opening H01--H24: if an
exception or nonfinite value prevents construction of the complete registered
grid, the runner does not impute replacement values or compute a reduced-sample
metric. It retains every table completed before the failure, writes a structured
ledger with expected, observed, missing, and excess rows, seals a final manifest,
and assigns the whole confirmatory study `NO_GO`. This is the fail-closed
study-level realization of counting missing numerical outcomes as errors.

## 4. Frozen physics blocks

All blocks use `J_s = 1`, `h = 0.5`, `h_in = 1`, 33-level input quantization, one
virtual node/snapshot, all single-site Pauli observables over axes `X,Y,Z`, and
same-axis two-site observables `X_iX_j`, `Y_iY_j`, and `Z_iZ_j`. Observable
dimensions are 9, 18, and 30 for `N = 2,3,4`, respectively. Within
a block, every implementation uses exactly the same Hamiltonian couplings,
dissipator, input histories, observables, and shots per expectation.

### Design blocks (explored, not confirmatory)

These physics tuples are the actual explored design scan. Fresh history seeds are
locked here; the already explored task streams retain seeds `720000..720007`.

| block | coupling seed | history seed | task input seed | N | dissipator | gamma | dt |
|---|---:|---:|---:|---:|---|---:|---:|
| D01 | 710001 | 730001 | 720000 | 2 | local_loss | 0.08 | 0.35 |
| D02 | 710002 | 730002 | 720001 | 3 | local_loss | 0.20 | 0.50 |
| D03 | 710003 | 730003 | 720002 | 4 | local_loss | 0.35 | 0.65 |
| D04 | 710004 | 730004 | 720003 | 2 | collective_loss | 0.20 | 0.65 |
| D05 | 710005 | 730005 | 720004 | 3 | collective_loss | 0.35 | 0.35 |
| D06 | 710006 | 730006 | 720005 | 4 | collective_loss | 0.08 | 0.50 |
| D07 | 710007 | 730007 | 720006 | 2 | dephasing | 0.35 | 0.50 |
| D08 | 710008 | 730008 | 720007 | 4 | dephasing | 0.20 | 0.35 |

### Untouched held-out blocks

| block | coupling seed | history seed | task input seed | N | dissipator | gamma | dt |
|---|---:|---:|---:|---:|---|---:|---:|
| H01 | 810001 | 830001 | 820001 | 2 | local_loss | 0.05 | 0.35 |
| H02 | 810002 | 830002 | 820002 | 3 | collective_loss | 0.10 | 0.40 |
| H03 | 810003 | 830003 | 820003 | 4 | dephasing | 0.15 | 0.45 |
| H04 | 810004 | 830004 | 820004 | 2 | collective_loss | 0.20 | 0.50 |
| H05 | 810005 | 830005 | 820005 | 3 | dephasing | 0.25 | 0.55 |
| H06 | 810006 | 830006 | 820006 | 4 | local_loss | 0.30 | 0.60 |
| H07 | 810007 | 830007 | 820007 | 2 | dephasing | 0.35 | 0.65 |
| H08 | 810008 | 830008 | 820008 | 3 | local_loss | 0.45 | 0.70 |
| H09 | 810009 | 830009 | 820009 | 4 | collective_loss | 0.05 | 0.50 |
| H10 | 810010 | 830010 | 820010 | 2 | local_loss | 0.10 | 0.55 |
| H11 | 810011 | 830011 | 820011 | 3 | collective_loss | 0.15 | 0.60 |
| H12 | 810012 | 830012 | 820012 | 4 | dephasing | 0.20 | 0.65 |
| H13 | 810013 | 830013 | 820013 | 2 | collective_loss | 0.25 | 0.70 |
| H14 | 810014 | 830014 | 820014 | 3 | dephasing | 0.30 | 0.35 |
| H15 | 810015 | 830015 | 820015 | 4 | local_loss | 0.35 | 0.40 |
| H16 | 810016 | 830016 | 820016 | 2 | dephasing | 0.45 | 0.45 |
| H17 | 810017 | 830017 | 820017 | 3 | local_loss | 0.05 | 0.60 |
| H18 | 810018 | 830018 | 820018 | 4 | collective_loss | 0.10 | 0.65 |
| H19 | 810019 | 830019 | 820019 | 2 | local_loss | 0.15 | 0.70 |
| H20 | 810020 | 830020 | 820020 | 3 | collective_loss | 0.20 | 0.35 |
| H21 | 810021 | 830021 | 820021 | 4 | dephasing | 0.25 | 0.40 |
| H22 | 810022 | 830022 | 820022 | 2 | collective_loss | 0.30 | 0.45 |
| H23 | 810023 | 830023 | 820023 | 3 | dephasing | 0.35 | 0.50 |
| H24 | 810024 | 830024 | 820024 | 4 | local_loss | 0.45 | 0.55 |

## 5. Frozen implementation panel

Each block contains nine implementations:

1. `qrc`: the state-carrying recurrent reservoir.
2. `replay_qelm_d1`, `replay_qelm_d3`, and `replay_qelm_d5`: reset to the same
   initial state for every evaluated row, then replay exactly the most recent
   `d` inputs through the matched channel.
3. `spatial_qelm_dmin`: reset for every evaluated row and spatially encode
   `d = min(N,3)` recent inputs.
4. `forced_reset_qrc_d3`: follow the QRC carry trajectory for `L < 3`, then
   deterministically erase the carried state and use its ideal-reset trajectory for
   `L >= 3`. This is a finite-cutoff negative mechanism control.
5. `leaky_qelm_control`: inject a reproducible state-leak fault by assigning the
   matched QRC carry/reset trajectories to a QELM-labelled case. It is a positive
   assay control, not a genuine finite-window QELM or independent architecture.
6. `reset_failure_qrc_control`: make the reset instruction a no-op, so both the
   carry and nominal reset conditions use the QRC carry trajectory. It must expose
   a recurrent but reset-insensitive signature.
7. `dead_readout_control`: replace both arms by the same all-zero observable vector
   in both conditions. It must trigger the registered low-signal abstention.

Thus there are five QELM-labelled variants only when the deliberately leaky control
is counted; the primary architecture scoring set is `qrc` plus the four genuine
finite-window QELMs. All four controls are reported separately and cannot improve a
primary gate.

## 6. Paired-history recurrence fingerprint

For each physics block, generate 24 complete paired histories. A pair has a common
12-symbol burn-in, two independently generated 16-symbol divergent prefixes, and
then the same suffix. Evaluate suffix lengths

`L = [0,1,2,3,4,5,6,8,12,16]`.

Inputs are uniform on `[0,1]` and quantized identically for every implementation.
A deterministic NumPy `SeedSequence([history_seed, coupling_seed, history_id])`
defines each per-pair RNG stream. The common burn-in, arm-A prefix, arm-B prefix,
and one length-16 maximum suffix are disjoint sequential draws from that stream;
all evaluated suffixes are nested prefixes of the maximum suffix. Complete curves
remain paired in every resample.

Two intervention conditions are evaluated:

- `carry`: preserve the implementation state accumulated before the shared suffix.
- `ideal_quantum_reset`: replace the quantum state at the branch/suffix boundary by the
  fixed initial state, then apply the same suffix. This tests idealized state
  dependence; it does not test reset fidelity.

The shot tiers are exact expectation values, `S = 10,000` (primary), and `S = 1,000`
(stress test). A finite `S` means shots per observable, arm, and independent batch.
There are two independent batches for each of the two history arms, so physical
shot accounting includes the factor four. The same eight locked measurement seeds
`840001..840008` are used within each block and finite tier, with block/device/curve
coordinates included in the derived RNG stream. Negative cross-batch estimates are
retained, never clipped.

For history pair `i`, condition `c`, suffix `L`, and observable vector of width `m`,
the unbiased finite-shot signal estimator is

`d_i,c(L) = m^-1 (z_A^(1)-z_B^(1))^T (z_A^(2)-z_B^(2))`.

Let `D_c(L)` be its mean over the 24 histories and
`R_c(L) = D_c(L) / D_carry(0)`. A block/device/measurement replicate abstains when
the lower 95% bootstrap bound of `D_carry(0)` is not greater than `1e-4`.

## 7. Architecture-blind rule

All confidence bounds below are 95% percentile bounds from 1,000 resamples of the
24 complete paired-history trajectories. The classifier receives only the blinded
history-level distance curves `d_i,c(L)` and identifiers needed to preserve their
pairing; it derives `D_c(L)` and `R_c(L)` internally.

For every candidate cutoff `k in {1,2,3,4,5}`, define

- pre-cutoff liveness `A_k = R_carry(k-1)`;
- post-cutoff equivalence `M_k = max_{L >= k} |R_carry(L)|` over registered suffixes.

Candidate `k` qualifies when `lower95(A_k) > 0.05` and
`upper95(M_k) < 0.02`. If more than one qualifies, infer the smallest as `k_hat`.

For the fixed beyond-window tail `G = {5,6,8,12}`, define

- `T = mean_{L in G} R_carry(L)`;
- `Q = max_{L in G} |R_reset(L)|`;
- `C = T - mean_{L in G} |R_reset(L)|`.

The recurrent criterion qualifies when `lower95(T) > 0.03`,
`upper95(Q) < 0.02`, and `lower95(C) > 0.03`.

The deterministic decision is:

- baseline failure -> `INDETERMINATE_LOW_SIGNAL`;
- recurrent criterion only -> `RECURRENT_RESET_SENSITIVE`;
- one or more cutoff candidates and no recurrent criterion ->
  `FINITE_WINDOW_COMPATIBLE_d{k_hat}`;
- a live recurrent carry tail with no qualifying cutoff and
  `lower95(mean_G |R_reset|) > 0.03` -> `RECURRENT_RESET_INSENSITIVE`;
- conflicting or otherwise insufficient evidence -> `INDETERMINATE`.

The two indeterminate labels are abstentions for scoring.

No device-specific threshold, declared window, or architecture label enters this
rule. Declared windows are used only after prediction to score cutoff accuracy.

## 8. Hierarchical uncertainty and scoring

The independent generalization unit is the physics block, not a history, device,
or measurement draw. Aggregate 95% intervals use 1,000 hierarchical bootstrap
replicates: resample the 24 held-out blocks first; within each selected block,
resample its eight measurement seeds; keep all device variants paired. History
resampling is already nested inside each curve-level decision. Task-link intervals
resample whole held-out blocks, preserving all delays and both architectures.
Within a block/shot/measurement replicate, every case variant uses the same history
resample indices. The classifier-bootstrap RNG domain is keyed by `physics_id`,
shot tier, and measurement seed, never by the config-derived case pseudonym.

For all primary classifier metrics, either indeterminate label, numerical failure,
or a missing row counts as incorrect. The primary scoring population is the QRC
(recurrent positive) and four genuine finite-window QELMs (finite negative). All
four fault controls are tabulated separately.

- Balanced accuracy is `(sensitivity_recurrent + specificity_finite)/2`.
- False-recurrent rate is the fraction of genuine finite-window QELM replicates
  labeled recurrent; abstentions are not false recurrent but do lower balanced
  accuracy and cutoff accuracy.
- Exact agreement compares each 10,000-shot decision with that block/device's exact
  decision. An abstention at either tier counts as disagreement.
- Cutoff accuracy is the fraction of genuine finite-window QELM replicates with
  `|k_hat - d_effective| <= 1`; all other decisions count as failures.
- Exact correct/error/abstention rates and all four control outcomes are mandatory
  reports, not acceptance gates.

## 9. Frozen task link

For each block, generate a distinct length-1,200 `Uniform(0,1)` input stream with
NumPy `default_rng(task_input_seed)`. The physical drive is quantized to the registered 33
levels inside each feature generator, while degree-one Legendre delay targets use
the original unquantized stream at `tau = [1,3,5,8,12]`. Use a washout of 100.
After removing invalid/washout rows, use ordered contiguous train/validation/test
fractions `0.6/0.2/0.2`. Feature standardization is fit on training rows only.
Ridge alpha maximizes validation squared-correlation capacity on the shared fixed
grid

`[0,1e-10,1e-8,1e-6,1e-4,1e-2,1,100]`.
Ties retain the first grid entry and therefore select the smallest tied alpha.

After alpha selection, the readout weights and unpenalized intercept are refit on
the combined training and validation rows; the test rows remain untouched until
the final score.

The primary task link uses exact features and the squared-correlation test capacity.
Its fixed comparator is `replay_qelm_d5`, matching the design pilot. For held-out
block `b` and delay `tau`,

`delta_b(tau) = C_test,qrc - C_test,replay_qelm_d5`.

Define the block crossover shift

`s_b = mean(delta_b(8), delta_b(12)) - mean(delta_b(1), delta_b(3))`.

Delay 5 is reported as a transition diagnostic but does not enter the primary shift.
No alternative comparator may replace the frozen replay-`d5` comparator.

## 10. Acceptance gates

All integrity prerequisites and every hard gate must pass. The primary finite-shot
tier is 10,000 shots; 1,000 shots is report-only stress testing.

1. **Integrity:** all 24 held-out blocks, 24 histories, ten suffixes, nine device
   rows, two interventions, exact tier, and eight measurement seeds at both finite
   tiers are complete; config/runner hashes match the opening manifest.
2. **Bounded false recurrence:** held-out false-recurrent rate for the four genuine
   finite-window QELMs at 10,000 shots is at most `0.05`.
3. **Finite-shot classification:** held-out balanced accuracy at 10,000 shots is at
   least `0.85`, and its block-cluster 95% lower bound is at least `0.70`.
4. **Exact agreement:** strict 10,000-shot versus exact decision agreement is at
   least `0.85`.
5. **Cutoff recovery:** at least `0.90` of genuine finite-window QELM replicates have
   cutoff error at most one symbol.
6. **Task crossover:** the block-bootstrap 95% lower bound for mean `s_b` is greater
   than zero, and at least `0.75` of the 24 held-out blocks have `s_b > 0`.

Failure of any hard gate yields `NO-GO` for the registered confirmatory claim. The
complete results, including nulls, errors, abstentions, and stress-tier degradation,
must still be reported. Passing permits only the bounded claim in Section 1.

## 11. Pre-held-out design amendment

The first complete D01--D08 design-only run was evaluated before any H01--H24
curve was generated.  It recovered every honest finite cutoff with zero false
recurrent calls, but the original recurrent-tail and reset-contrast margins of
0.05 abstained on D03 even though its exact mean tail was 0.040 and its lower
bootstrap bound was 0.036.  The two margins were therefore reduced together from
0.05 to 0.03, leaving a non-overlap zone above the unchanged 0.02 equivalence
margin.  On the design split this changed the 10,000-shot balanced accuracy from
0.8125 to 0.875, retained zero decisive errors and 1.0 cutoff recovery, and left
the two genuinely shot-unresolved dephasing cases as low-signal abstentions.  No
baseline, cutoff, reset-equivalence, gate, seed, suffix, task, or held-out setting
was changed.  The held-out split remained unopened when this amendment was locked.
