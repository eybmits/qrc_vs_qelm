# Matched QRC-to-QELM Emulation Pilot

**Status:** the three-block `matched_pilot` discovery run was executed and
passed its frozen gates. It reused one random QELM bank across teacher blocks,
so it is not a bank-seed robustness test. Before any `replicated_pilot` output,
a second held-out phase was frozen below with four fresh, independent bank
draws and fresh teacher/data/pair seeds.

## Simple question

When a QRC forgets old inputs, how much window memory `d` and how many
independent QELM feature reservoirs `R` are needed to reproduce its accessible
observable trajectory?

The intended empirical statement is deliberately bounded:

> fading makes finite-window reset replay accurate, while an independently
> randomized QELM may additionally need a larger accessible feature bank.

This is not a theorem about every QRC or every QELM.

## Audit-motivated corrections

The initial pilot varied the Hamiltonian and data streams together with
dissipation. This extension uses three independent replicate blocks. Within
each block, the teacher coupling, train/validation/test streams, paired-history
panel, and student bank are fixed while only `gamma={0,0.05,0.2,0.5}` changes.
All seeds are fresh relative to the initial pilot.

The paired-history audit now evaluates the production delay embedding
separately on both complete history arms. Identical predictions are measured,
not constructed by copying one prediction. The linear-delay control receives
the same 33-symbol quantized input alphabet as the teacher and QELMs.

## Frozen design

- three matched Hamiltonian/data replicate blocks;
- four dissipation strengths per block, including the unitary boundary;
- windows `d={1,2,4,8,16,32,64,128}`;
- nested independent QELM banks `R={1,2,4}`;
- 480 train, 160 validation, and 320 untouched test rows per system;
- 32 independent paired histories per system;
- exact simulator expectations for 18 shared one- and two-body Pauli outputs;
- ridge selection on validation data, with train-only feature scaling;
- separately reset train, validation, test, and certificate streams.

The full configuration and exact thresholds are in
`configs/emulation_pilot.yaml` under `matched_pilot`.

## Held-out bank-seed replication

The `replicated_pilot` profile repeats the same windows, sample sizes,
observables, controls, and decision thresholds in four fresh matched blocks.
Each block has its own four-reservoir nested student bank, held fixed only
across the four gamma values in that block. Student banks, teacher
Hamiltonians, input streams, and paired-history panels are mutually fresh and
distinct between blocks. This phase tests whether the feature-budget effect
survives independent QELM bank draws; it was specified after seeing the
matched discovery output but before seeing any replication output.

## Frozen empirical gates

1. **Implementation:** separately evaluated common-suffix predictions agree and
   every sampled `D_inf(d)/2` and `D_2(d)/2` lower bound holds.
2. **Liveness:** at least 75% of systems have a nontrivial initial observable
   separation.
3. **Independent mechanism association:** in at least two thirds of
   dissipative systems, the suffix-memory witness from the paired panel and
   oracle truncation error on the independent stream test have within-system
   Spearman correlation at least 0.8.
4. **Constructive convergence:** in at least two thirds of dissipative systems,
   same-Hamiltonian reset replay reduces test NRMSE by at least 50% between
   `d=1` and `d=128`.
5. **Generic window convergence:** the `R=4` random bank reaches NRMSE at most
   0.25 or reduces it by at least 50% in at least two thirds of dissipative
   systems.
6. **Feature-resource gain:** at `d=128`, `R=4` improves over `R=1` by at least
   15% in at least two thirds of dissipative systems.
7. **Classical-control specificity:** at `d=128`, `R=4` beats the matched linear
   delay readout by at least 10% in at least two thirds of dissipative systems.
8. **Non-fading boundary:** at least two thirds of unitary systems retain 90%
   of their initial mean state trace distance after the 128-input common
   suffix.
9. **Matched mixing order:** in at least two thirds of replicate blocks,
   `gamma` and the remaining normalized observable suffix witness have
   Spearman correlation at most -0.8.

Passing all gates is labeled only `PILOT_SUPPORTS_TWO_RESOURCE_SCALING`.
Constructive gates passing without all generic-student gates is labeled
`PILOT_SUPPORTS_CONSTRUCTIVE_TRUNCATION`. Neither label establishes a law,
quantum advantage, continuous-input universality, or hardware robustness.

## Scope boundary

The input alphabet has 33 symbols, the simulator is exact, the family has three
qubits and local loss, and only the registered 18-observable projection is
distilled. Replicates quantify seed robustness within this family; broader
qubit counts, dissipators, continuous inputs, finite shots, and a fresh
confirmatory seed gate remain separate work.

## Wide-ridge sensitivity amendment

After `replicated_pilot` completed, an independent audit found that many
learned models selected the largest registered ridge penalty, `alpha=1`. Before
running the sensitivity output, `replicated_pilot_wide_ridge` was frozen with
the same data and physical seeds but a validation-only grid extending through
`alpha=10^6`. No test target enters alpha selection. Learned-student and
classical-control conclusions must use this wide-grid output; the constructive
identity replay and suffix certificates do not depend on ridge regression.
After that output showed that only near-nonlearnable unitary fits still selected
the finite upper boundary, a final pre-run sensitivity profile added the exact
`alpha=infinity` intercept-only limit. Final learned-model numbers use
`replicated_pilot_intercept_complete`; an infinite selected penalty means that
validation preferred to discard the candidate features entirely.
