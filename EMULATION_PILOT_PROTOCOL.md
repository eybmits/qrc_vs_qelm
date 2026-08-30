# Exploratory QRC-to-QELM Observable-Emulation Pilot

**Status:** initial exploratory protocol executed. A post-run independent audit
found that the first four-system dissipation sweep changed the Hamiltonian and
input seeds together with `gamma`. Those outputs remain valid pipeline checks,
but they are confounded and must not support a dissipation-scaling claim. The
prospective matched extension is frozen separately in
`EMULATION_MATCHED_PROTOCOL.md`.

**Relationship to earlier studies:** separate work package. It does not amend
`PROTOCOL.md`, `PREREGISTRATION.md`, or the held-out recurrence-fingerprint
addendum. It uses a fresh exploratory seed namespace and exact simulator
expectations only.

## Question

How much explicit input window and static feature budget must a reset QELM pay
to reproduce the accessible observable output vector of a recurrent QRC?

The target is the complete shared vector of single-site and same-axis two-site
Pauli expectations. It is not an external benchmark target.

## Models and controls

- **Teacher QRC:** recurrent input-affine Lindblad reservoir.
- **Constructive oracle:** reset-replay QELM with the teacher's exact coupling
  matrix and only the last `d` inputs. Its identity readout isolates history
  truncation from representation learning.
- **Random-bank QELM:** one, two, or four independently seeded reset-replay
  QELMs with the same qubit count, operating parameters, measurements, and
  window. Their features are concatenated in a fixed nested order. This is the
  non-tautological learned student.
- **Linear-delay control:** ridge regression directly on the same last `d`
  classical inputs.
- **Constant control:** the training-target mean.

Every trainable model uses one global ridge parameter selected on validation
aggregate NRMSE from the frozen ordered grid. Feature scaling is fit on training
rows only. The final readout is fit on train plus validation using that frozen
scaler and evaluated once on an independently generated reset test stream.

## Exact common-suffix certificate

For paired histories `h_a,h_b` sharing their last `d` inputs, let

`r_i(d) = ||F(h_a)-F(h_b)||_2 / sqrt(m)`

for the `m=18` accessible outputs. Define

- `D_inf(d) = max_i r_i(d)`,
- `D_2(d) = sqrt(mean_i r_i(d)^2)`.

Every deterministic `d`-window predictor must have maximum arm error at least
`D_inf(d)/2`, and paired RMSE at least `D_2(d)/2`. These are exact certificates
on the sampled history pairs, not estimates of the global suprema. All model
predictions for the two arms must agree to the configured numerical tolerance.

## Pilot axes

The primary pilot varies local-loss strength over a non-fading unitary control
and three dissipative regimes. It sweeps windows
`d={1,2,4,8,16,32,64,128}` and random-QELM bank budgets `R={1,2,4}`. Full values,
stream sizes, and fresh seeds are in `configs/emulation_pilot.yaml`.

The constant-input Liouvillian spectral gap is reported only as a mixing proxy.
It is not treated as a certified uniform contraction coefficient. Empirical
trace-distance decay under the paired common suffix is the direct mechanism
diagnostic.

## Pre-run pilot decision rules

1. **E0 implementation gate:** deterministic execution, independent splits,
   train-only scaling, identical suffix predictions, and both lower-bound
   inequalities hold for every model/system/window. Failure is `NO_GO`.
2. **E1 liveness:** the configured fraction of systems has `D_2(0)` above the
   frozen floor.
3. **E2 memory-error association:** the Spearman association between the
   constructive-oracle paired error and `D_2(d)` meets the frozen threshold.
4. **E3 constructive convergence:** the configured fraction of dissipative
   systems achieves the frozen reduction in oracle test NRMSE from the smallest
   to largest window.
5. **E4 generic learned emulation:** the largest fixed random-QELM bank meets
   either the frozen absolute NRMSE or reduction requirement.
6. **E5 non-fading boundary:** the unitary control retains the frozen fraction
   of its `D_2(0)` separation at the largest window.

Outcomes are interpreted as follows:

- E0 failure: implementation `NO_GO`.
- E1--E3 pass and E4 pass: evidence for a finite-resource QELM emulation law.
- Oracle convergence with E4 failure: feature-accessibility bottleneck; expand
  the QELM bank before confirmatory testing.
- Persistent certificate and oracle error at the largest window: recurrence is
  acting as compression on the tested horizon; extend the window.

No outcome is evidence of quantum advantage. Classical controls are reported
as mechanism comparisons.
