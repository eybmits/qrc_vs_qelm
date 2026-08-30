# The Two-Bottleneck Principle for QRC-to-QELM Emulation

This note states the theory used by the canonical empirical study in the
simplest form needed for the paper. It does **not** claim that every finite
QELM approximates every QRC.

## 1. The objects being compared

A recurrent QRC teacher updates an internal state,

`rho_t = Phi[u_t](rho_{t-1})`,

and reports an observable vector `y_t = h(rho_t)`. Its output can depend on the
entire past because `rho_{t-1}` carries information forward.

A nonrecurrent length-`d` student is reset on every row and has the form

`p_t = g(u_t, ..., u_{t-d+1})`.

The student may have arbitrarily complicated features, quantum or classical,
but it cannot distinguish two histories that have the same final `d` inputs.

## 2. Proposition A: the common-suffix obstruction

Fix a suffix `s` of length `d`. Suppose two older prefixes followed by `s`
produce teacher outputs `y_1` and `y_2`. Every stateless `d`-window model has
the same prediction law `A|s` on both branches. The triangle inequality gives

`max_i E||y_i-A|| >= ||y_1-y_2||/2`.

So a large teacher-output separation after a common suffix is a certificate
that **no** stateless model with that window can be uniformly accurate.
This is architecture independent: adding QELM qubits, observables, snapshots,
Finite-shot randomness cannot help because its conditional output law is also
the same on the two branches.

The claim is deliberately local. It refers to the tested suffix length, input
distribution, output observables, norm, and tolerance. It is not a statement
that every finite window fails.

## 3. Exact recurrent-QRC versus reset-window separation

The paper also constructs one fixed two-qubit QRC for which the local witness
exists at **every** finite window. An old symbol applies either the identity or
an addressable one-qubit pulse. Every later symbol drives permutation-symmetric
Hamiltonian dynamics with collective loss. The singlet projector

`P_minus = (II - XX - YY - ZZ)/4`

is conserved by every suffix channel, the old pulse changes its population,
and the declared Pauli readout sees it. With pulse angle one, every finite
window admits a same-suffix witness pair with output gap `0.1149244235`.
Therefore every reset-based finite-window QELM has worst-case expected absolute
error at least `0.0574622`; under the balanced witness law its MSE is at least
`0.0033019058`.

The quantifier order is the important point:

`there exists one fixed teacher; for every d; for every stateless suffix-only student; there exists a witness pair.`

The theorem does not include stateful, prefix-augmented, or branch-informed
students, and it does not say that every QRC defeats every QELM.

## 4. Proposition B: exact history-versus-representation decomposition

For one suffix group, let `y_k` be the teacher outputs obtained from `K` old
prefixes, let `mean_y` be their mean, and let `p` be the student's one suffix
prediction. Averaging squared error over suffix groups, prefixes, and outputs,

`MSE_total = MSE_history + MSE_representation`,

where

- `MSE_history = mean ||y_k - mean_y||^2`, and
- `MSE_representation = mean ||mean_y - p||^2`.

The cross term vanishes because the deviations `y_k-mean_y` sum to zero in
each group. Therefore:

- history error is a student-independent finite-window floor;
- representation error is what remains because the complete fitted student
  pipeline does not reproduce the suffix-conditioned mean.

The second name is shorthand for a **student-side representation/readout
residual**. In a finite experiment it can include feature-span mismatch,
finite training data, ridge selection, and readout estimation. The exact
identity alone does not isolate a pure feature-span invariant; rank and
target-projection diagnostics provide separate mechanism evidence.

This identity is exact on every sampled suffix panel, not an asymptotic
approximation. Because the realized within-group term uses divisor `K`, it is
downward biased as a population conditional-variance estimator: for
conditionally iid prefix branches its expectation is `(K-1)/K` times the
suffix-averaged population quantity. At the registered `K=4`, the separate
population-numerator calibration is `4/3`; it is not substituted into the exact
empirical decomposition.

## 5. Proposition C: fading memory predicts the needed window

Assume that applying the same input contracts any two teacher states by at
most a common factor `kappa < 1` in trace distance:

`||Phi[u](rho)-Phi[u](sigma)||_1 <= kappa ||rho-sigma||_1`.

After a common suffix of length `d`, repeated application gives

`||rho_d-rho'_d||_1 <= kappa^d ||rho_0-rho'_0||_1 <= 2 kappa^d`.

Every measured Pauli expectation changes by no more than the trace distance,
so the RMS observable diameter also decays no slower than `2 kappa^d`. This
predicts an exponentially shrinking history floor and a tolerance window of
roughly

`d_required proportional to log(1/epsilon) / -log(kappa)`.

The assumption must be checked. Pure unitary evolution generally has no
strict state-level contraction, and symmetry-protected or decoherence-free
sectors can retain old-prefix information. In the experiment we therefore
report the measured common-suffix decay and an empirical exponential fit; we
do not infer uniform contraction from one spectral gap.

For a tunable local-loss family it is natural to expect, over a fixed operating
range, `kappa(gamma) approximately exp(-c * gamma * dt)`. Substitution gives the
particularly simple testable collapse

`history influence approximately A * exp(-c * gamma * dt * d)`.

Thus the dimensionless product `gamma * dt * d`, rather than dissipation or
window alone, should approximately locate the history-feasibility boundary.
This scaling is a conditional physical prediction, not an identity: Hamiltonian
mixing, normalization by target variance, and noncontracting invariant sectors
can spoil it.

## 6. Proposition D: the feature-span obstruction

At a fixed window, collect the student's features in a design matrix `X` and
the suffix-conditioned teacher means in `Y`. With an affine linear readout,
the smallest in-sample squared error is exactly the residual after projecting
`Y` onto the column span of `[1, X]`.

Consequently, more nominal features help only if they add stable directions
aligned with the target. Duplicated, nearly collinear, or target-orthogonal
measurements do not remove the representation error. The study reports rank,
effective rank, conditioning, and target-aligned projection fraction alongside
held-out error for this reason. In the implementation these diagnostics are
computed on the streaming training design and targets, so they are finite-sample
mechanism proxies; they are not the exact suffix-conditioned projection defined
in the proposition above.

## 7. Constructive reset-QELM approximation theorem

For a bounded scalar output, define `D_d` as the largest output range among
histories with the same length-`d` suffix. If suffix predictors may be arbitrary
functions, their optimal uniform error is exactly `D_d/2`. Hence uniform
finite-window approximation is equivalent to `D_d -> 0`.

The representation step is explicit for a growing Bernstein QELM. Assume

`Dtr(Phi_u(rho), Phi_v(sigma)) <= eta Dtr(rho,sigma) + L_u |u-v|`

with `eta < 1`, scalar inputs in `[0,1]`, and output
`Y=Tr(O rho)`, `M=||O||_infinity`. Encode each of the `d` lags into `r` freshly
prepared qubits and measure the tuple of blockwise Hamming weights. The
resulting `(r+1)^d` probabilities are the tensor Bernstein basis. Readout
coefficients equal to the reset-replay target at the grid nodes give

`uniform error <= 2 M eta^d + M L_u (1-eta^d)/((1-eta) sqrt(r))`.

The first term is history truncation and the second is finite-window
representation. Directly scoring and averaging `Q` Hamming outcomes adds, with
confidence `1-alpha`,

`M sqrt(2 log(2/alpha)/Q)`.

For fixed problem constants, the qubit count is
`O(epsilon^-2 log(1/epsilon))`; the coefficient table has `(r+1)^d` entries and
is not efficient in general. The construction follows the established
parallel Bernstein quantum feature map of Goto, Tran, and Nakajima. The new
paper-level result is its combination with QRC observable-memory truncation and
the protected obstruction. It does not prove efficient learning or
universality of the fixed six-qubit QELM used in the phase map.

## 8. Falsifiable empirical predictions

The theory yields four plain predictions.

1. Stronger teacher forgetting shifts the history-feasible boundary to shorter
   windows.
2. Changing student feature budget cannot move the measured history floor.
3. After the history floor is below tolerance, added useful QELM features can
   reduce representation error and cross into emulation.
4. Before the history floor is below tolerance, added features cannot cross the
   emulation threshold, even if their regression fit becomes richer.

The canonical study varies teacher dissipation, window length, and nested
measurement budget independently so that these predictions can fail rather
than being built into the conclusion.
