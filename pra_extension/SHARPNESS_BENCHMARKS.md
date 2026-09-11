# Exact sharpness verification examples

This is a separate, post-hoc implementation-verification package, introduced
after the original experimental cohorts. It neither reruns nor changes the
preregistered comparisons, frozen protocols, original data, or acceptance gates.
There is no fitted readout, train/test split, shot noise, or hardware run here.

## Model and interpretation

`code/sharpness_benchmarks.py` implements the one-qubit replacement channel

```
Phi_a(rho) = eta*rho + (1-eta)*diag(1-g(a), g(a)),  0 < eta < 1,
g(a) = a^2  or  (exp(beta*a)-1)/(exp(beta)-1),      0 <= a <= 1.
```

For the diagonal initial state used here, its population evolution is also a
classical scalar leaky integrator. The quantum channel supplies a controlled
reference example, not evidence of quantum advantage or generic QRC performance.
From the fixed zero population, the reachable-state closure contains populations
from zero to one. The upper endpoint is a limit of longer finite prehistories;
it need not be attained by an individual finite history.

The analytical benchmark definitions are `D_d = eta^d` and the compatible-output
midpoint `m_d = eta^d/2 + sum_i (1-eta)*eta^(d-i)*g(a_i)`. The error calculation
uses the analytic identity `E(f) = D_d/2 + ||f-m_d||_infinity`. Numerical suffix
samples do not establish this global identity.

Repeated product-state encoding represents each input with `r` qubits. The
quadratic readout uses the binomial factorial moment `K*(K-1)/(r*(r-1))` for
`r >= 2`; the exponential construction uses Bernstein midpoint coefficients.
All coefficients are supplied analytically, not learned. The one-copy LP is
restricted to the specified computational count-feature family, not all quantum
encodings or measurement bases. In particular, the bias of a specified
Bernstein readout is not an unavoidable error for every readout at that size.

The general joint-count representation has `(r+1)^d` coefficients and `rd`
qubits. These additive examples admit compact marginal evaluation. The
convergence table uses factorized binomial moments, **not** statevector simulation
of large registers. Explicit amplitude enumeration is separately limited to
16 qubits and includes mixed inputs and endpoint windows. The analytic proof,
not enumeration of a few windows, establishes a worst-case statement.

The variance example holds the admissible history set fixed while changing the
sampling distribution over extreme old populations. A four-branch panel has zero
variance with probability `p^4+(1-p)^4`. Its finite-sample correction estimates
population variance, not the distribution-free worst-case diameter. The
square root of corrected variance is not itself an unbiased estimator of the
population standard deviation.

## Reproduction and plotting API

Use the repository's normal NumPy/SciPy/pytest environment; no additional package
is required. From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=pra_extension/code:. \
.venv/bin/pytest -q pra_extension/tests/test_sharpness_benchmarks.py
```

The focused tests compare Kraus evolution with the explicit replacement map,
finite-prefix gaps with their finite-length formula, small-register statevectors
with direct binomial sums, factorial moments with the midpoint, a finite LP with
its matching uniform construction, and seeded four-branch statistics with their
exact distributional laws. The nonpolynomial maximum is located using its unique
derivative zero; a direct binomial grid independently checks the evaluation, not
the global proof. Random checks accept explicit seeds and do not use global RNG
state. The source does not write generated result files.

```python
from sharpness_benchmarks import convergence_rows

rows = convergence_rows(d=8, eta=0.8, beta=2.0,
                        copies=(1, 2, 4, 8, 16, 32, 64, 128, 256))
# Each row: r, qubits, error, floor, excess, maximizing_input.
```

Full repository checks and manuscript integration are separate release steps.

## Supplied-source provenance

The user-supplied archive is retained unchanged outside the repository. This
module is an adapted, parameterized implementation of its reviewed derivations;
it does not import saved numbers as experimental results.

| Supplied item | SHA-256 |
| --- | --- |
| `QELM_Sharpness_Tests.zip` | `a28882c199eca0d4b4b405f9add1c2eaddd7dbb6cf9befcc363c8a4080168cab` |
| Archive `run_sharp_tests.py` | `84884ea1875f858e26299a0e64620b4bc94170ba98718e2abb8eff85d5da856e` |

The redundant protected-memory rerun in the supplied archive is not included in
this module. These checks do not re-estimate the worst-case diameter of the
original random-reservoir cohorts, establish learning efficiency, or replace the
paper's general theorem and its assumptions.
