# Matched QRC-to-QELM Emulation: Exploratory Result

## Verdict

The supported result is a **phase-dependent two-resource complementarity**, not
a universal statement that any finite-window QELM reproduces any QRC:

> In the tested three-qubit local-loss family, fading controls the window needed
> by constructive reset replay, while an independently randomized QELM pays an
> additional accessible-feature cost. The feature-bank benefit appears in the
> medium/fast fading regimes, remains weak in the slow regime, and vanishes in
> the unitary non-fading control.

The runner's frozen automatic label is
`PILOT_SUPPORTS_TWO_RESOURCE_SCALING`. Scientifically, the more precise wording
above should be used because no asymptotic scaling law was fit.

## Final evidence package

The final analysis is `results_emulation_replicated_pilot_final/`. It uses four
fresh matched replicate blocks, four independent nested QELM banks, exact
intercept-only regularization as a validation candidate, 16 systems, 32 paired
histories per system, and untouched reset test streams.

All 896 common-suffix certificate rows pass. Predictions produced separately
from both complete history arms agree exactly, and the maximum measured
prediction disagreement is zero. The full repository suite passes with 245
tests.

## Main quantitative result

Values below are medians across the four independent blocks at `d=128`.

| regime | remaining suffix witness `D2(d)/D2(0)` | oracle NRMSE reduction | random-bank `R=4` NRMSE | `R=1 -> R=4` gain | `R=4` gain over linear delay |
|---|---:|---:|---:|---:|---:|
| unitary, `gamma=0` | 0.965 | 12.5% | 1.019 | approximately 0% | approximately 0% |
| slow, `gamma=0.05` | 0.0323 | 97.2% | 0.727 | 15.4% | 11.5% |
| medium, `gamma=0.2` | `2.72e-5` | 99.998% | 0.303 | 43.4% | 45.5% |
| fast, `gamma=0.5` | `1.44e-10` | approximately 100% | 0.0936 | 65.7% | 74.8% |

The state-level trace-distance fraction after the common 128-input suffix is
1.000, 0.0266, `2.32e-5`, and `1.13e-10`, respectively. Within every one of
the four matched blocks, increasing `gamma` orders the remaining observable
memory perfectly (Spearman `-1`).

The first registered window where the median observable suffix witness falls
below 10% is:

- no finite tested window for `gamma=0`;
- `d=128` for `gamma=0.05`;
- `d=32` for `gamma=0.2`;
- `d=16` for `gamma=0.5`.

This is the clean constructive result: faster forgetting requires a shorter
reset-replay window.

## Why the second resource is real but phase dependent

At `d=128`, enlarging the independently randomized bank from one to four QELMs
improves NRMSE in 11 of 12 dissipative systems under the frozen 15% gate. The
median gain grows sharply with fading: 15.4% (slow), 43.4% (medium), and 65.7%
(fast). In the unitary systems the feature gain is numerically zero because
validation often chooses the exact intercept-only limit.

The slow regime is the informative bottleneck. Same-Hamiltonian reset replay is
already accurate there, but a four-reservoir random bank remains at NRMSE 0.727.
Thus window sufficiency does not imply accessible-feature sufficiency. This is
the empirical separation between memory truncation and representation access.

## What is not established

- Not every QRC has fading memory; the unitary control remains non-fading.
- An arbitrary fixed finite-window random QELM does not always emulate the QRC.
- The experiment uses a 33-symbol input alphabet, exact simulation, three
  qubits, local loss, and 18 one-/two-body Pauli outputs.
- Beating a linear delay readout is not quantum advantage. A dimension- and
  training-matched nonlinear classical random-feature control is still needed.
- Teacher and student dissipation are matched. A teacher-gamma by student-gamma
  cross experiment is needed to separate target forgetting from student
  expressivity.
- The held-out bank-seed phase is stronger than the discovery run, but the work
  remains exploratory rather than a formal external preregistration.

## Paper-facing claim

Use:

> Finite-window emulation of a fading QRC exhibits two distinct resource
> requirements: a history window set by forgetting and an accessible feature
> budget set by the student representation. These costs separate empirically,
> and the separation disappears at the non-fading boundary.

Do not use: "a QELM can always approximate any QRC."

## Next decisive experiment

Run a crossed `teacher gamma x student gamma` surface with independent banks and
add a matched nonlinear classical random-feature baseline. Then repeat the
frozen phase test across qubit counts and at least one second dissipator. That
is the smallest extension that can turn this robust proof of concept into a
family-level paper claim.
