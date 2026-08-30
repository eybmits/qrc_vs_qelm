# Literature Positioning: Canonical QRC-to-QELM Emulation

**Audit date:** 30 August 2026. This is a bounded positioning audit, not proof
of priority over every paper or unpublished result.

## What is already established

| Established result | Closest source | Consequence for our claims |
|---|---|---|
| A QELM is the memoryless/feedforward member of the QRC family. | Innocenti et al., *Potential and limitations of quantum extreme learning machines*, Communications Physics 6, 118 (2023), [article](https://www.nature.com/articles/s42005-023-01233-w). | The basic QRC-versus-QELM distinction is not new. |
| Finite-dimensional QRC fading-memory and echo-state conditions can be analyzed rigorously. | Martinez-Pena and Ortega, *Quantum reservoir computing in finite dimensions*, Physical Review E 107, 035306 (2023), [article](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.107.035306). | We must not present fading memory or contractive forgetting as our theorem. |
| Continuous fading-memory filters admit finite-memory approximation. | Grigoryeva and Ortega, *Differentiable reservoir computing*, JMLR 20, 179 (2019), [article](https://www.jmlr.org/papers/v20/19-150.html). | Finite truncation of a fading-memory filter is established theory. |
| Artificially restricting QRC replay history can reduce cost and alter task performance/nonlinearity. | Cindrak et al., *Enhancing the performance of quantum reservoir computing and solving the time-complexity problem by artificial memory restriction*, Physical Review Research 6, 013051 (2024), [article](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.013051). | Reset-and-replay is a known memory-restricted QRC control, not canonical QELM. |
| A canonical time-delayed QELM resets each row, encodes several delayed inputs in parallel on separate qubits, and applies one fixed reservoir evolution. | Kawanabe et al., *Efficient time-series prediction on NISQ devices via time-delayed quantum extreme learning machine*, arXiv:2602.21544v2 (2026), [preprint](https://arxiv.org/abs/2602.21544). | This architecture itself is not new. Our student faithfully matches its reset/parallel one-lag-per-qubit structure while explicitly generalizing to a fixed six-qubit register and XYZ readout. |
| QELM encoding fixes the nonlinear input feature library; the channel mixes it linearly; restricted measurements, rank, conditioning, and decodability control accessibility. | Gross and Rieser, *Theory and interpretability of Quantum Extreme Learning Machines: a Pauli-transfer matrix approach*, arXiv:2602.18377v3 (2026), [preprint](https://arxiv.org/abs/2602.18377). | “Accessible feature span matters” is known. Our rank/projection diagnostics validate a mechanism rather than establish it from scratch. |
| Parallel repeated angle encoding and a linear quantum-feature readout implement multivariate Bernstein polynomials, uniformly approximate continuous functions, and admit non-asymptotic circuit bounds. | Goto, Tran, and Nakajima, *Universal Approximation Property of Quantum Machine Learning Models in Quantum-Enhanced Feature Spaces*, Physical Review Letters 127, 090506 (2021), [article](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.090506); Yu et al., *Non-asymptotic Approximation Error Bounds of Parameterized Quantum Circuits*, NeurIPS 37 (2024), [article](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b360fd42ced877429882a2a68b4a4343-Abstract-Conference.html). | The Bernstein quantum feature construction, its universality, and generic approximation bounds are prior art. The block-Hamming implementation is a coarse-grained realization, not a new approximation mechanism. |
| Dissipative QRC families can universally approximate fading-memory maps under stated conditions. | Sannia et al., *Dissipation as a resource for Quantum Reservoir Computing* (2022), [preprint](https://arxiv.org/abs/2212.12078); Chen et al., *Temporal Information Processing on Noisy Quantum Computers*, Physical Review Applied 14, 024065 (2020), [article](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.14.024065). | Existing QRC universality does not imply that one fixed finite QELM approximates every QRC. |

## The narrower open angle tested here

The searched sources separately establish finite-window TD-QELM, QRC fading
memory and finite-memory approximation, Bernstein quantum-feature universality,
QELM feature accessibility, and decoherence-free sectors. The remaining paper
contribution is their observable-level integration: an exact common-suffix
minimax criterion, a contractive QRC-to-growing-reset-QELM resource bound, a
protected-memory obstruction, and the following matched
QRC-teacher/QELM-student experiment:

1. use the complete observable trajectory of a recurrent QRC as the supervised
   target rather than using an external NARMA or chaotic-flow target;
2. sample multiple old prefixes followed by identical suffixes;
3. decompose finite-window emulation error exactly into a student-independent
   history term and a student-dependent representation term;
4. vary teacher forgetting, student window, and nested accessible feature
   budget independently;
5. classify held-out cells as history-impossible, emulated,
   representation-limited, unresolved, or degenerate;
6. compare the canonical parallel TD-QELM with reset-replay, raw delay,
   quadratic delay, and dimension-matched classical random features.

The paper's contribution is therefore the **two-sided observable approximation
framework**. For scalar outputs, the optimal unrestricted suffix error is
`D_d/2`. Under incremental contraction, the growing Bernstein QELM has the
explicit sum of history, representation, and shot terms. Protected memory keeps
`D_d` nonzero. The paper does not invent TD-QELM, fading memory, Bernstein
quantum features, conditional variance, decoherence-free subspaces, or QELM
feature decodability.

## Strongest safe priority claims

> The paper combines an exact scalar common-suffix criterion with a constructive
> contractive-QRC-to-reset-QELM error and resource bound, then realizes the
> nonfading side with one fixed protected quantum teacher whose output has a
> strictly positive floor for every finite suffix-only predictor.

Separately, we are unaware of an earlier matched held-out QRC-as-teacher study
that estimates omitted-prefix risk independently from suffix
representation/readout error. The exact claim does not concern generic
bounded-memory impossibility, all recurrent QRCs, stateful or prefix-augmented
students, unbounded windows, or the established ingredients individually. The
empirical phase map remains bounded by its teacher families, input laws,
observable projection, exact-simulation setting, windows, feature budgets, and
error tolerance. A literature search supports positioning but cannot prove
absolute priority.

## Claims this study must not make

- every QELM can approximate every QRC;
- that the Bernstein quantum feature construction or its continuous-function
  universality is newly introduced here;
- all recurrent QRCs defeat finite-window QELMs;
- the theorem covers stateful, prefix-augmented, or branch-informed QELMs;
- the elementary common-suffix inequality or risk decomposition is a new
  mathematical theorem;
- QELM has a quantum advantage over classical feature models;
- a result for measured observables is full quantum-state emulation;
- exact simulation establishes NISQ hardware robustness.
