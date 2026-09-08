# Notation audit

Scope: the complete current journal and conference manuscripts, their mathematical
appendices and numerical methods, and all three retained figures. Archived draft
figures and frozen experiment/configuration files are provenance, not the paper's
notation authority; they are not rewritten.

## Literature conventions and editorial choice

There is no single standard alphabet shared by QRC, QELM, approximation theory,
and control theory. The defensible standard is to preserve familiar mathematical
roles, define each quantity near first use, and keep its meaning stable.

- [Innocenti et al. (2023), notation and theoretical background](https://www.nature.com/articles/s42005-023-01233-w)
  uses lowercase outputs, density operators, quantum channels, observables and
  linear readouts. We keep those roles, without copying its training/test indices.
- [Fujii and Nakajima (2017), Sections III-IV](https://link.aps.org/accepted/10.1103/PhysRevApplied.8.024030)
  uses density operator rho, Hamiltonian H, observable expectations Tr(O rho),
  discrete input indices and physical evolution intervals. Its inputs are s_k;
  that letter is not compulsory. We keep this paper's a_t and explicitly separate
  physical time from the discrete step.
- [Grigoryeva and Ortega (2018), Sections 1-2](https://jmlr.org/papers/volume19/18-020/18-020.pdf)
  uses y_t for outputs and introduces the domains, norms and time indices before
  its results. Its input letter z differs from the QRC reference. We retain a_t
  throughout instead of switching alphabets between disciplines.

The choices below are a consistent house notation informed by those references,
not a claim that every individual letter is universally standardized. The existing
citation keys and mathematical claims remain unchanged.

## Core vocabulary

| Symbol | One meaning and explanation location |
|---|---|
| a, a_t | One input and the input at discrete step t; Section 2. Scalar a in numerical target/encoder; the protected construction explicitly uses a=(u,q). |
| t; k | Discrete current step; displayed-step index in Figure 1. Neither is continuous evolution time. |
| h, h' | Admissible full input histories; Section 2. A prime means a second history, not a derivative. |
| d | Number of inputs in the visible window; Section 2 and every resource figure. |
| S_d(h); s | Last d inputs of h; a concrete suffix/window in the proofs. S_d is only abbreviated in the schematic. |
| rho_t; rho, sigma | Carried quantum state; generic density operators. Section 2 and trace-distance definition in Section 4. |
| Phi_a, Phi_s | Quantum update driven by input a; ordered composition along suffix s. Section 2 and Appendix window proof. |
| C (calligraphic) | The fixed declared output map, never a second unrelated C matrix. Section 2. |
| y_t, y(h) | Target QRC output at a step or for a history. Scalar or finite vector as explicitly stated; uppercase Y remains only a Pauli operator. |
| widehat y_t, widehat y_s | Emulator prediction of that output, at a step or shared by a suffix group. Section 2 and panel identity. |
| f | A window prediction function; the same symbol in the unrestricted optimum and QELM theorem. Section 2-3. |
| Q_d (calligraphic) | Growing family of exact-expectation QELM prediction functions f at fixed d. Plain Q is never a predictor. Section 3. |
| O_j, j=1,...,m | Declared target observables and output dimension m. Section 2; upper bank of Figure 1. O alone is a scalar-output observable. |
| P_j, j=1,...,B | Measured QELM feature observables, with j including checkpoints. Section 2; lower bank of Figure 1. |
| E(S_d) | Window encoding operation in the reset architecture; Section 2 and Figure 1. |
| B | Measured-feature count, not qubit count. Sections 2/5 and Figures 2-3. |
| Q | Number of circuit repetitions/shots. Total per-input budget in experiments; repeated joint readouts in the explicit construction. Defined locally in both. |
| D_d | Largest output gap between histories sharing their last d inputs; Section 3. Not a measured NRMSE. |
| epsilon (varepsilon glyph) | Chosen absolute-error tolerance or arbitrary approximation accuracy. Not D_d/2 itself and not the experimental normalized 0.10 target. |
| d_epsilon | Shortest window reaching that tolerance; infinity if no finite window suffices. Section 4. |
| eta | Uniform state-contraction factor, 0<=eta<1; Section 4. |
| r_vis | Joint spectral radius of the reachable-observable updates; vis means output-visible. Section 4. |

## Proof-only vocabulary

| Symbol | Meaning and retained reason |
|---|---|
| Y_d(s) (calligraphic), R_d | Compatible-output set and its worst-case enclosing-ball radius. Needed for the vector extension; distinct from scalar D_d/2. |
| z | Candidate center of that ball, not an input. |
| y_-(s), y_+(s) | Extreme compatible scalar outputs in the proof. The former Figure 1 midpoint inset is removed. Infimum/supremum agree with min/max over reachable-state closure. |
| m_d | Midpoint of that compatible scalar range. Retained to distinguish optimal midpoint prediction from replay. |
| R (calligraphic), bar rho | Compact reachable-state closure and a fixed reachable reference state. Not an empirical average state. |
| g_d; rho_t^(d) | Replay output function and replay state from bar rho using the last d inputs. No breve/tilde decoration. |
| e | Uniform emulator error in the vector necessity argument. No longer also a word length. |
| Delta, V | Reachable state-difference set and its real span. Delta is distinct from Lindblad dissipator D (calligraphic). |
| N (calligraphic), W, pi | Invisible subspace, quotient, and quotient projection. Necessary types in the spectral proof, defined together. |
| v; x | State difference before quotient; its quotient-space counterpart. |
| A_a, A_s, A_w | Induced update and ordered products for suffix/future words; not a new physical channel. |
| w, \|w\|; ell, d_0 | Future input word and its length; second length for submultiplicativity; one contracting length. All local and explicit. |
| B (calligraphic) | Symmetric reachable body, not the plain feature count B. Replaces a long reach subscript. |
| c_d | Largest length-d product norm. No longer a_d, which resembles the input. |
| bar C (calligraphic) | The induced output map on the quotient. The bar is necessary to distinguish its domain; definition links it to C. |
| kappa_-, kappa_+ | Positive norm-equivalence constants, not output extrema. |
| r_-, r_+ | Temporary rates below/above r_vis for the asymptotic window proof. |
| r | Positive integer Bernstein repetition count, not r_vis. Fixed before its use. |
| bold a, bold a'; a_i | Two d-coordinate windows and a coordinate. Bold font is used only where the vector itself matters. |
| bold k, K_i | Deterministic Bernstein grid multi-index and random Hamming-weight counts. Capital K_i is random; lowercase grid entries are not. |
| L_in, M | Input Lipschitz constant and scalar observable operator norm. Both defined before the resource bounds. |
| f, widehat f | Fixed-parameter exact Bernstein expectation and its shot estimate; d and r are fixed in the preceding sentence, not stacked onto every formula. |
| f^mid, widehat f^mid | Separate midpoint-coefficient construction and estimate. The short mid label prevents conflating it with the implemented replay construction. |
| epsilon_hist, epsilon_rep, epsilon_shot | Named upper-bound contributions of replay history, representation, and measurement. No additional total-error alias. These are not the fitted panel's residual decomposition. |
| zeta | Failure probability, 0<zeta<1. Guarantees concern each fixed prediction, not all histories simultaneously. |
| delta; delta_d, delta_0 | Protected output gap; signed finite-horizon gap and its initial value under imperfect protection. |
| epsilon_prot | Largest one-step protection defect. The descriptive suffix avoids mistaking it for an accuracy target. |
| y_sk, bar y_s | Target output for suffix s/prefix branch k, and its branch mean. The prediction is consistently widehat y_s. |
| n_pre | Number of prefix branches in the finite-branch correction; registered value 4. |
| b=0,1 | Balanced pair label in the randomized lower-bound calculation. Not a feature budget. |

## Physical/statistical notation and operations

| Symbol | Meaning and convention |
|---|---|
| I, X, Y, Z; X_i, etc. | Identity/Pauli operators; subscript is qubit index. XX means X tensor X. Defined before the singlet equation. |
| psi^-, P_-; P | Antisymmetric singlet and its projector; general conserved observable in the obstruction. |
| u, q | Drive amplitude and binary pulse flag only in protected input a=(u,q). No u/a switching in the scalar target or encoder. |
| H(u), J; H_T(a), J_ij | Protected Hamiltonian/coupling and numerical target Hamiltonian/pair couplings. Uniform draws are described in words, without a new distribution symbol. |
| T, N_T, gamma_T | T means target; target qubit count and loss rate. Figure 2 also defines gamma_T and Delta t beside scaled context. |
| tau; Delta t | Continuous physical evolution time; physical duration of one discrete input update. Replaces unexplained t_evol. |
| dot rho | Derivative d rho/d tau inside a channel update, not a derivative in discrete t. |
| sigma_i^-, L_c, D[L] | Lowering operator \|0><1\| on qubit i, collective jump, Lindblad dissipator. Defined in the numerical methods. |
| R_x^(1)(theta), theta | Rotation of the first qubit about x and its angle. Explicit exponential definition precedes numerical use. |
| lambda | Affine ridge penalty; lambda=infinity is explicitly the intercept-only model. |
| Tr, angle brackets | Trace and observable expectations, explained in the model/figure context. |
| dagger; Phi_a^dagger | Hermitian adjoint of operators; channel's Heisenberg adjoint acting on observables. Related, not interchangeable types. |
| [.,.], {.,.} | Commutator/anticommutator in the Lindblad model, not unnamed extra operators. |
| norm, trace norm, Euclidean norm, infinity norm | Fixed output norm; trace norm; empirical Euclidean norm; locally specified uniform/function or operator norm. Domains disambiguate the latter. |
| B-norm, obs-norm | Reachable-body and observable norms, defined only in the spectral proof. |
| E, Pr (blackboard/probability notation) | Expectation and probability over prediction/shot randomness. Indexed E_s,k and E_s are explicitly empirical equal-weight averages in the panel identity. |
| real span, conv, closure, sup/inf, min/max, log | Standard operations, not new model parameters. Supremum/infimum are retained where attainment is not guaranteed. |

Generic indices i,j,k are locally bound; primes indicate alternatives. Standard
number sets, imaginary i, ket/bra notation, tensor products and arithmetic keep
their usual mathematical meanings. Removing those conventions would add prose
without improving the intended readers' understanding.

## Invariants and verification scope

- Scalar optimum, vector bounds, free-concatenation/continuity assumptions,
  nonattainment, spectral equivalence, protected gap and all numerical constants
  are retained. No approximation/measurement guarantee is strengthened.
- Figure 1 label changes only; source samples, scales, palette and geometry stay
  fixed. Figures 2 and 3 keep their existing labels and numerical artifacts.
- Regression tests guard notation consistency, all three terms of the normalized
  panel identity, exact resource-bound formulas and the separate midpoint scope.
- Both compiled versions now meet the twelve-page limit after appendix compression.
  Section 4 separates observable forgetting, required window length
  and protected memory, preserving equations (4)--(11), local symbol definitions
  and automatic appendix references. All changed pages were visually reviewed, with no overfull boxes
  or unresolved references. Fonts, margins and figure sizes are unchanged.
  The conference twelve-page gate passes and its current source ZIP rebuilds
  identically. Regression checks enforce the cap for both versions.
  The unused imperfect-protection aside and its one-off symbols are removed.
  Vector bounds and balanced-pair risk are retained in concise prose;
  appendix equation numbers and their references update automatically.
  Section 2 retains three unnumbered displays. Body text, theorem labels and
  caption prose now use normal weight; section headings, panel references and
  mathematical vector notation are preserved. Section 5
  separates models, history, feature/classical comparisons and measurements;
  squared-error and residual-error qualifications remain explicit.
- All 79 checked CSV, frozen and seed-related artifacts, both Figure 1 sample
  hashes and the four Figure 2/3 PDF/PNG artifacts match the pre-audit state.
  The separate visual-review and release manifests bind the delivered artifacts.
  This audit does not claim independent experimental reproduction or submission
  readiness.
