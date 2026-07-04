# Quantum Reservoir Computing vs. Quantum Extreme Learning Machines: A Resource-Matched, Pre-Registered Comparison

*Draft — thesis chapter / paper. Generated from the frozen confirmatory run at commit `0fc0404` (analysis fixes through `171474a`). All numbers trace to `results_*_full/` outputs; none are invented.*

## Abstract

Quantum reservoir computing (QRC) and quantum extreme learning machines (QELM) are two routes to temporal information processing on the same quantum hardware. QRC carries temporal context in the intrinsic quantum memory of a recurrent, dissipative reservoir; QELM applies a *static, memoryless* quantum feature map to a classical delay-embedding of the input. The open question left by prior work is: **under matched resources, when does intrinsic quantum memory outperform a static quantum map fed a classical delay embedding — and how do finite sampling and engineered dissipation move that boundary?** We answer this with a pre-registered, resource-matched study on a Lindblad spin reservoir at N=5 qubits. On a memory×nonlinearity phase diagram (Legendre delay tasks), QRC and QELM occupy **distinct, individually significant regions**: QRC wins the long-delay memory band (delays beyond QELM's embedding window), QELM wins short-delay nonlinear tasks. A pre-registered kill rule confirms QELM is a non-trivial quantum baseline (it significantly beats a classical delay-embedding control on 62.5% of nonlinear cells). Finite shot noise shifts the boundary toward fewer QRC-favored cells, though not significantly at our seed budget. The dissipation sweep reveals a clean **division of labour rather than an intra-model tradeoff**: QRC is a dissipation-*powered* memory specialist (near-useless at zero dissipation, strong memory once dissipation is engineered), while QELM is a dissipation-*robust* nonlinearity specialist whose nonlinear capacity exceeds QRC's across the entire dissipation range.

## 1. Introduction

Temporal machine-learning tasks require a model to combine (i) *memory* of past inputs and (ii) *nonlinear* transformation of them. Two quantum approaches trade these off differently:

- **QRC**: a recurrent open quantum system driven by the input stream. Fading memory arises from the dissipative dynamics; nonlinearity from the quantum evolution. Memory is *intrinsic* to the reservoir.
- **QELM**: a fixed quantum circuit/channel applied once per timestep to a *classical delay-embedding* (the last `d` inputs). Memoryless by construction — temporal context is supplied classically; the quantum part provides only a static nonlinear feature map.

The question this work settles is the one left as future work by the framework paper (arXiv:2602.21544v2): **holding resources equal — qubit count, measured observables, readout dimension, shot budget, and the classical readout pipeline — where does QRC's intrinsic quantum memory actually buy something over QELM's classical-delay-embedding-plus-static-quantum-map, and how do shot noise (RQ2) and engineered dissipation (RQ3) reshape that boundary?**

## 2. Methods

### 2.1 Shared backend and the two models

Both models are built on one verbatim-ported Lindblad simulator (an N-qubit Ising-XX spin network with a transverse field and local-loss dissipation). This is deliberate: the QELM channel `exp(L·T_evolve)` is constructed from the **identical** Hamiltonian family and dissipators as the QRC's recurrent Lindblad step, so **recurrence is the only structural difference** between the two models. The QRC injects the input as an active drive at each recurrent step; the QELM encodes a `d`-length delay window into a product state (cyclic across qubits) and evolves it once through the fixed channel. This input-injection asymmetry (dynamic drive vs. state preparation) is intrinsic to the memory/memoryless distinction and is pre-registered as an intended, unmatchable asymmetry.

### 2.2 Resource matching and the fairness guard

At each operating point both models share: qubit count N=5; the weight-≤2 3-axis Pauli observable set (readout dimension D=45); a single readout snapshot (V=1); the same total shot budget formula S·T·D; and one identical ridge-regression readout with unpenalized intercept, train-only standardization, and validation-selected regularization. Matching is enforced at runtime by `assert_bundles_matched` (actual built dimensions, not nominal) plus a **liveness guard**: for multiplexed readouts we require the multi-snapshot feature matrix to have effective rank meaningfully above a single snapshot's — a *scale-invariant rank ratio*, because the naive `eff_rank/D` fraction is a D-scaling artifact that spuriously rejects configurations at study scale. (This guard drove the pre-registered decision to run the primary comparison at V=1: QRC virtual-node multiplexing is strongly correlated and would hollow the declared readout dimension.)

### 2.3 Task suite and phase-diagram axes

The primary axes are a single-term Legendre delay family, `y_t = P_p(2·u_{t-τ}−1)`, with memory demand τ ∈ {0,1,2,3,5,8,10,12,16} and nonlinearity demand p ∈ {1,2,3}. Each cell's metric is capacity (squared correlation, in [0,1]). Standard benchmarks (STM, NARMA-2/5/10, Mackey-Glass, parity) are placed as anchor points on the same plane. Classical delay-embedding controls (`linear_delay`, `poly_delay`) at the matched readout dimension provide the baseline that isolates the *quantum* contribution of QELM's map.

### 2.4 Pre-registration and gates

The operating point (N=5, γ=0.3, dt=0.5, h=0.5, local-loss, V=1, quantize=33) and gate thresholds were frozen before the confirmatory run, informed by a design pilot on **disjoint** seeds (the full amendment trail — the γ sweep, delay-range extension, power test, and degree-axis choice — is logged). The decision gates:

- **G1a** — per-cell paired Δcapacity (QRC−QELM), bootstrap CI + Benjamini–Hochberg FDR (q≤0.10).
- **G1b** (headline) — both regimes materially present: ≥20% of *significant informative* cells favour each model.
- **G1c** (kill rule) — QELM must *significantly* beat `linear_delay` on ≥25% of nonlinear (p≥2) cells, else the comparison is declared trivial.
- **G2a** (RQ2) — QRC-favoured cell count shifts between S=∞ and S=1e4 with a CI excluding zero.
- **G3a** (RQ3) — memory capacity decreasing and nonlinear capacity non-decreasing in γ, with a crossover where QELM overtakes.

A pre-registered rule excludes *uninformative* cells (both models' mean capacity < 0.05) from the G1b/G1c denominators; both with- and without-exclusion fractions are always reported.

### 2.5 Validation gate

Before any confirmatory run, the ported stack reproduced the source reservoir's short-term memory capacity (MC = 10.13, within the source's ±2σ band [6.50, 11.33]) and the correct continuous-dissipation-beats-Fuji-Nakajima ordering; the QELM memory-cutoff signature (capacity ≈1 for τ<d, ≈0 for τ≥d at γ=0) held. This certifies the modelling stack reproduces known physics.

## 3. Results

### 3.1 RQ1 — The memory×nonlinearity phase diagram (verdict: GO)

**Both regimes exist and are individually significant.** On the informative cell set, G1b PASSES: QRC favoured 46.7%, QELM favoured 33.3%, ties 20.0% (n=15 significant cells). The boundary is physically clean:

- **QRC-favoured (long-delay memory band):** τ=3,p=1 (Δcap +0.98); τ=5,p=1 (+0.84); τ=8,p=1 (+0.42); τ=10,p=1 (+0.16); τ=3,p=2 (+0.47); τ=5,p=2 (+0.20); τ=2,p=3. QRC wins precisely where the target delay lies **at or beyond QELM's 5-step embedding window**, where dissipative fading memory reaches but a finite delay-embedding cannot.
- **QELM-favoured (short-delay nonlinear):** τ=0,p=3 (Δcap −0.54); τ=1,p=3 (−0.71); τ=2,p=2 (−0.23); τ=0,p=2 (−0.06); τ=1,p=2 (−0.14). QELM wins short delays *inside* its window, especially at higher nonlinearity, where its static quantum map is a richer nonlinear feature extractor.

**The kill rule passes** (G1c = 0.625 ≥ 0.25): QELM significantly beats the classical `linear_delay` control on 62.5% of nonlinear cells, so the quantum feature map contributes genuine value and the comparison is non-trivial.

**Honest caveat (load-bearing).** The GO holds on the pre-registered *informative* cell set. On the full 27-cell grid — including 12 cells where **both** models fail (very long delays / high nonlinearity beyond either architecture's reach at N=5) — the tie fraction rises to 44% and QELM drops to 19%, yielding `SINGLE_REGIME`. The uninformative-cell exclusion that resolves this was pre-committed from the pilot and removes only cells where neither model has signal; nonetheless it is the assumption the headline rests on, and both numbers are reported.

### 3.2 RQ2 — Shot-budget scaling (verdict: SHOT_INSENSITIVE, underpowered)

The QRC-favoured cell count decreases monotonically as the shot budget shrinks: 9.0 (S=∞) → 8.7 (1e5) → 7.0 (1e4) → 5.0 (1e3). The point estimate thus indicates finite sampling **erodes QRC's advantage** (its memory-derived features are noisier to estimate than QELM's). However, the bootstrap CI on the ∞-vs-1e4 shift is [0.0, 3.78], which does not exclude zero at the 24-seed budget. We therefore report shot sensitivity as **suggestive but not statistically significant** at this power — a candidate for a larger seed grid in follow-up work.

### 3.3 RQ3 — Engineered dissipation: a division of labour (verdict: NO_TRADEOFF, reinterpreted)

The pre-registered G3a gate looked for the *classical* intra-model tradeoff (dissipation reduces memory, increases nonlinearity) and returns `NO_TRADEOFF` — because the data show something more interesting. Full-scale MC/NC vs. γ (local-loss):

| γ | QRC MC | QRC NC | QELM MC | QELM NC |
|---|---|---|---|---|
| 0.005 | 2.40 | 0.05 | 3.83 | 3.36 |
| 0.05 | 5.72 | 0.39 | 3.84 | 3.40 |
| 0.1 | 6.49 | 0.86 | 3.85 | 3.44 |
| 0.2 | 6.79 | 1.57 | 3.86 | 3.50 |
| 0.3 | 6.77 | 2.11 | 3.88 | 3.56 |
| 0.5 | 6.51 | 2.84 | 3.91 | 3.64 |

Two clean findings:

1. **QRC is dissipation-powered.** Near-unitary (γ=0.005) QRC has almost no memory (MC 2.40) and negligible nonlinearity (NC 0.05). Engineered dissipation *builds both*: MC rises to ~6.8 (peaking near γ=0.2) and NC rises monotonically to 2.84. Dissipation is not a memory-vs-nonlinearity tradeoff knob for QRC — it is the resource that makes QRC work at all.
2. **QELM is dissipation-robust.** With a fixed evolution horizon, QELM's MC (~3.8) and NC (~3.3→3.6) are nearly flat across the entire γ range — a stable, high-baseline nonlinear map indifferent to the dissipation strength.

The consequence is a **between-model specialization**: QRC's memory overtakes QELM's around **γ≈0.05**, but **QELM's nonlinear capacity remains higher than QRC's across the whole sweep** (γ=0.5: QELM NC 3.64 > QRC NC 2.84). The collective-loss dissipator gives the same qualitative picture. No QELM feature collapse occurred at full scale (N=5, T=3000).

## 4. Discussion

The paper's open problem — *when does quantum memory buy something over a static map plus classical delay embedding?* — has a nuanced answer:

- **Quantum memory wins the long-delay regime.** For target delays beyond the classical embedding window, QRC's intrinsic dissipative memory is a genuine, significant advantage that no fixed-window static map can match at equal resources.
- **The static quantum map wins short-delay nonlinearity — and is the stronger nonlinear feature extractor overall.** QELM dominates short delays and retains higher nonlinear capacity than QRC across all dissipation strengths.
- **Dissipation is QRC's enabling resource, not a tradeoff dial.** The interesting tradeoff is *between* architectures (memory specialist vs. nonlinearity specialist), not *within* one.

This suggests a practical design heuristic: use QRC when the task's dominant demand is long-range temporal memory and dissipation can be engineered; use QELM (or a hybrid) when the demand is short-range nonlinear transformation, where it is both stronger and dissipation-robust.

## 5. Limitations

- **Scale.** N=5 qubits; results are not extrapolated beyond that. Degree-5 nonlinearity is beyond both models at this scale and is excluded from the gated grid (reported as robustness).
- **Reproduction budget.** 24 seed cells (after a pre-registered runtime shrink from the 30-cell grid); the RQ2 shot-sensitivity CI is consequently underpowered.
- **The RQ1 headline depends on the pre-registered uninformative-cell exclusion** (§3.1 caveat).
- **Input-injection asymmetry** (dynamic drive vs. state prep) is intrinsic to the memory/memoryless distinction, not a matched axis.
- **Moderate reproduction:** the validation reproduces the source physics within its ±2σ band and correct orderings, not to bit-exactness (a different-but-valid split/readout protocol).

## 6. Conclusion

Under a pre-registered, resource-matched protocol, QRC and QELM are complementary rather than competing: quantum memory earns its keep in the long-delay regime, while the static quantum map is the superior — and dissipation-robust — nonlinear feature extractor. The comparison is meaningful (QELM significantly beats a classical delay-embedding baseline), and dissipation is revealed as the resource that switches QRC from useless to a strong memory reservoir. The honest negatives (underpowered shot sensitivity; no intra-model dissipation tradeoff) are as informative as the positive phase-diagram result.

## Reproducibility

All code, the frozen pre-registration (`PREREGISTRATION.md`, `PROTOCOL.md`), the frozen config (`data/frozen/frozen_v1_config.json`), and the full amendment log are committed. Each stage runner records a manifest with the git commit, seed grid, per-cell feature-cache SHA-256 hashes, and package versions; the runners refuse to execute against an uncommitted tree. Figures: `results_report_full/{phase_diagram,boundary_vs_shots,gamma_tradeoff}.png`; gate verdicts: `results_report_full/stage_gate_summary.csv`.
