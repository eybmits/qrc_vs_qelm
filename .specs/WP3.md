# WP3 — Unified model API, resource matching, shot noise

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Core is at
`src/qrc_vs_qelm/core/` (WP1), QELM at `src/qrc_vs_qelm/models/qelm.py` (WP2).
You may READ `/Users/superposition/Coding/qrc_dataset_profiler/src/qrc_dataset_profiler/reservoir.py`
(shot-noise source, line ~82).

## Design rule (fairness-critical)

Both model builders return **full-length** feature matrices — one row per input
timestep, including washout/warm-up rows. Validity masking happens ONCE, in the
shared pipeline (WP4), via a single valid-index set. No model-specific row dropping
inside builders. For the QRC this means calling `Reservoir.run(..., washout=0)`.

## Part 0 — WP2 hardening (do first; from dual review, all "narrow but fix before building on the channel")

a. `models/qelm.py::QELMChannel.from_terms` — validate `H_static.shape == (2**n_qubits,
   2**n_qubits)` and each jump operator likewise; raise `ValueError` with a clear
   message (currently a mismatch surfaces as an opaque matmul error deep in `features()`).
b. `models/qelm.py::QELMChannel` — take a defensive `.copy()` of `L_base`/`propagators`
   on construction so in-place mutation of the returned arrays can't corrupt the
   "precomputed once" channel across pipeline stages.
c. `d > n_qubits` silently drops tail lags in cyclic encoding — this would silently
   lose delay information in a resource-matching sweep. Enforce `1 <= d <= n_qubits`
   at the sanctioned API boundary: `QELMConfig.__post_init__` raises `ValueError` if
   violated (the frozen grid only uses d ∈ {1,3,N}, so this closes an unreachable-by-
   design hole loudly rather than silently).
d. Add a `build_qelm_features(..., gamma=0.0, ...)` end-to-end test asserting the
   public zero-gamma path equals a manually-built unitary channel (closes the coverage
   gap where the empty-jumps branch is only hit indirectly).

## Deliverables

1. `src/qrc_vs_qelm/models/api.py`:
   - `quantize_inputs(u: np.ndarray, levels: int) -> np.ndarray` — uniform grid on
     [0,1]: `round(u*(levels-1))/(levels-1)`; `levels=0` or `None` → passthrough copy.
     Applied IDENTICALLY to both models' drive/encoding (targets are always computed
     from unquantized inputs — that happens in WP4, not here).
   - Frozen dataclasses (both with `n_qubits, J_s, h, gamma, dissipator, quantize_levels`):
     - `QRCConfig(dt, h_in, n_virtual, ...)` — recurrent model.
     - `QELMConfig(d, T_evolve, n_snapshots, ...)` — static model.
   - `@dataclass FeatureBundle`: `features: np.ndarray` (T, D), `model: str`,
     `config: dict`, `couplings: np.ndarray` (the J matrix actually used),
     `n_obs: int`, `n_multiplex: int` (V), `first_valid: int` (0 for QRC,
     d-1 for QELM).
   - `build_qrc_features(inputs, cfg: QRCConfig, coupling_seed: int, observables=None) -> FeatureBundle`:
     J = `random_couplings(n, J_s, default_rng(coupling_seed))`;
     H_static = `ising_xx_hamiltonian(J, h, n)`; H_drive = `cfg.h_in * transverse_drive(n)`;
     jumps from `core.dissipators` by name (empty list if gamma == 0);
     reservoir = `LindbladReservoir.from_terms(...)` with `cfg.dt` and propagator
     caching over the quantized input levels; features =
     `run(quantize_inputs(inputs, L), observables, washout=0, n_virtual=cfg.n_virtual)`.
   - `build_features(cfg, inputs, coupling_seed, observables=None) -> FeatureBundle`
     — dispatch on config type (QRCConfig → QRC, QELMConfig → WP2's builder, wrapped
     to return a FeatureBundle; pass `quantize_inputs`-processed inputs to QELM too).
   - Default observables (both paths): `core.readout.pauli_observables` axes x/y/z,
     max_weight 2.
2. `src/qrc_vs_qelm/matching.py`:
   - `class MatchingError(ValueError)`.
   - `assert_matched(qrc_cfg: QRCConfig, qelm_cfg: QELMConfig, observables=None) -> None`
     — raises MatchingError naming every violated axis: n_qubits; J_s; h; gamma;
     dissipator; quantize_levels; multiplexing (n_virtual == n_snapshots); readout
     dimension D = V·n_obs equal on both sides.
   - `physical_shot_budget(shots_per_expectation: int, n_timesteps: int, readout_dim: int) -> int`
     — the explicit budget formula S·T·D (identical for both models by construction).

   ### Effective-dimension fairness guard (REQUIRED — from WP2 adversarial review)
   Declared D-parity is necessary but NOT sufficient: QELM's per-snapshot feature
   blocks can silently collapse to input-independent dead features when `T_evolve`
   sits far from the shared Hamiltonian's dynamical timescale (empirically verified:
   last-snapshot variance → ~1e-6 at large `T_evolve`; all snapshots identical at
   `T_evolve=0`). `assert_matched` would then certify a hollow D-parity. Add:
   - `effective_dimension(features: np.ndarray) -> dict` — on the valid rows: per-
     column across-input variance vector, and effective rank = participation ratio
     `(Σσ_i)² / Σσ_i²` of the singular values of the column-centered matrix (reuse
     `core.diagnostics.effective_rank` if it computes exactly this — CHECK the source;
     else implement the participation ratio here). Returns `{n_dead_cols, min_col_var,
     eff_rank, eff_rank_frac = eff_rank/D}`.
   - `assert_features_live(bundle: FeatureBundle, *, min_col_var=1e-8,
     min_rank_frac=0.5) -> None` — raises MatchingError if any feature column's
     across-input variance < `min_col_var` (dead feature) OR `eff_rank_frac <
     min_rank_frac` (multiplex/rank collapse). This is a CONFIG-SANITY guard that
     converts the silent confound into a loud failure at feature-build time. Run it
     on BOTH the QRC and QELM bundles before any scoring.
   - `recommend_T_evolve(qelm_cfg, coupling_seed, n_snapshots) -> dict` — build
     `L_base` and return a physically-grounded target `T_evolve` window so the V
     snapshots span the live dynamical regime:
       * γ > 0 (dissipative): use `core.diagnostics.mixing_time` / `spectral_gap`;
         target `T_evolve ≈ mixing_time` (last snapshot near, not at, steady state).
       * γ == 0 (unitary, NO fixed point → mixing_time undefined): use the H_static
         spectral bandwidth Δ = λ_max − λ_min; target `T_evolve ≈ 2π/Δ` (snapshots
         sample distinct phases). CHECK `core/diagnostics.py` for existing helpers
         (spectral_gap/mixing_time/effective_rank) before writing your own.
     Return `{t_lo, t_recommended, t_hi, regime, timescale}`. This helper INFORMS the
     WP7 pilot's frozen `T_evolve` grid — it does not itself freeze anything.
   Tests: a config with `T_evolve=0` and one with `T_evolve` = 100×mixing_time each
   trip `assert_features_live`; a well-chosen `T_evolve` (from `recommend_T_evolve`)
   passes; `recommend_T_evolve` returns a finite window in BOTH γ=0 and γ>0 regimes.
3. `src/qrc_vs_qelm/shots.py`:
   - `add_shot_noise(features, shots, rng) -> np.ndarray` — PORT of
     `_add_shot_noise` from qrc_dataset_profiler reservoir.py:82 (public name,
     docstring records source path). Extension: `shots=None` or `np.inf` returns an
     unmodified float64 copy (the S=∞ tier).
4. Tests (`tests/test_api_matching.py`, `tests/test_shots.py`):
   - Matched config pair on identical inputs + same coupling_seed → equal feature
     shapes, equal `couplings` matrices (bit-identical J), equal D.
   - `assert_matched` passes on the matched pair; flipping EACH axis one at a time
     raises MatchingError mentioning that axis.
   - Builders are deterministic (two calls → `np.array_equal`).
   - `quantize_inputs`: idempotent; levels=201 grid values; passthrough on None.
   - Shot noise: for a constant feature column f, empirical std over many draws ≈
     `sqrt((1-f²)/shots)` within 5%; outputs clipped to [-1,1]; rng-deterministic;
     `shots=np.inf` → exact passthrough; features outside [-1,1] are clipped first
     (document: only valid for ±1-valued observable expectations).
   - QRC full-length contract: `build_qrc_features` returns T rows for T inputs.
   - Keep tests at N ≤ 4, T ≤ 60.

## Acceptance

```
.venv/bin/pytest -q
```
All green. Also run a 3-line sanity snippet (print shapes + budget) and include it
in your summary.

## Rules

Follow the ported core/WP2 code where names differ from this spec; note deviations.
Explicit rng everywhere. No results files. No commits. Don't touch `.specs/`.
