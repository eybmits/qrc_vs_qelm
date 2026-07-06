# WP-I1 — Time-delayed QELM encoding (TD-QELM): decouple window k from N

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. This is Phase 2 (the IPC-gap
pivot). Do NOT regress the existing 153 tests or change the spatial QELM behavior.

## Why

The thesis measures the IPC gap Δ(k) between a full QRC and a **windowed QELM** as the
window length `k` grows. The current QELM encodes the window SPATIALLY (one input per
qubit, cyclic) so window ≤ N — capped at 5. We add a **time-delayed** encoding where the
`k`-input window is processed by **temporal re-uploading**, so `k` is decoupled from qubit
count and can run to ~12–16 at N=5.

## Design (implement this; it is the fairest convergence test)

The TD-QELM(k) at timestep `t` is the QRC's OWN input-affine dynamics run for `k` steps
over the window `w_t = (u_{t−k+1}, …, u_t)` **from a fresh reset state**, then measured —
i.e. a QRC whose memory is artificially truncated to `k` steps. This makes TD-QELM(k) →
QRC exactly as `k → ∞`, so Δ(k) is precisely the capacity carried by the QRC's memory
beyond the window. It also reuses the QRC's input-affine propagators (minimal new code)
and keeps the input-injection mechanism IDENTICAL to the QRC (fairness).

Concretely, for each row `t`:
1. Start from the reservoir's initial state `ρ0` (same `initial_state()` the QRC uses).
2. For j = t−k+1 … t (oldest→newest, in temporal order): apply the **input-affine
   propagator** for input `u_j` — the SAME step the `LindbladReservoir` applies (input
   drives `L(s) = L_base + s·L_drive`; reuse the existing quantized propagator cache).
3. After the k-th step, read the weight-≤2 Pauli observables → the feature row.

Because `ρ0` is reset every `t`, row `t` depends ONLY on `w_t` → strictly memoryless
(the defining QELM property). Window `k` is unbounded by N.

## Deliverables

1. `QELMConfig` (`models/api.py`): add `encoding: str = "spatial"` (values `"spatial"`,
   `"time_delayed"`) and reuse the existing `d` field as the **window length** (rename
   its docstring to "delay-embedding window length k"). In `__post_init__`: keep the
   `1 <= d <= n_qubits` check ONLY for `encoding == "spatial"`; for `"time_delayed"`
   require just `d >= 1` (decoupled from N). Validate `encoding` value.
2. `models/qelm.py`: add the TD path. Prefer reusing the QRC's input-affine step
   (`core/reservoirs.py` `LindbladReservoir` propagators built from the SAME
   `ising_xx_hamiltonian`/`transverse_drive`/dissipators as the QRC at this config) run
   over the k-window from `initial_state()`. Keep `n_snapshots` handling consistent
   (snapshots along the k-step trajectory OR just the final state — match the spatial
   convention so D is unchanged; document the choice). `first_valid = k − 1`.
   `build_qelm_features(..., encoding=...)` dispatches spatial vs time_delayed.
3. `models/api.py` `build_features`/`_build_qelm_bundle`: thread `encoding`; set
   `first_valid = d − 1` (= k−1) for both encodings; **D, n_obs, n_multiplex UNCHANGED**
   (assert_bundles_matched/assert_features_live must still pass against a matched QRC).

## Tests (`tests/test_qelm.py` additions)

- **Memorylessness at k > N**: `encoding="time_delayed"`, N=3, k=6; perturb `u[t0−k]` →
  row `t0` and later windows-excluding-it are BIT-IDENTICAL; rows whose window includes
  it differ. (The current test but with k>N.)
- **Convergence sanity**: at fixed config, TD-QELM(k) features approach the QRC's
  features as k grows toward the QRC's washout (qualitative: the per-row state at large k
  matches a QRC run for k steps from reset to ~1e-10).
- **Determinism** (same seed → bit-identical); **γ=0 purity** preserved; window k>N does
  not raise (spatial k>N still raises via QELMConfig).
- **Matching preserved**: build a matched QRC + TD-QELM(k) pair (same N, observables,
  n_snapshots) → `assert_bundles_matched` passes, D equal.

## Acceptance

```
.venv/bin/pytest -q
```
All green (153 + new). Summary: the TD path, the k>N memorylessness result, and
confirmation D/matching are unchanged.

## Rules

Reuse the QRC input-affine propagators (do not write a second dynamics). Explicit rng.
No results files. No commits. Don't touch `.specs/`. Note any deviation from this design
(e.g. if snapshot handling needs adjusting) in your summary.
