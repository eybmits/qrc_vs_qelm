# WP2 — QELM model (new code)

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. The Lindblad core is
ported at `src/qrc_vs_qelm/core/` (WP1). Build QELM on top of it — do NOT duplicate
core functionality.

## Scientific definition (implement exactly this)

QELM = **static, memoryless** quantum feature map applied to a classical
delay-embedding of the input. Per timestep t:
window `w_t = (u_t, u_{t-1}, ..., u_{t-d+1})` → product state via cyclic encoding
(qubit q gets `core.operators.input_state(w_t[q mod d])`) → evolve under a **fixed,
input-independent** Lindblad channel for time `T_evolve` (same Hamiltonian family and
dissipators as the QRC: `ising_xx_hamiltonian(random_couplings(...), h, n)` + jumps
from `core.dissipators`) → weight-1/2 Pauli expectations at V snapshot times
`k·T_evolve/V, k=1..V`. Row t depends on `w_t` ONLY — strictly memoryless. Because
the channel is input-independent, the V snapshot propagators are precomputed ONCE
per configuration.

## Part 0 — WP1 review fixes (do these first; from dual review of the WP1 commit)

a. **Provenance**: `src/qrc_vs_qelm/core/__init__.py` cites source commit 7ac8c57,
   but `sparse_evolve.py` and `test_sparse_evolve.py` are UNCOMMITTED in the source
   repo. Amend the docstring: state those two files came from the uncommitted
   working tree and record `hashlib.sha256` hex digests of the two SOURCE files
   (read them from /Users/superposition/Coding/qrc_dissipation_engineering/).
b. **tasks module**: port the source repo's `tests/test_tasks.py` (imports adjusted);
   add a `from .tasks import ...` re-export block in `core/__init__.py` matching the
   pattern used for the other modules.
c. **`__all__` regression test**: add a test asserting `qrc_vs_qelm.core.__all__` is
   a superset of every public (non-underscore) name defined in each ported submodule.
d. **Positivity test** (physics gap): add a test that evolves a valid density matrix
   with `apply_propagator(propagator(lindbladian(H, jumps), t), rho)` at N=3 with
   local_loss jumps and asserts `min(np.linalg.eigvalsh(rho_out)) >= -1e-10` and
   trace ≈ 1, at a few t values.
e. **README**: WP index — mark WP1 complete, fix the WP2 row description to match
   this spec's title.

## Deliverables

1. `src/qrc_vs_qelm/models/__init__.py` and `src/qrc_vs_qelm/models/qelm.py`:
   - `delay_embed(u: np.ndarray, d: int) -> np.ndarray` — shape `(len(u), d)`, row t
     = `(u[t], u[t-1], ..., u[t-d+1])`; rows with t < d-1 are the caller's problem —
     also return/expose `first_valid_index(d) -> int` (= d-1) so the shared pipeline
     (WP4) can build the valid-index set.
   - `encode_window(window: np.ndarray, n_qubits: int, scheme: str = "cyclic") -> np.ndarray`
     — density matrix of the product state ⊗_q input_state(window[q % len(window)]).
     Raise on unknown scheme.
   - `class QELMChannel`:
     - `@classmethod from_terms(cls, n_qubits, H_static, jumps, T_evolve, n_snapshots)`
       — build `L_base` via `core.liouvillian.lindbladian`, precompute snapshot
       propagators `P_k = core.liouvillian.propagator(L_base, k*T_evolve/n_snapshots)`
       for k = 1..n_snapshots. Store propagators, n_qubits, n_snapshots.
     - `features(self, windows: np.ndarray, observables) -> np.ndarray` — for each
       window row: encode → for each P_k: `apply_propagator` → expectations via
       `core.readout.features_from_states` (or `expectation` per observable) →
       concatenate snapshots → feature row. Output shape `(n_windows, n_snapshots * n_obs)`.
   - `build_qelm_features(inputs: np.ndarray, *, n_qubits: int, d: int, T_evolve: float,
     n_snapshots: int, gamma: float, dissipator: str, J_s: float, h: float,
     coupling_seed: int, observables=None) -> np.ndarray` — convenience builder:
     couplings from `random_couplings(n_qubits, J_s, np.random.default_rng(coupling_seed))`,
     dissipator ∈ {"local_loss", "collective_loss", "dephasing"} mapped to
     `core.dissipators.*(n_qubits, gamma)` (empty jump list if gamma == 0),
     default observables = `core.readout.pauli_observables` weight ≤ 2, axes x/y/z.
     NOTE: gamma == 0 must be supported (unitary channel) — `lindbladian` with no
     jumps. Deterministic: same args → bit-identical output.
2. Tests (`tests/test_qelm.py`):
   - **Memorylessness** (the defining property): build features for a fixed-seed
     input sequence; perturb `u[t0 - d]` for some t0 ≥ d; assert feature rows at t0
     and later-window-disjoint rows are BIT-IDENTICAL where windows don't contain
     the perturbed value, and different where they do.
   - **Determinism**: two calls with identical args → `np.array_equal`.
   - **Shape**: `(len(u), n_snapshots * n_obs)` with n_obs = 3N + 3·C(N,2) for the
     default observable set (N=3 or 4 to keep it fast).
   - **Propagator correctness**: each stored `P_k` agrees with
     `scipy.sparse.linalg.expm_multiply(L_base * (k*T/V), v)` on a random complex
     vector to atol 1e-10.
   - **γ=0 purity**: with gamma=0, a pure encoded state stays pure through every
     snapshot (`trace(rho@rho) ≈ 1` within 1e-9) and features are real in [-1, 1].
   - **Cyclic encoding**: for d=2, N=4, qubits 0/2 and 1/3 carry identical
     single-qubit reduced states (use `core.operators.partial_trace`).
   - Keep all tests at N ≤ 4, T ≤ 60 — the whole file must run in seconds.

## Acceptance (run yourself)

```
.venv/bin/pytest -q
```
All tests green (scaffold + core + qelm).

## Rules

- Read the ported core under `src/qrc_vs_qelm/core/` and reuse its functions; if a
  name differs from this spec, follow the CODE, and note the difference in your summary.
- Explicit rng/seed args everywhere; no global RNG. No results files. No commits.
  Don't touch `.specs/`.

When done, print: files created, test count, pytest result, any spec/code name deviations.
