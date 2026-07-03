# WP1 — Port the Lindblad core simulator

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. You may READ
`/Users/superposition/Coding/qrc_dissipation_engineering/` (the source repo).
The scaffold (pyproject, venv at `.venv/`, tests/) already exists from WP0.

## Goal

Port the mature Lindblad QRC core from `qrc_dissipation_engineering/src/qrc/` into
`src/qrc_vs_qelm/core/`, as close to **verbatim** as possible. This core will serve
BOTH models of the study: the recurrent QRC directly, and (in WP2) a static QELM
channel built from the same Hamiltonians and dissipators.

## Deliverables

1. Port these modules from `/Users/superposition/Coding/qrc_dissipation_engineering/src/qrc/`
   to `src/qrc_vs_qelm/core/` (same filenames):
   `operators.py`, `liouvillian.py`, `dissipators.py`, `reservoirs.py`,
   `readout.py`, `sparse_evolve.py`, `diagnostics.py`.
   - Change ONLY: import paths (`qrc.` → `qrc_vs_qelm.core.`) and anything strictly
     required to run inside this package. No refactors, no renames, no "improvements".
     Verbatim porting keeps provenance auditable.
2. `src/qrc_vs_qelm/core/__init__.py` — re-export the public API
   (reservoir classes, `pauli_observables`/observable builders, Hamiltonian and
   dissipator constructors — mirror whatever `qrc/__init__.py` exposes in the source
   repo) and include a module docstring: source repo path + source git commit hash
   (get it with `git -C /Users/superposition/Coding/qrc_dissipation_engineering rev-parse HEAD`).
3. Port the matching test files from the source repo's `tests/` directory
   (test_operators, test_liouvillian, test_reservoirs, test_readout, test_dissipators,
   test_sparse_evolve — use whatever exists there) into `tests/`, adjusting imports only.
4. `scripts/smoke_core.py` — minimal smoke: build a dissipative reservoir at
   `n_qubits=4` with a fixed coupling seed (use the source repo's canonical
   constructor, e.g. the continuous-dissipation / `LindbladReservoir.from_terms`
   path), drive it with a fixed-seed uniform input sequence (T=200, washout=50),
   extract weight-1/2 Pauli features, and print: feature-matrix shape and
   `hashlib.sha256(features.tobytes()).hexdigest()[:16]`. Running it twice must
   print identical output (assert this in `tests/test_smoke_determinism.py` by
   calling the underlying function twice, not the script).

## Acceptance (run yourself; all must pass)

```
.venv/bin/pip install -e ".[dev]"   # in case deps changed
.venv/bin/pytest -q                  # scaffold + all ported tests + smoke determinism
.venv/bin/python scripts/smoke_core.py
```

## Rules

- If a ported module imports something outside the listed set, port that too and note
  it in your summary rather than rewriting it.
- If a source test depends on results files or paths that don't exist here, adapt the
  test minimally or skip it with an explicit `pytest.mark.skip(reason=...)` and note it.
- No new scientific code. No git commits. Don't touch `.specs/`.

When done, print: files ported (with any deviations from verbatim), test count, pytest result, smoke output.
