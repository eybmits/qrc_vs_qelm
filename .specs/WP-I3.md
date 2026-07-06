# WP-I3 — IPC window-sweep runner: the Δ(k, γ) surface

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Depends on WP-I1 (TD-QELM +
`windowed_qelm_from_qrc`) and WP-I2 (`ipc.total_ipc`). MIRROR the structure of
`run_stage3.py` (gamma sweep) — reuse its manifest/checkpoint/clean-tree/config patterns.

## Goal

Produce the thesis's central data: total IPC of the **full QRC** (ceiling) vs the **windowed
TD-QELM** at every window length k, giving **Δ(k) = IPC(QRC) − IPC(QELM,k)**, swept over
dissipation γ → the surface **Δ(k, γ)**. (N is set by the config/profile; the maximal run
does N∈{5,6} as separate invocations/configs — WP-I0 wires N=6 sparse.)

## Deliverable — `src/qrc_vs_qelm/run_stage_ipc.py`

CLI mirrors run_stage3: `--profile smoke|full`, `--out`, `--allow-dirty`,
`--checkpoint-every`. Enforce `require_clean_tree`.

- **Items** = `window_grid` (k=1..~12) × `gamma_grid` × dissipator × seed cells (mirror
  `_stage3_items`). Item id includes k, γ, dissipator, cell.
- **Per item** (`_score_ipc_item`, mirror `_score_gamma_item`):
  - Build the **full QRC** feature matrix once per (γ, dissipator, seed) at the frozen
    operating config (this is the ceiling; cache it, reuse across all k for that (γ,seed)).
  - Build the **windowed TD-QELM** via `windowed_qelm_from_qrc(qrc_cfg, window_k=k)` (WP-I1
    single-source-of-truth → guaranteed same dynamics) at window k.
  - `assert_bundles_matched` + `assert_features_live` on both.
  - Compute `ipc.total_ipc(features, u, basis, splits, ...)` for QRC and QELM → `ipc_total,
    ipc_mc, ipc_nc, n_significant, threshold` each. **FAIRNESS CONTRACT (from WP-I2 review):**
    both models MUST use the IDENTICAL `basis` (from `ipc.dambre_basis(...)`, built once per
    seed input stream), the IDENTICAL `alphas` and `n_surrogates`, and be scored on the
    **SAME valid-row set** — because QRC (`first_valid`=washout) and TD-QELM(k)
    (`first_valid`=k−1) differ per k, use ONE shared valid-index set = the intersection
    across the MOST-RESTRICTIVE k in the sweep (largest `first_valid`), so every model at
    every k is scored on bit-identical rows. A test asserts the shared index set is identical
    across models/k. Verify `max_delay` (basis) ≥ the QRC memory horizon (the
    `basis_edge_capacity` diagnostic from WP-I2 must be negligible — else widen the basis).
  - `delta_ipc = ipc_total(QRC) − ipc_total(QELM,k)`; also `delta_mc`, `delta_nc`.
- **Output**: `ipc_scores.csv` — columns `cell_id, input_seed, coupling_seed, n_qubits,
  window_k, gamma, dissipator, model, ipc_total, ipc_mc, ipc_nc, n_significant, threshold,
  D, delta_ipc, delta_mc, delta_nc, features_dead, complete` — plus `ipc_manifest.json`
  (git state, seed grid, basis caps, config snapshot, per-cell cache sha256, runtime).
- **Dead-feature handling** (like stage-3): if `assert_features_live` fails at some (k,γ) for
  a model, record `features_dead=True`, IPC=0, continue (don't crash).
- **Runtime-budget guard**: reuse `project_runtime` + the pre-registered shrink message
  (shrink window range first, then seeds; never the γ grid or basis).

## Config additions (`configs/full.yaml`, `configs/smoke.yaml`)

```yaml
ipc:
  window_grid: [1, 2, 3, 4, 5, 6, 8, 10, 12]   # smoke: [1, 2, 3]
  basis: { max_delay: 20, max_degree: 4, max_total_degree: 4, max_functions: 600 }  # smoke: smaller
```
(γ grid + dissipator reuse the existing `gamma_grid`/`dissipator_grid`.)

## Tests (`tests/test_run_stage_ipc.py`)

- `main(["--profile","smoke","--out",tmp,"--allow-dirty"])` exits 0 end-to-end at reduced
  N/k/γ; writes well-formed `ipc_scores.csv` + manifest with the required columns.
- **Determinism**: two smoke runs bit-identical.
- **QRC ceiling ≥ windowed QELM**: on the smoke output, per (γ,seed), IPC(QRC) ≥ IPC(QELM,k)
  and `delta_ipc ≥ 0` within noise (the QRC is the ceiling — this is the sanity invariant).
- **IPC monotone-ish in k**: IPC(QELM,k) non-decreasing in k on average (more window → not
  less capacity), tested loosely on the smoke fixture.
- clean-tree enforced (no --allow-dirty → refuses on dirty tree).

## Acceptance

```
.venv/bin/pytest -q
PYTHONPATH=src .venv/bin/python -m qrc_vs_qelm.run_stage_ipc --profile smoke --out results_ipc_smoke --allow-dirty  # exit 0
```

## Rules

Reuse run_stage3's infra + WP-I1/WP-I2 APIs (follow as-built names; note deviations). The
QRC IPC ceiling is computed once per (γ,seed) and reused across k. Explicit rng. No commits.
Don't touch `.specs/`.
