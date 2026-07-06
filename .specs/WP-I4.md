# WP-I4 — IPC design pilot: T-reliability + the SIGNED Δ(k,γ) surface + gate freeze

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Depends on the committed
WP-I1/I2/I3. This is the GO/NO-GO pilot before the confirmatory run, on the DISJOINT
`pilot` seed grid (confirmatory seeds untouched). User-chosen framing: the **signed**
Δ(k,γ) = IPC(QRC) − IPC(QELM,k) surface (positive where the dissipation-tuned QRC wins;
negative where a near-unitary QRC loses to the reset window).

## Deliverable — `scripts/ipc_pilot.py` (writes `results_ipc_pilot/pilot.json` + printed summary)

### Part A — T-reliability (MUST come first; total IPC is only trustworthy at large T)

At a FIXED tuned operating point (γ=0.3, N=5, a few windows k∈{1,4,8}) on disjoint pilot
seeds, compute IPC(QRC) and IPC(QELM,k) at **T ∈ {500, 1000, 2000, 4000, 8000}** (washout
scaled with T). Report, per T: total IPC (QRC and QELM), the fraction of the rank clamp
that binds (`raw_total/rank` — if ~1 the estimate is rank-dominated, NOT trustworthy),
`n_significant`, and `basis_edge_capacity`. **Pick T\*** = the smallest T at which total IPC
has stabilized (successive-T change < ~5%) AND the clamp is not binding (`raw_total < rank`)
AND `basis_edge_capacity` is small. If none stabilizes by T=8000, report that and recommend
a larger basis or a longer T. This T\* sets the confirmatory T.

### Part B — the signed Δ(k,γ) surface (at T\*)

On disjoint pilot seeds, at T\*, run the coarse grid k∈{1,2,3,5,8,12} × γ∈{0.02,0.1,0.3,0.5}
(local_loss) via the existing `run_stage_ipc` machinery (or its scoring functions). Report:
- The **signed Δ(k,γ)** table and, per γ, **Δ_∞** = Δ at the largest k (with bootstrap CI).
- The **sign structure**: is there a γ* where Δ_∞ crosses zero (below: window wins, untuned
  QRC; above: QRC wins, persistent gap)? Report γ* and Δ_∞(γ) trend.
- **Convergence per γ**: does IPC(QELM,k) → IPC(QRC) as k grows (Δ(k)→Δ_∞)? Monotone-ish?
- Relate γ* / the required window k*(γ) to the reservoir spectral gap (`core.diagnostics`).

### Part C — proposed SIGNED gates (for the freeze) + GO/NO-GO

Propose (do not yet freeze — the user tags after review):
- **GI1** (per-γ convergence): for each γ, IPC(QELM,k) is significantly non-decreasing in k
  toward Δ_∞ (Spearman + CI).
- **GI2 (headline persistent gap)**: at the tuned γ (≈0.3), Δ_∞ > 0 with bootstrap CI
  excluding 0 ⇒ `PERSISTENT_GAP` (genuinely-temporal quantum memory the window can't
  recover); CI includes 0 ⇒ `QELM_CONVERGES`.
- **GI3 (RQ3 sign-crossover)**: Δ_∞(γ) changes sign at a γ* (window-wins ↔ QRC-wins), and γ*
  relates to the spectral gap. CI on γ* / the trend.
- **GO** iff (i) T\* found (IPC trustworthy at feasible T), (ii) the tuned-γ persistent gap is
  measurable (GI2 resolvable), (iii) the signed surface is non-degenerate (both signs
  present across γ). Print a clear GO/NO-GO with the proposed confirmatory (T\*, k, γ) grid
  and, if NO-GO, concrete diagnostics (basis too small? T too small? no crossover?).

## Tests (`tests/test_ipc_pilot.py`)

- `main(["--profile","smoke",...])` exits 0 at reduced T-list/grid; writes well-formed
  `pilot.json` with parts A/B/C keys.
- The T-reliability logic picks the stabilized T on a synthetic monotone-then-flat fixture.
- The sign-crossover detection fires on a synthetic Δ_∞(γ) that changes sign.

## Acceptance

```
.venv/bin/pytest -q
PYTHONPATH=src .venv/bin/python scripts/ipc_pilot.py --profile smoke --out results_ipc_pilot_smoke   # exit 0
```
Summary: the chosen T\*, the signed Δ(k,γ) table with Δ_∞ CIs, the γ* crossover, and the
GO/NO-GO with the proposed confirmatory grid.

## Rules

Disjoint pilot seeds ONLY (confirmatory untouched); any range/T change is design-stage,
amendment-logged. Reuse `run_stage_ipc`/`ipc`/`stats`. Explicit rng. `results_ipc_pilot*`
gitignored. No commits. Don't touch `.specs/`. This is a Codex-built script I then RUN at
full pilot scale and interpret for the GO/NO-GO.
