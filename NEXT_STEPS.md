# Where this project stands (resume guide)

_Snapshot for continuing on another machine. Clone, `pip install -e ".[dev]"`, `pytest -q`
(should be ~188 green), then pick up at "The decisive next run" below._

## Status

- **Phase 1 — the resource-matched QRC-vs-QELM comparison: DONE, committed, pushed, tagged
  `stage1-frozen`.** Results in `paper/` (`manuscript_draft.md`, `figures/`): QRC wins
  long-delay memory, QELM wins short-delay nonlinear (RQ1 GO); shots erode QRC's edge (RQ2,
  underpowered); dissipation = memory-vs-nonlinearity division of labour (RQ3).
- **Phase 2 — the IPC-gap pivot (the thesis's actual question): code DONE, pilot DONE = NO-GO.**
  TD-QELM encoding (WP-I1), Dambre IPC estimator (WP-I2, 3 calibration passes), window-sweep
  runner `run_stage_ipc.py` (WP-I3), pilot `scripts/ipc_pilot.py` (WP-I4). Signed framing chosen:
  Δ(k,γ) = IPC(QRC) − IPC(QELM at window k).

## Pilot result (capped T≤2000; raw JSON in `checkpoints/ipc_pilot_result.json`)

**Signed Δ(k,γ) surface** (positive = QRC wins, negative = window wins):

| γ | k=1 | k=3 | k=5 | k=8 | k=12 |
|---|---|---|---|---|---|
| 0.02 | +5.4 | −9.5 | −19.8 | −18.6 | −16.8 |
| 0.1 | +18.4 | +3.5 | −7.0 | −6.1 | −4.7 |
| 0.3 (tuned) | +27.2 | +11.2 | +0.1 | +0.2 | −0.6 |
| 0.5 | +29.1 | +13.1 | +1.5 | +0.9 | −0.8 |

- **Convergence shape is REAL**: at tuned γ=0.3 the QRC wins big at small k and the window
  closes the gap around **k≈5** (Δ→~0). At weak γ the window beats the near-unitary QRC.
- **BUT numbers are NOT trustworthy yet** — the pilot returned NO-GO because:
  - **T\* not found**: IPC still rising at T=2000 (QRC IPC 23→26→29 across T=500/1000/2000),
    rank-clamp fraction rising (0.59→0.74) → QRC IPC is UNDER-estimated → needs bigger T.
  - **Basis too small at low γ**: `basis_edge_capacity` up to 0.44 at max_delay=20 → QRC's
    long memory leaks past the basis → widen `max_delay`.
  - Tuned-γ Δ_∞ = **−0.6 ± 0.5** — the AAAI-deciding number, currently within the noise. Could
    be a small POSITIVE persistent gap (quantum memory matters) once the QRC IPC is properly
    resolved, or a true ZERO (window imitates QRC). **Unresolved.**

## The decisive next run (do this first when resuming)

Run the bigger-T + wider-basis pilot **overnight in tmux + caffeinate**, focused on the
tuned-γ sign. Edit `configs/full.yaml` `ipc.basis.max_delay` to ~40 (from 20), then:

```bash
# keep the Mac awake (lid open) tied to the run; if closing the lid: sudo pmset -a disablesleep 1
tmux new-session -d -s qrc_pilot2 \
  "cd $(pwd) && PYTHONPATH=src .venv/bin/python scripts/ipc_pilot.py --profile full \
     --t-values 2000,4000,8000 --gammas 0.3,0.5 --out results_ipc_pilot2 2>&1 | tee ipc_pilot2.log"
```

- If tuned-γ Δ_∞ resolves **> 0** (CI excludes 0) → PERSISTENT_GAP → "quantum memory has
  irreducible capacity no finite window recovers" (AAAI story A).
- If it resolves **≈ 0** → QELM_CONVERGES → "a windowed QELM imitates a QRC; window law
  k\*(γ) ~ 1/spectral-gap" (AAAI story B — see `core/diagnostics` for the spectral gap).

Then: freeze the IPC pre-registration (`ipc-frozen` tag by the user), wire N=6 sparse
(WP-I0, task pending), run the maximal Δ(k,γ) at N∈{5,6}, and write it up (WP-I5).

## Orchestration
Opus plans + writes one `.specs/WP-*.md` per WP; Codex (`codex exec`) implements; Sonnet
subagents review (code + adversarial) each WP before commit. See `AGENTS.md` / `CLAUDE.md`.
