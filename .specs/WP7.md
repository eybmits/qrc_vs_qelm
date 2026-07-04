# WP7 — Pre-registration freeze (docs + config; NO new science code)

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. This locks the frozen
operating point and pre-registration. The user applies the `stage1-frozen` git tag
AFTER this lands (you do NOT commit or tag). Populate configs + finalize docs.

## Frozen operating point (user-approved from the design pilot)

Write these into `configs/full.yaml` (+ `configs/smoke.yaml` scaled-down twin) and a
single source-of-truth `data/frozen/frozen_v1_config.json`:

- **Primary**: N=5 qubits; γ=0.3 (RQ1 operating point); dt=0.5; h=0.5; J_s=1.0;
  dissipator=local_loss; multiplexing V=1 (n_virtual=1 / n_snapshots=1);
  quantize_levels=33 (timing pilot: 33 vs 201 costs only 0.007 NRMSE — near-free);
  observables = default weight-2 3-axis Pauli (D=45).
- **Phase-diagram family (RQ1)**: Legendre delay, τ ∈ {0,1,2,3,5,8,10,12,16},
  degree p ∈ {1,2,3} → 27 cells.
- **Uninformative-cell exclusion rule (pre-registered)**: a cell is excluded from the
  gate fractions if BOTH models' test capacity < 0.05 (both fail → uninformative for the
  QRC-vs-QELM contrast). This is the objective rule that sets the degree axis to {1,2,3}
  (pilot showed degree-5 is a genuine both-fail band at N=5, `ties_genuine`, not power).
  The full {1,2,3,5} grid is still RUN and reported as a robustness appendix.
- **Secondary product family**: cross-terms (p1,p2) ∈ {(1,1),(1,2),(2,2)}, τ ∈ {1,2,4,8}.
- **Anchors (not gated)**: STM, NARMA-2/5/10, Mackey-Glass, parity, Santa Fe (if cached).
- **HP grids (equal cardinality, selected on val)**: QRC dt∈{0.25,0.5,1.0} × h_in∈{0.5,1.0}
  (6); QELM d∈{1,3,5} × T_evolve∈{two values inside the recommended live window
  [t_lo,t_hi] from recommend_T_evolve} (6). γ is NOT tuned (it is the RQ3 axis).
- **Shot tiers (RQ2)**: {∞, 1e5, 1e4, 1e3}.
- **Dissipation sweep (RQ3)**: γ ∈ {0.005,0.02,0.05,0.1,0.2,0.3,0.5} × dissipator ∈
  {local_loss, collective_loss}; both models channel-matched at each γ.
- **Robustness replication**: N=4 (coarse grid). (N=6 optional sparse anchor only.)
- **Seeds**: stage1_full (30 cells), stage2_shots, stage3_gamma (from seeds.py).
- **Gate thresholds (frozen)**: G1b regime fraction ≥ 0.20 each; G1c ≥ 0.25 significant
  QELM-over-linear_delay; tie_band = 0.02; FDR q ≤ 0.10.

## Runtime projection + shrink rule

Import `bench_configs.project_runtime` and project full-profile stage-1/2/3 runtime with
the frozen cell/seed/HP counts at quantize=33. If any stage exceeds a 10 h budget, apply
the PRE-REGISTERED shrink order and record it: cut input seeds first (10→ keep ≥6), then
the secondary product family; NEVER cut gates, shot tiers, the γ grid, or the τ/degree
axes. Write the projection + any shrink into the manifest and PREREGISTRATION.

## Docs to finalize

1. `PROTOCOL.md` (METHODS): resource-matching protocol (assert_bundles_matched +
   assert_features_live are the gates; V=1 rationale; identical shots/observables/
   readout/pipeline), shot-noise semantics, HP grids, split pipeline, the intentional
   QRC(dynamic-drive)-vs-QELM(state-prep) input-injection asymmetry (from
   WP7_prereg_notes), the uninformative-cell exclusion rule.
2. `PREREGISTRATION.md` (FROZEN COMMITMENTS): auto-render the gate table between the
   `<!-- GATES:BEGIN/END -->` markers via `gates.gates_markdown_table()`; frozen seed
   grids; row caps from the runtime projection; shrink rules; and the **AMENDMENT LOG**,
   which MUST record the full design-stage pilot trail verbatim and honestly:
   - γ swept 0 → 0.1 → {0.2,0.3,0.5} → {0.25,0.3} on DISJOINT pilot seeds; γ=0.3 chosen
     as the QRC-memory optimum (QRC region maximal, QELM still present).
   - τ axis extended 8 → 16 (diagnostic-driven; QRC's memory band sits at τ≥5).
   - pilot seeds expanded 6 → 15 to test power; result: ties are GENUINE not
     power-limited (ties_power_limited=0 at 15 seeds).
   - degree axis set to {1,2,3} via the uninformative-cell rule (degree-5 both-fail).
   - Full-grid pilot result at γ=0.3: QRC 19.4% / QELM 19.4% (both-fail p=5 dilutes);
     informative-subgrid: QRC 25.9% / QELM 22.2% → GO. Confirmatory seeds UNTOUCHED.
   - WP7 freeze checklist items from WP7_prereg_notes (poly_delay D floor, T_evolve grid
     inside [t_lo,t_hi] for both models, threshold rationale) — verify and check off.
3. `scripts/render_prereg.py` (+ test): renders the gate table into PREREGISTRATION and
   asserts it stays in sync (sync test in the suite).

## Acceptance

```
.venv/bin/pytest -q            # incl. the prereg render-sync test
```
Print: the frozen config JSON, the runtime projection (+ any shrink), and confirm the
amendment log contains the pilot trail. No commits, no tag (user does that). Don't touch `.specs/`.
