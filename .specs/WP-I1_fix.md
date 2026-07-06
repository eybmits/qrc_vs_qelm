# WP-I1-fix — thread ALL QRC dynamics into the TD-QELM (convergence bug class)

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Both WP-I1 reviewers (fix-first
/ approach-sound) found a convergence blocker, verified numerically. Do NOT regress the 161
tests. The DESIGN is validated (memorylessness, reset-transient, matching all correct) — only
parameter threading is broken.

## The bug (verified)

The TD-QELM path builds `H_drive = transverse_drive(n)` UNSCALED, while the QRC uses
`h_in · transverse_drive(n)` (`api.py:224`). `QELMConfig` has no `h_in`, so it's dropped.
Result: TD-QELM(k) converges to a DIFFERENT reservoir than the QRC whenever `h_in ≠ 1.0` —
and the frozen operating point uses `h_in=0.5` (`run_stage3.py:32 FIXED_QRC_H_IN`). Proof:
convergence distance ~1e-10 at h_in=1.0 but ~1.97 (non-decaying) at h_in=0.4. Every existing
convergence test hardcodes h_in=1.0, so it's untested. Currently DORMANT (TD not wired into a
runner) — fix before it is.

This is a bug CLASS: any QRC dynamics parameter not threaded to the TD-QELM breaks
convergence. Fix it structurally, not just for h_in.

## Required fix

1. **Add `h_in: float = 1.0` to `QELMConfig`** (`models/api.py`). Used ONLY by
   `encoding="time_delayed"`; the spatial static-channel path has no drive and must ignore
   it (confirm no spatial regression).
2. **Thread `h_in` into the TD path** (`models/qelm.py` build_qelm_features time_delayed):
   `H_drive = h_in · reservoirs.transverse_drive(n)` — identical to the QRC.
3. **Single source of truth — `models/api.py` helper**
   `windowed_qelm_from_qrc(qrc_cfg: QRCConfig, *, window_k: int, n_snapshots: int = ...) -> QELMConfig`
   that COPIES every dynamics parameter from the QRC config (`n_qubits, J_s, h, h_in, gamma,
   dissipator, quantize_levels`, and the per-step evolution time so each of the k TD steps
   equals one QRC `dt` step), setting `encoding="time_delayed"`, `d=window_k`. This guarantees
   TD-QELM(window_k) → QRC by CONSTRUCTION and is what the IPC runner (WP-I3) will use — no
   hand-built QELMConfig for the convergence experiment.
4. **Matching guard**: extend `matching.assert_matched` so that for `encoding="time_delayed"`
   pairs it also checks `h_in` (and any other shared dynamics param it can see) match; add
   `h_in` to the violation list. Spatial pairs unaffected.

## Tests (add to `tests/test_qelm.py` / `tests/test_api_matching.py`)

- **Convergence at the operating point (the critical missing test)**: TD-QELM(k→large) built
  via `windowed_qelm_from_qrc` from a QRC with **h_in=0.5** (and other non-trivial params)
  converges to that QRC's features to `< 1e-9`. Also test one more non-unit h_in.
- **Matching guard**: a time_delayed QELM with mismatched h_in vs its QRC → `assert_matched`
  raises naming `h_in`.
- **from_qrc helper**: the produced pair passes `assert_matched` + `assert_bundles_matched`,
  and D is unchanged.
- **Edge cases** (reviewer-suggested): `d==1` time_delayed smoke; `T_evolve==0` (or the
  step-time analog) with `d>1` has defined behavior (assert what it is).

## Acceptance

```
.venv/bin/pytest -q
```
All green. Summary: the h_in convergence result at 0.5 (`dist`), the from_qrc helper, and the
matching-guard test.

## Rules

Structural fix (single source of truth), not just an h_in patch. Reuse the QRC's propagators.
Explicit rng. No commits. Don't touch `.specs/`.
