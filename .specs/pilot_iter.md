# Pilot iteration — make operating point CLI-configurable

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Small change to
`scripts/design_pilot.py` so I can iterate the operating point without editing code.
Do NOT change the scoring/labeling/aggregation logic — only expose parameters.

## Context

The first pilot (γ∈{0,0.1}, τ∈{0,1,2,3,5,8}, p∈{1,2,3,5}) confirmed a physically-correct
two-regime structure (QELM wins short delays, QRC wins τ=5,8 at p=1) but returned NO-GO
because the QRC region is under-developed: γ=0.1 is below the QRC memory optimum and τ
barely extends past the QELM delay window d=5. We need to re-run at stronger dissipation
and a longer delay axis.

## Change

1. Add CLI args to `design_pilot.py` (argparse), each OVERRIDING the current hardcoded
   defaults when provided:
   - `--gammas` : comma-separated floats (the γ slices; both QRC and QELM get each γ, as
     now via `_base_configs(gamma)`). Default = current profile's `gamma_slices`.
   - `--taus`   : comma-separated ints (the delay axis TAUS). Default = current TAUS.
   - `--degrees`: comma-separated ints (the degree axis DEGREES). Default = current.
   - Keep `--profile` and `--out` as-is. The first γ in `--gammas` is the "primary"
     slice for reporting; ALSO print a per-slice GO check for EVERY γ (so I can see which
     γ gives GO), and set the top-level GO iff ANY provided γ slice meets the GO bar
     (frac_qrc≥0.20 AND frac_qelm≥0.20) — because the RQ1 operating γ will be chosen as
     the memory-active one, not γ=0. Print which γ slice(s) pass.
2. Do not touch the disjoint pilot seed grid, the significance-gated labeling, the
   noise/CI-width reporting, or the tie_band sensitivity. Those stay exactly as built.
3. Keep it deterministic and side-effect-free except writing `--out`.

## Acceptance

Run and paste the output of:
```
PYTHONPATH=src .venv/bin/python scripts/design_pilot.py --profile full \
  --gammas 0.2,0.3,0.5 --taus 0,1,2,3,5,8,10,12,16 --degrees 1,2,3,5 \
  --out results_pilot_iter
```
Report: per-γ frac_qrc/frac_qelm/frac_tie, which γ (if any) hits GO, and the QRC-region
cell coordinates at the best γ. `results_pilot_*` is gitignored. No commits. Don't touch `.specs/`.
