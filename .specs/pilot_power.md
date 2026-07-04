# Pilot power confirmation — does 15-seed power clear the 20% GO bar?

Work ONLY inside `/Users/superposition/Coding/qrc_vs_qelm/`. Small change +
a run. The tuned pilot at γ=0.3 gave QRC 19.4% / QELM 13.9% with 67% ties at only
6 pilot seeds; the diagnostic says most ties are power-limited (median CI width ≈ 2×
tie_band). The confirmatory grid uses 30 seeds. Confirm whether higher power resolves
the regions BEFORE we freeze.

## Changes

1. Expand the DISJOINT pilot seed grid in `src/qrc_vs_qelm/seeds.py`: grow `pilot` to
   15 cells — input_seed ∈ {900,901,902,903,904} × coupling_seed ∈
   {987654, 123456, 555553}. Keep them PROVABLY DISJOINT from every confirmatory grid
   (update the existing disjointness test to cover the new seeds). Do not touch the
   confirmatory grids.
2. Add to `scripts/design_pilot.py` a report field per γ slice: of the current tie
   cells, how many become SIGNIFICANT (CI excludes 0, i.e. would leave "tie") vs stay
   genuine ties — call these `ties_power_limited` (direction non-tie by point estimate
   but CI straddles 0) and `ties_genuine` (point estimate within tie_band). Print both.
   This distinguishes "needs more seeds" from "both models genuinely fail (e.g. high p)".

## Run and report

```
PYTHONPATH=src .venv/bin/python scripts/design_pilot.py --profile full \
  --gammas 0.25,0.3 --taus 0,1,2,3,5,8,10,12,16 --degrees 1,2,3,5 \
  --out results_pilot_power
```
Report per γ: frac_qrc / frac_qelm / frac_tie, GO status, median CI width, and the
ties_power_limited vs ties_genuine counts. Also report the same fractions restricted to
degrees {1,2,3} only (dropping the p=5 saturation column) so we can see whether the
informative-degree subgrid clears 20%/20%.

## Rules

`results_pilot_*` gitignored. Run `pytest -q` (the seed-grid change touches the
disjointness test). No commits. Don't touch `.specs/`. Report numbers verbatim.
