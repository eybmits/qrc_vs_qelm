# Reference reproduction data

These files are the external reference against which the validation gate
(`scripts/run_validation.py`, check V1) confirms the ported Lindblad modeling
stack reproduces known physics:

- `best_vs_best.json` — short-term memory capacity and NARMA NMSE (with std) for the
  continuous-dissipation reservoir at N=5, used for the V1 tolerance band.
- `confirm_stm.json` — per-delay memory capacity for the continuous-dissipation vs.
  Fuji–Nakajima reservoirs, used for the V1 ordering sub-check.

**Provenance:** copied verbatim from the source repository
`qrc_dissipation_engineering` (`results/`) so this repository is self-contained.
`run_validation.py` reads them from here by default; set
`QRC_REFERENCE_RESULTS_DIR` to point elsewhere.
