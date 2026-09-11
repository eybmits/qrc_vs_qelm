# Laser time-series demonstration: reproduction and paper integration

The experiment is specified in [LASER_DEMONSTRATION_PROTOCOL.md](LASER_DEMONSTRATION_PROTOCOL.md).
Its numerical cohort remains separate from the frozen confirmatory cohorts.
On 10 September 2026 the author approved integrating the combined design into
ICAART Figure 3. The original protocol's no-integration status records the
pre-simulation scope and is preserved unchanged.

## Current Figure 3 update (11 September 2026)

The author-approved Figure 3 now places the paired-history memory test beside
the unchanged natural excerpt. Panel b shows absolute QRC output gaps for the
first 48 shared inputs, with medians and 10th–90th percentiles over all 12 seeds
per loss rate. The inset contains the actual two 128-input prefixes and common
128-input continuation, vertically offset for clarity. Its slight overlap with
the plot frame covers no data. The QELM windows agree from shared input four.
Neither the gap nor its percentile band estimates a global diameter or a
confidence interval. The numerical experiment and its protocol are unchanged.

The N5 curves previously shown in panel b remain archived in their original
figures, with all 360 rows in the combined source ledger and the numerical
findings in Section 5. The analytic panel c remains unchanged. Earlier
layout descriptions below are retained as revision history.

## Reproduce

From the repository root, with the repository environment installed:

```sh
mkdir -p data/santa_fe_laser
curl -L --fail 'https://raw.githubusercontent.com/shubhomoydas/ad_examples/81b9fbe053b7da1454202f3e604e25009ded60f3/datasets/SantaFeCompetition/SF_Acont.dat' -o data/santa_fe_laser/SF_Acont.dat
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=pra_extension/code:. .venv/bin/python pra_extension/code/laser_demonstration.py --output results_laser_demonstration_v1
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/private/tmp/qrc-mpl-cache PYTHONPATH=pra_extension/code:. .venv/bin/python pra_extension/code/render_laser_demonstration.py --source results_laser_demonstration_v1 --output results_laser_demonstration_v1/figures
.venv/bin/python -m pytest -q
```

Choose a new output directory for a repeated simulation. The runner refuses to
overwrite an existing run. Data and generated results remain ignored by Git.
The source checksum is checked before use; test runs do not require a download.

## Read the two figures

`laser_time_series.pdf` shows the same measured input excerpt and the resulting
QRC outputs alongside a trained QELM. Both quantum systems have four qubits,
48 features, and the same fixed couplings and physical evolution parameters
within each loss-rate comparison. The input interface differs: recurrent drive
versus product-state window encoding. The QRC output is fixed, not a separately
trained forecasting readout. An error label is the full held-out test error,
not the error on the displayed 128-sample excerpt. Normalization uses the
training-output standard deviation; percentages are not probabilities.

`laser_same_window_control.pdf` is the mechanism check. Two measured prefixes
are followed by the same measured continuation. From the fourth common input
onwards, the QELM must give both histories the same prediction. Any remaining
separation between the two QRC outputs witnesses missing-history ambiguity.
The histories are deliberate interventions assembled from real samples, not
two independent real-world records with naturally identical windows.

`laser_paper_style.pdf` is a compact alternate layout of the first figure at
the ICAART template's actual column width. It retains the same predeclared seed,
all 128 excerpt samples, and the same predictions. It follows the aligned-panel
and shared-legend style of the author's supplied reference (arXiv:2608.25511v1,
Figure 2, page 3), without copying that paper's phase bands or scientific claims.
The manuscript is not changed; the accompanying caption is a draft for integration.

## Combined Figure 3

`figure3_candidate/figure3_combined_candidate.pdf` combines the same laser
excerpt with the existing N5 window study and the analytic construction.
It borrows the grouped-axes idea from Figure 5 of the supplied reference,
not its phase boundaries or composition markers. It is 6.221 by 3.65 inches,
at the actual ICAART full-text width, about 4 mm taller than the present
Figure 3 at its manuscript scale.

The candidate retains all 360 N5 window-study rows and both loss rates.
Color and marker identify the series; solid/filled versus dashed/open
identify the loss rate. Every analytic copy count is retained. Only the
separate N5 feature-budget panel is omitted from this candidate: its source,
original figure, and manuscript discussion remain untouched. The candidate
does not pool the laser and N5 cohorts or mix NRMSE with absolute worst-case
error. A caption draft and source ledger accompany it.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/private/tmp/qrc-mpl-cache PYTHONPATH=pra_extension/code:. .venv/bin/python pra_extension/code/render_figure3_candidate.py
```

The author-selected version is now
`paper/figures/emulation_evidence/fig04_emulation_evidence.pdf`, integrated
through the ICAART asset pipeline. Figure 3(a) shows the measured-input
demonstration, (b) the combined N5 window curves, and (c) the exact construction.
Section 5.1 reports both 12-seed median errors; Appendix E.5 gives the separate
splits, shared dynamics, fitting, quantization and deliberately spliced control.
The 12-page manuscript distinguishes the displayed seed's 30.5% slow-loss
error from the across-seed median of 27.9%. Main-theory text and proofs are
unchanged. No experiment was rerun or retuned for this integration.

Regenerate the selected asset and paper with:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/private/tmp/qrc-mpl-cache PYTHONPATH=pra_extension/code:. .venv/bin/python pra_extension/code/render_figure3_candidate.py --publication --output paper/figures/emulation_evidence
.venv/bin/python paper/icaart2027/prepare_assets.py
.venv/bin/python paper/icaart2027/build_submission.py
```

The laser views show the first predeclared seed, not a visually selected winner.
They use separate output-axis ranges to reveal within-model dynamics; the
underlying observables remain in their original expectation-value units.
These panels demonstrate a finite-window limitation at the tested setting,
not memory that provably persists forever. They are not laser forecasting,
quantum hardware results, or evidence of a quantum advantage.

## Inspect the evidence

- `run_specification.json`: configuration and source/protocol/runner hashes
  recorded before the simulation.
- `preprocessing.json`: disjoint source blocks, training-only input scaling,
  quantization, clipping counts and unused samples.
- Per-cell JSON and NPZ: fitted coefficients, complete held-out trajectories,
  intervention trajectories, selected ridge penalty, and numerical checks.
- `summary.json`: all 24 cells, seed-wise summaries and every predeclared gate,
  including failures if any. Root-to-root ranges are not confidence intervals.
- `figures/figure_manifest.json`: array-to-summary checks, rendered-file hashes,
  numerical dependency hashes and versions, and the fixed excerpt indices.

The fitted error and the paired lower bound answer different questions. The
former includes representation and fitting limitations. The latter is a
pointwise, pair-specific ambiguity bound independent of training, not the
global common-suffix diameter and not a bound on this test set's average RMSE.
