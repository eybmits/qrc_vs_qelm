# Useful history on the actual NARMA10 target

The recurrent QRC outperforms the four-input parallel QELM in all 12 independent
runs. Limiting that same QRC to four inputs, with a newly fitted readout, removes
almost all of the mean advantage. This is a direct task-memory demonstration,
not another experiment in matching a QRC's outputs.

NARMA10 is a synthetic driven time-series benchmark, not measured real-world
data. Both models learn y[t+1] from u through t. Neither sees past target values.
The reduced-range iid-input setting follows [Fujii and Nakajima (2017), Eq. (18)
and Sec. IV D](https://doi.org/10.1103/PhysRevApplied.8.024030).

## Results

Test NRMSE = RMSE / training-target standard deviation. Percentages are error
scales, not proportions of correctly predicted samples. Medians across all
12 independently drawn input/coupling roots:

| Model | Accessible input history | Features | Median NRMSE |
| --- | --- | ---: | ---: |
| Parallel QELM | 4 latest inputs | 48 | 75.37% |
| Same QRC, reset/replayed and refitted | 4 latest inputs | 48 | 75.63% |
| Recurrent QRC | continuously carried state | 48 | 60.27% |
| Same circuit, reset/replayed and refitted | 12 latest inputs | 48 | 58.56% |
| Classical quadratic window | 4 latest inputs | 14 | 75.40% |
| Classical quadratic window | 12 latest inputs | 90 | 9.89% |

Paired exploratory effects in NRMSE percentage points (positive = benefit):

| Comparison | Mean difference | 95% paired bootstrap interval | Positive roots |
| --- | ---: | ---: | ---: |
| QELM(4) minus QRC | 15.62 | [13.01, 18.03] | 12/12 |
| Reset/replay(4) minus full QRC | 15.60 | [13.04, 17.95] | 12/12 |
| Reset/replay(4) minus reset/replay(12) | 19.13 | [16.66, 21.63] | 12/12 |

Loss rates .1/.3/1/4 and six ridge parameters were available to both primary
models, with model-specific selection on validation only. Every candidate's
test score is retained. The history controls keep the selected QRC dynamics
fixed and refit the readout for each window. Data are independent across
train/validation/test and across roots; each model receives the same input
realization within a root. There are 3967/895/1919 scored rows per split.

## What the plots establish

[Main candidate](../paper/figures/narma_memory_candidate/narma_memory_candidate.pdf)
and [all-run comparison](../paper/figures/narma_memory_candidate/narma_memory_all_runs.pdf)
are separate from the manuscript.

- a: Actual NARMA targets (black) and learned predictions. The first root and
  first 96 test rows were specified before running. The displayed test errors
  are for that root's entire 1919-row test split, not the excerpt or cohort
  median. The 12-input replay trace illustrates the first tested window longer
  than the task order; it is not a validation-selected window or a claimed
  optimum. All six tested windows appear in b.
- b: Access to more history helps at the same four qubits and 48 features.
  Between 8 and 12 inputs, the window gains access to the task's lag-nine input.
  It is not a universally monotone performance law: some longer replay windows
  are worse than 12, and as replay grows it approaches the full QRC dynamics.
  Bands/bars are empirical 10th–90th percentiles, not confidence intervals.
  Gray stars keep the strong classical control visible; its feature count is
  not matched to the quantum models.
- c: Two different prefixes receive an identical continuation. Their actual
  NARMA targets can remain different after their last four inputs have become
  identical. The four-input QELM then necessarily predicts the same value for
  both. This shows task-relevant missing history directly, independently of
  which model has lower average error. First 32 continuation steps are shown;
  all 47 causally available paired targets are exported.

Simple caption:

> Older inputs matter for the task itself. The models learn the actual NARMA10
> output, not another model's response. A four-input window misses useful
> history; a recurrent state or a longer replay window improves prediction.
> Different beginnings can still require different answers after the recent
> inputs have become identical. All-run results accompany the fixed example.

## Limits

This supports a benefit of accessible history in this setup, not a necessity
for quantum recurrence. The primary models have different input encodings;
the same-circuit reset/refit control is therefore essential. Longer replay
uses more evolution steps even though feature/qubit counts stay fixed.
The best quantum errors remain substantial. The 12-input classical control
is much more accurate; there is no quantum-advantage evidence here.

The input range is known and rounded to 33 levels identically for both quantum
models and classical controls; the target uses the underlying continuous
input. Only readout weights are learned. Quantum features are calculated as
exact expectations, with no gate noise, finite shots, or measurement-backaction
claim. The simulation uses the existing Ising/Lindblad model, not hardware.

This is a newly specified exploratory extension after earlier laser studies,
not a retroactive change to the paper's confirmatory protocol. The paired
intervals describe variability across these independent input/coupling draws,
not universal guarantees. No parameters, seeds or excerpt were searched to
produce a favorable outcome. The matched history comparison permits different
refitted readouts and tests available history plus its usable representation,
not a pure additive numerical estimate of an information-theoretic floor.

## Reproduction and checks

[Pre-run protocol](NARMA_MEMORY_PROTOCOL.md). All raw simulated sequences,
predictions and fitted readout parameters remain in ignored
`results_narma_memory_v1/`; do not commit that directory.

Run from the repository root with a new, nonexistent results directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/private/tmp/qrc-mpl-cache PYTHONPATH=pra_extension/code:src .venv/bin/python pra_extension/code/narma_memory.py --output results_narma_memory_reproduction
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/qrc-mpl-cache PYTHONPATH=pra_extension/code:src .venv/bin/python pra_extension/code/render_narma_memory.py --source results_narma_memory_reproduction --output paper/figures/narma_memory_reproduction
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/qrc-mpl-cache PYTHONPATH=pra_extension/code:src .venv/bin/python pra_extension/code/verify_narma_memory.py --source results_narma_memory_reproduction --output paper/figures/narma_memory_reproduction/verification.json
```

The implementation, protocol and imported core code were hashed into
`run_specification.json` before simulation. The completed source summary is
SHA256 `5cb4f7664ec84f04609644924e0d6eebbd84af3953e2d6edd4f81af754973df3`.
Simulation of all 12 roots took about 136 seconds on this local environment.

The verifier independently recomputed all 228 selected/candidate test scores,
all task targets and the input RNG streams. It reconstructed 1440 predictions
from saved readouts (12 fixed test points per fitted model/root), agreeing
within 5.4e-12. Same-window QELM predictions are identical. Physical-state
checks and the eight focused protocol/plot tests pass. Renderer tests also
check exact target bindings, retention of all windows, and legend separation.
PNG and PDF-render visual inspection confirms no clipped annotations or
overlapping legends. The complete repository test suite passes: 923 tests
in 161.93 seconds. The release inventory was refreshed for the additional
extension code/tests without changing the manuscript or its source archive.

No manuscript sections or paper figures were replaced. The ICAART PDF remains
the existing 12-page document; this candidate is for reviewing the additional
task-level evidence before any integration decision.
