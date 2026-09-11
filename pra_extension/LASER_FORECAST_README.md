# Forecasting the measured laser, with a controlled history test

This new experiment predicts actual future laser measurements, not a reservoir
observable. It supports a practical but narrower conclusion than “more memory
is always better”: retained history helps the tested fast-loss reservoir at the
one-step horizon, while slower loss makes that forecast worse. The four-step
QRC/QELM comparison is much less decisive.

Status: separate candidate figure, not integrated into the twelve-page ICAART
manuscript. The original emulation results and theoretical benchmarks are intact.
The [protocol](LASER_FORECAST_PROTOCOL.md) and its implementation were recorded
before this follow-up ran. This is a separately specified follow-up on a familiar
dataset, not a newly blinded dataset or an externally registered confirmation.

## Main results

Errors below are test NRMSE percentages, divided by the training target standard
deviation. They are medians across twelve matched coupling seeds. They are not
percentages of each target value, accuracy scores, or worst-case error bounds.

| Horizon | Loss setting | Full QRC | Four-input QELM | Same QRC, four-input reset/replay with refitted readout |
| --- | --- | ---: | ---: | ---: |
| One step | Fast | 16.13% | 20.21% | 18.39% |
| One step | Slow | 24.86% | 20.26% | 18.26% |
| Four steps | Fast | 37.23% | 37.41% | 38.13% |
| Four steps | Slow | 36.47% | 37.49% | 36.23% |

For the primary one-step fast-loss comparison, the full QRC beats the QELM in
all twelve seed pairs. The mean paired difference is 3.99 percentage points,
with a seed-bootstrap 95% interval of [3.40, 4.57]. Within that same QRC
architecture, restricting history to four inputs and refitting increases error
in all twelve seeds: mean paired difference 2.17 points, interval [1.58, 2.74].
This is evidence that retaining state beyond four inputs helps this fitted
forecasting architecture. It is not an exact additive decomposition of the
QRC/QELM performance gap, and it does not show that every four-input model fails.

The slow-loss one-step result reverses the ranking: the full QRC is worse than
the QELM in all twelve seeds (mean difference 4.74 points). Resetting the same
QRC to four inputs improves it in all twelve seeds (mean difference 6.61 points).
Thus a reservoir that holds information longer is not automatically more useful.

At four steps, neither loss setting has a clear paired QRC/QELM separation:
the mean QELM-minus-QRC intervals are [-0.42, 1.06] points for fast loss and
[-1.04, 1.93] for slow loss. The fast-loss four-input restriction has a small
positive mean effect, [0.11, 1.57] points, but only eight of twelve seeds agree;
the corresponding slow-loss effect is mixed, [-2.24, 0.69] points.
All intervals concern coupling-seed variation conditional on this one record;
they are not uncertainty over independent real-world time series and are not
adjusted for multiple comparisons. No primary claim is selected from the
secondary four-step comparisons.

## What the history curve means

We hold the reservoir dynamics, four qubits, 48 features and readout-fitting
procedure fixed. For each prediction, the control resets and replays only its
last 1, 2, 4, 8, 16 or 32 actual inputs. Each history length receives its own
train/validation-fitted readout. The dashed references are the full-history
QRC errors, not theoretical lower bounds.

For the next-step task, median error falls from about 73% at one retained input
to about 20% at two inputs. It is lowest among the displayed settings at eight
inputs: 15.09% for fast loss and 16.89% for slow loss. These are descriptive test
curve minima, not validation-selected window choices. Keeping still more history
does not monotonically improve the fitted predictor. The replay model is a
different encoding from the parallel QELM and spends up to 32 channel steps per
prediction; its results do not establish a free improvement at matched runtime.

The frozen-readout figures are secondary sensitivity diagnostics. Removing
history while keeping a full-history fit unchanged can produce extremely large
errors, because the features shift outside the readout's fitted distribution.
Those errors must not be called the unavoidable cost of missing history. The
refitted-readout figure is the primary comparison.

## Figure and short caption

- [Next-step figure](../paper/figures/laser_forecast_candidate/laser_forecast_h1.pdf)
- [Four-step figure](../paper/figures/laser_forecast_candidate/laser_forecast_h4.pdf)
- [All 720 model/seed/window/horizon rows](../paper/figures/laser_forecast_candidate/laser_forecast_all_rows.csv)
- [Source ledger](../paper/figures/laser_forecast_candidate/laser_forecast_source.json)
- [Independent verification](../paper/figures/laser_forecast_candidate/verification.json)

Suggested caption: “Forecasting the measured laser signal. a Both models predict
the next measured value from past inputs. b Their errors across twelve matched
reservoir realizations. c Resetting the same reservoir and limiting its history
tests which past inputs help prediction; each readout is refitted. Dashed lines
show full-history performance, not unavoidable error. Points show medians and
bars or bands the 10th–90th percentiles. All errors are test NRMSE percentages.”

The trace uses the specified first seed and first 128 scored test points, with
identical intensity scales across loss settings. The export also retains every
scored prediction for that seed, rather than only the shown excerpt.

## Scope and limits

- All future targets are unquantized real measurements. Both models see the
  same train-scaled, clipped, 33-level input representation. Quantized persistence
  gives 86.93% at one step and 162.50% at four steps; this is only a basic context
  baseline, not a competitive classical forecasting benchmark.
- Training, validation and test blocks are chronological and disjoint. No
  prediction crosses a split; no test values select readout coefficients,
  loss rates, windows, horizons, seeds or the displayed excerpt. However this
  time series was already used in the earlier emulation study, so the record
  itself was not unseen during research development.
- Exact-expectation simulations, twelve couplings, one measured record, two
  loss settings, two horizons. No finite-shot or hardware forecasting claim.
- Matching qubits, feature count and readout selection does not equalize all
  information access or encoding costs. The within-QRC reset control reduces
  the architecture confound, but it does not identify an architecture-independent
  optimal memory error or prove that quantum recurrence is necessary.
- A longer-window reset model can work well here. This experiment illustrates
  useful finite context; it does not demonstrate an infinite-history obstruction.

## Reproduction and checks

From the repository root, with the existing hash-verified laser file in place:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=pra_extension/code:src .venv/bin/python pra_extension/code/laser_forecast.py --output results_laser_forecast_reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=pra_extension/code:src .venv/bin/python pra_extension/code/render_laser_forecast.py --source results_laser_forecast_reproduction --output paper/figures/laser_forecast_reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=pra_extension/code:src .venv/bin/python pra_extension/code/verify_laser_forecast.py --source results_laser_forecast_reproduction --output paper/figures/laser_forecast_reproduction/verification.json
```

The runner refuses to overwrite an existing results directory. Numeric outputs
are deterministic; elapsed-time metadata and hashes containing it may differ.
Verification independently recomputes all 720 score rows, checks every test
target against the raw record and reconstructs 60 sampled predictions from
channel features and saved readouts. Unit tests also compare batched replay
with the original step-by-step implementation, check causality and train-only
scaling, and bind plotted marks and summaries to the exported source rows.
