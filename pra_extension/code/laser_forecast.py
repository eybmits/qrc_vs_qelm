"""Prospectively specified real-target forecast and history-intervention study."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time

import numpy as np

from laser_demonstration import (
    Config, ROOT, ALPHAS, architecture, digest, dump, fit_readout,
    input_blocks, load_laser, physical_check,
)
from qrc_vs_qelm.models.qelm import delay_embed


@dataclass(frozen=True)
class ForecastConfig(Config):
    seeds: tuple[int, ...] = tuple(range(918000, 918012))
    horizons: tuple[int, ...] = (1, 4)
    replay_windows: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    bootstrap_seed: int = 919991
    bootstrap_draws: int = 10000


def forecast_blocks(raw, cfg):
    """All horizons share causal feature indices; targets cannot cross a split."""
    blocks, preprocessing = input_blocks(raw, cfg)
    cuts = np.r_[0, np.cumsum(cfg.block_lengths)]
    raw_blocks = [np.asarray(raw[left:right], float)
                  for left, right in zip(cuts[:-1], cuts[1:])]
    indices = [np.arange(cfg.washout, len(block)-max(cfg.horizons)) for block in blocks]
    if any(len(index) == 0 for index in indices) or cfg.washout < max(cfg.replay_windows)-1:
        raise ValueError("Insufficient washout or scored rows for the declared windows")
    targets = {h: [block[t+h] for block, t in zip(raw_blocks, indices)] for h in cfg.horizons}
    return blocks, indices, targets, preprocessing


def replay_features(qrc, block, indices, window, observables, snapshots):
    """Batch independent resets, exactly matching qrc.run(window)[-1]."""
    block = np.asarray(block, float)
    indices = np.asarray(indices, int)
    if window < 1 or np.any(indices < window-1) or np.any(indices >= len(block)):
        raise ValueError("Replay window extends outside the observed input block")
    windows = block[indices[:, None] - np.arange(window-1, -1, -1)]
    initial = qrc.initial_state().reshape(-1, order="F")
    states = np.broadcast_to(initial, (len(indices), len(initial))).copy()
    for step in range(window-1):
        values = windows[:, step]
        for value in np.unique(values):
            selected = values == value
            propagator = qrc._sub_propagator(float(value), qrc.dt)
            states[selected] = states[selected] @ propagator.T
    measure = np.stack([o.matrix.T.reshape(-1, order="F") for o in observables])
    output = np.empty((len(indices), snapshots*len(observables)), float)
    for value in np.unique(windows[:, -1]):
        selected = windows[:, -1] == value
        step = qrc._sub_propagator(float(value), qrc.dt/snapshots)
        maps, power = [], np.eye(len(initial), dtype=complex)
        for _ in range(snapshots):
            power = step @ power
            maps.append(measure @ power)
        output[selected] = np.real(states[selected] @ np.vstack(maps).T)
    return output


def score_prediction(prediction, truth, training_sd):
    error = np.asarray(prediction)-np.asarray(truth)
    if training_sd <= 1e-12 or not np.isfinite(error).all():
        raise ValueError("Invalid forecast scale or nonfinite predictions")
    rmse = float(np.sqrt(np.mean(error**2)))
    return {"rmse": rmse, "nrmse": rmse/training_sd}


def fit_forecaster(features, targets):
    mean, sd = float(np.mean(targets[0])), float(np.std(targets[0]))
    if sd <= 1e-12:
        raise ValueError("Constant forecast target")
    fit = fit_readout(features[0], (targets[0]-mean)/sd,
                      features[1], (targets[1]-mean)/sd)
    predicted = fit.predict(features[2])*sd+mean
    return fit, mean, sd, predicted


def parameter_arrays(prefix, fit, mean, sd):
    return {f"{prefix}_{name}": np.asarray(value) for name, value in {
        "mean": fit.mean, "scale": fit.scale, "weights": fit.weights,
        "intercept": fit.intercept, "alpha": fit.alpha,
        "target_mean": mean, "target_sd": sd,
    }.items()}


def evaluate_cell(blocks, indices, targets, preprocessing, seed, rate, cfg, output):
    started = time.perf_counter()
    qrc, qelm, obs, coupling = architecture(seed, rate, cfg)
    full = [qrc.run(block, obs, n_virtual=cfg.snapshots)[t]
            for block, t in zip(blocks, indices)]
    parallel = [qelm.features(delay_embed(block, cfg.window), obs,
                             encoding_scheme="one_to_one")[t]
                for block, t in zip(blocks, indices)]
    if any(x.shape != y.shape or x.shape[1] != 48 for x, y in zip(full, parallel)):
        raise AssertionError("Expected matched 48-feature primary models")
    lo, hi = preprocessing["training_min"], preprocessing["training_max"]
    persistence = blocks[2][indices[2]]*(hi-lo)+lo
    arrays = {"test_indices": indices[2], "coupling": coupling}
    rows, fitted_qrc = [], {}
    for horizon in cfg.horizons:
        target = targets[horizon]
        arrays[f"target_h{horizon}"] = target[2]
        for name, features in (("qrc", full), ("qelm", parallel)):
            fit, mean, sd, predicted = fit_forecaster(features, target)
            key = f"{name}_h{horizon}"
            arrays[key] = predicted
            arrays.update(parameter_arrays(key, fit, mean, sd))
            rows.append({"model": name, "horizon": horizon,
                         "window": cfg.window if name == "qelm" else None,
                         "alpha": fit.alpha, "validation_mse_normalized": fit.validation_mse,
                         **score_prediction(predicted, target[2], sd)})
            if name == "qrc":
                fitted_qrc[horizon] = (fit, mean, sd)
        arrays[f"persistence_h{horizon}"] = persistence
        rows.append({"model": "persistence", "horizon": horizon, "window": 1,
                     **score_prediction(persistence, target[2], np.std(target[0]))})
    for window in cfg.replay_windows:
        restricted = [replay_features(qrc, block, t, window, obs, cfg.snapshots)
                      for block, t in zip(blocks, indices)]
        for horizon in cfg.horizons:
            target = targets[horizon]
            fit, mean, sd, predicted = fit_forecaster(restricted, target)
            key = f"replay_d{window}_h{horizon}"
            arrays[key] = predicted
            arrays.update(parameter_arrays(key, fit, mean, sd))
            rows.append({"model": "replay_refit", "horizon": horizon, "window": window,
                         "alpha": fit.alpha, "validation_mse_normalized": fit.validation_mse,
                         **score_prediction(predicted, target[2], sd)})
            original, mean, sd = fitted_qrc[horizon]
            unchanged = original.predict(restricted[2])*sd+mean
            arrays[f"frozen_d{window}_h{horizon}"] = unchanged
            rows.append({"model": "replay_frozen", "horizon": horizon, "window": window,
                         "alpha": original.alpha,
                         **score_prediction(unchanged, target[2], sd)})
    name = f"seed{seed}_gamma{rate:g}"
    np.savez_compressed(output/f"{name}.npz", **arrays)
    physical = physical_check(qrc, blocks[2][:128])
    result = {"seed": seed, "gamma": rate, "rows": rows,
              "features_each": 48, "n_qubits_each": cfg.n_qubits,
              "scored_rows": [len(t) for t in indices], "physical_check": physical,
              "array_file": f"{name}.npz", "seconds": time.perf_counter()-started}
    dump(output/f"{name}.json", result)
    return result


def paired_summary(values, cfg):
    values = np.asarray(values, float)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    draws = values[rng.integers(0, len(values), (cfg.bootstrap_draws, len(values)))].mean(axis=1)
    return {"n_seeds": len(values), "differences": values.tolist(),
            "mean": float(values.mean()), "median": float(np.median(values)),
            "mean_ci95": np.quantile(draws, [.025, .975]).tolist(),
            "positive_seeds": int(np.count_nonzero(values > 0))}


def summarize(results, cfg):
    expected = {(s, r) for s in cfg.seeds for r in cfg.rates}
    if len(results) != len(expected) or {(r['seed'], r['gamma']) for r in results} != expected:
        raise ValueError("Incomplete forecasting cohort")
    contrasts = {}
    for rate in cfg.rates:
        cells = sorted((r for r in results if r['gamma'] == rate), key=lambda r: r['seed'])
        for h in cfg.horizons:
            delta_qelm, delta_history = [], []
            for cell in cells:
                rows = {(r['model'], r['window']): r['nrmse']
                        for r in cell['rows'] if r['horizon'] == h}
                delta_qelm.append(rows['qelm', cfg.window]-rows['qrc', None])
                delta_history.append(rows['replay_refit', cfg.window]-rows['qrc', None])
            contrasts[f"gamma{rate:g}_h{h}"] = {
                "qelm_minus_qrc": paired_summary(delta_qelm, cfg),
                "restricted_d4_minus_qrc": paired_summary(delta_history, cfg),
            }
    return {"config": asdict(cfg), "results": results, "paired_contrasts": contrasts,
            "scope": "direct measured-target forecasting; no known worst-case floor; one record",
            "checks": {"complete_cohort": True,
                       "sampled_physical_states": all(r['physical_check']['passed'] for r in results)}}


def run(data, output, cfg):
    output.mkdir(parents=True, exist_ok=False)
    dependencies = [Path(__file__), Path(__file__).with_name('laser_demonstration.py'),
                    ROOT/'src/qrc_vs_qelm/core/reservoirs.py',
                    ROOT/'src/qrc_vs_qelm/models/qelm.py', ROOT/'src/qrc_vs_qelm/pipeline.py',
                    ROOT/'pra_extension/LASER_FORECAST_PROTOCOL.md']
    raw = load_laser(data)
    dump(output/'run_specification.json', {
        'status': 'specified_before_forecasting_results', 'config': asdict(cfg),
        'data_sha256': digest(data), 'alphas': ALPHAS,
        'sources': {str(p.relative_to(ROOT)): digest(p) for p in dependencies},
        'display_seed': cfg.seeds[0], 'display_scored_rows': [0,128],
    })
    blocks, indices, targets, preprocessing = forecast_blocks(raw, cfg)
    dump(output/'preprocessing.json', preprocessing)
    results = []
    for seed in cfg.seeds:
        for rate in cfg.rates:
            cell = evaluate_cell(blocks, indices, targets, preprocessing, seed, rate, cfg, output)
            results.append(cell)
            import json
            print(json.dumps({'seed': seed, 'gamma': rate, 'seconds': cell['seconds'],
                              'primary': [r for r in cell['rows'] if r['horizon']==1 and
                                          r['model'] in ('qrc','qelm','persistence')]}), flush=True)
    summary = summarize(results, cfg)
    summary['files_sha256'] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    dump(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/santa_fe_laser/SF_Acont.dat')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output, ForecastConfig())
