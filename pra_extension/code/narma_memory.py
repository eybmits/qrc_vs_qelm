"""Direct NARMA target study, prospectively specified separately from the paper."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from laser_demonstration import Config, ROOT, ALPHAS, architecture, digest, dump, physical_check
from laser_forecast import fit_forecaster, parameter_arrays, replay_features, score_prediction, paired_summary
from qrc_vs_qelm.core.tasks import narma_target
from qrc_vs_qelm.models.qelm import delay_embed


@dataclass(frozen=True)
class NarmaConfig(Config):
    seeds: tuple[int, ...] = tuple(range(920000, 920012))
    rates: tuple[float, ...] = (.1, .3, 1., 4.)
    order: int = 10
    input_scale: float = .2
    replay_windows: tuple[int, ...] = (1, 4, 8, 12, 24, 48)
    quadratic_windows: tuple[int, ...] = (4, 12)
    continuation: int = 48
    bootstrap_seed: int = 929991
    bootstrap_draws: int = 10000


def target_series(raw, cfg):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        y = narma_target(raw, order=cfg.order, input_scale=cfg.input_scale)
    if not np.isfinite(y[cfg.order:]).all():
        raise ValueError('Nonfinite NARMA target; do not clip or replace the sequence')
    return y


def make_data(seed, cfg):
    if cfg.washout < max(cfg.replay_windows)-1 or cfg.washout < cfg.order:
        raise ValueError('Insufficient common warm-up')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7301]))
    raw = [rng.uniform(size=n) for n in cfg.block_lengths]
    blocks = [np.rint(u*(cfg.levels-1))/(cfg.levels-1) for u in raw]
    indices = [np.arange(cfg.washout, len(u)-1) for u in raw]
    targets = [target_series(u, cfg)[t+1] for u, t in zip(raw, indices)]
    return raw, blocks, indices, targets


def quadratic_features(block, indices, window):
    if np.any(np.asarray(indices) < window-1):
        raise ValueError('Window crosses input boundary')
    x = np.asarray(block)[np.asarray(indices)[:, None]-np.arange(window)]
    left, right = np.triu_indices(window)
    return np.column_stack((x, x[:, left]*x[:, right]))


def fitted_row(name, features, targets, *, rate=None, window=None):
    fit, mean, sd, predicted = fit_forecaster(features, targets)
    row = {'model': name, 'gamma': rate, 'window': window,
           'features': features[0].shape[1], 'alpha': fit.alpha,
           'validation_mse': fit.validation_mse,
           **score_prediction(predicted, targets[2], sd)}
    return row, (fit, mean, sd, predicted)


def choose(candidates):
    """Only validation scores enter selection; stable tie order is predeclared."""
    return min(candidates, key=lambda candidate: candidate[0]['validation_mse'])


def pair_predictions(raw_test, cfg, qrc, qelm, obs, fits):
    p, c = cfg.prefix, cfg.continuation
    suffix = raw_test[2*p:2*p+c]
    histories = [np.r_[raw_test[k*p:(k+1)*p], suffix] for k in (0, 1)]
    output = {'pair_common_input': suffix}
    for label, raw in zip(('a', 'b'), histories):
        u = np.rint(raw*(cfg.levels-1))/(cfg.levels-1)
        t = np.arange(p, len(raw)-1)
        output[f'pair_raw_{label}'] = raw
        output[f'pair_target_{label}'] = target_series(raw, cfg)[t+1]
        features = {
            'qrc': qrc.run(u, obs, n_virtual=cfg.snapshots)[t],
            'qelm': qelm.features(delay_embed(u, cfg.window), obs,
                                  encoding_scheme='one_to_one')[t],
        }
        for name, x in features.items():
            fit, mean, sd, _ = fits[name]
            output[f'pair_{name}_{label}'] = fit.predict(x)*sd+mean
    after = cfg.window-1
    if not np.allclose(output['pair_qelm_a'][after:], output['pair_qelm_b'][after:],
                       atol=1e-11, rtol=0):
        raise AssertionError('Same input window produced different QELM predictions')
    return output


def evaluate(seed, cfg, output):
    started = time.perf_counter()
    raw, blocks, indices, targets = make_data(seed, cfg)
    arrays = {'test_target': targets[2], 'test_indices': indices[2],
              'training_target_sd': np.std(targets[0])}
    for k, label in enumerate(('train', 'validation', 'test')):
        arrays[f'{label}_raw_input'] = raw[k]
    candidates = {'qrc': [], 'qelm': []}
    candidate_rows = []
    for rate in cfg.rates:
        qrc, qelm, obs, coupling = architecture(seed, rate, cfg)
        features = {
            'qrc': [qrc.run(u, obs, n_virtual=cfg.snapshots)[t] for u, t in zip(blocks, indices)],
            'qelm': [qelm.features(delay_embed(u, cfg.window), obs,
                                   encoding_scheme='one_to_one')[t] for u, t in zip(blocks, indices)],
        }
        for name, fx in features.items():
            if any(x.shape[1] != cfg.snapshots*len(obs) for x in fx):
                raise AssertionError('Unmatched feature widths')
            row, fitted = fitted_row(name, fx, targets, rate=rate,
                                     window=cfg.window if name=='qelm' else None)
            candidates[name].append((row, fitted))
            candidate_rows.append(row)
            arrays[f'candidate_{name}_gamma{rate:g}'] = fitted[3]
        print(json.dumps({'seed': seed, 'gamma': rate,
                          'candidate_validation': {n: candidates[n][-1][0]['validation_mse']
                                                   for n in candidates}}), flush=True)
    rows, fits, selection = [], {}, {}
    for name, options in candidates.items():
        row, fitted = choose(options)
        rows.append(row.copy())
        fits[name] = fitted
        selection[name] = row['gamma']
        arrays[name] = fitted[3]
        arrays.update(parameter_arrays(name, *fitted[:3]))
    qrc, _, obs, coupling = architecture(seed, selection['qrc'], cfg)
    _, qelm, _, _ = architecture(seed, selection['qelm'], cfg)
    arrays['coupling'] = coupling
    for window in cfg.replay_windows:
        features = [replay_features(qrc, u, t, window, obs, cfg.snapshots)
                    for u, t in zip(blocks, indices)]
        row, fitted = fitted_row('replay', features, targets,
                                 rate=selection['qrc'], window=window)
        rows.append(row)
        key = f'replay_d{window}'
        arrays[key] = fitted[3]
        arrays.update(parameter_arrays(key, *fitted[:3]))
    for window in cfg.quadratic_windows:
        features = [quadratic_features(u, t, window) for u, t in zip(blocks, indices)]
        row, fitted = fitted_row('quadratic', features, targets, window=window)
        rows.append(row)
        key = f'quadratic_d{window}'
        arrays[key] = fitted[3]
        arrays.update(parameter_arrays(key, *fitted[:3]))
    baseline = np.full(len(targets[2]), np.mean(targets[0]))
    arrays['training_mean'] = baseline
    rows.append({'model':'training_mean', 'gamma':None, 'window':None, 'features':0,
                 **score_prediction(baseline, targets[2], np.std(targets[0]))})
    arrays.update(pair_predictions(raw[2], cfg, qrc, qelm, obs, fits))
    pair_gap = np.abs(arrays['pair_target_a']-arrays['pair_target_b'])
    result = {'seed': seed, 'selection': selection, 'candidate_rows': candidate_rows,
              'rows': rows, 'scored_rows': [len(t) for t in indices],
              'physical_check': physical_check(qrc, blocks[2][:128]),
              'pair_gap_after_four_inputs': float(pair_gap[cfg.window-1]),
              'array_file': f'seed{seed}.npz', 'seconds':time.perf_counter()-started}
    np.savez_compressed(output/result['array_file'], **arrays)
    dump(output/f'seed{seed}.json', result)
    print(json.dumps({'completed_seed':seed, 'seconds':result['seconds'],
                      'test_rows':rows}), flush=True)
    return result


def summarize(results, cfg):
    if len(results)!=len(cfg.seeds) or {r['seed'] for r in results}!=set(cfg.seeds):
        raise ValueError('Incomplete NARMA cohort')
    deltas = {'qelm_minus_qrc': [], 'replay_d4_minus_qrc': [], 'replay_d4_minus_d12': []}
    for cell in sorted(results, key=lambda x:x['seed']):
        rows = {(r['model'],r['window']):r['nrmse'] for r in cell['rows']}
        deltas['qelm_minus_qrc'].append(rows['qelm',4]-rows['qrc',None])
        deltas['replay_d4_minus_qrc'].append(rows['replay',4]-rows['qrc',None])
        deltas['replay_d4_minus_d12'].append(rows['replay',4]-rows['replay',12])
    return {'config':asdict(cfg), 'results':results,
            'paired_contrasts':{k:paired_summary(v,cfg) for k,v in deltas.items()},
            'scope':'Direct synthetic NARMA10 target; exploratory; exact expectations; not measured data',
            'checks':{'complete_cohort':True,
                      'physical_states':all(c['physical_check']['passed'] for c in results)}}


def run(output, cfg):
    output.mkdir(parents=True, exist_ok=False)
    dependencies = [Path(__file__), ROOT/'pra_extension/NARMA_MEMORY_PROTOCOL.md']
    dependencies += [ROOT/'pra_extension/code'/name for name in ('laser_demonstration.py','laser_forecast.py')]
    dependencies += list((ROOT/'src/qrc_vs_qelm').rglob('*.py'))
    dump(output/'run_specification.json', {
        'status':'specified_before_results', 'config':asdict(cfg), 'alphas':ALPHAS,
        'sources':{str(p.relative_to(ROOT)):digest(p) for p in dependencies},
        'display_seed':cfg.seeds[0], 'display_scored_rows':[0,96],
    })
    results = [evaluate(seed,cfg,output) for seed in cfg.seeds]
    summary = summarize(results,cfg)
    summary['files_sha256'] = {p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    dump(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output,NarmaConfig())
