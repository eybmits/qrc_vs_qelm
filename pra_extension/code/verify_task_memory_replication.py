"""Independent fresh-cohort score, target, selection, and directional checks.

The original frozen runners and their renderers are not modified or reused for
verification. Channel/readout reconstruction is a fixed-row sample, not a
second complete fit or a hardware validation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

import task_memory_replication as replication
from laser_demonstration import ALPHAS, DATA_HASH, ROOT, architecture, digest, dump
from qrc_vs_qelm.models.qelm import delay_embed


def independent_target(inputs, order, scale):
    """Literal NARMA recurrence, independently of the simulation task helper."""
    output = np.zeros(len(inputs), dtype=float)
    for time in range(order-1, len(inputs)-1):
        feedback = sum(float(output[time-lag]) for lag in range(order))
        output[time+1] = (.3*output[time]+.05*output[time]*feedback
                          +1.5*scale**2*inputs[time-order+1]*inputs[time]+.1)
    return output


def parameter_prediction(saved, key, features):
    standardized = (features-saved[key+'_mean'])/saved[key+'_scale']
    fitted = standardized@saved[key+'_weights']+saved[key+'_intercept']
    return fitted*saved[key+'_target_sd']+saved[key+'_target_mean']


def json_value(value):
    return json.loads(json.dumps(value))


def expected_keys(study, cfg):
    if study == 'narma':
        keys = {('qrc', None), ('qelm', cfg.window), ('training_mean', None)}
        keys.update(('replay', d) for d in cfg.replay_windows)
        keys.update(('quadratic', d) for d in cfg.quadratic_windows)
        return keys
    return {(model, window, h) for h in cfg.horizons
            for model, window in [('qrc', None), ('qelm', cfg.window), ('persistence', 1)]
            + [(model, d) for model in ('replay_refit', 'replay_frozen') for d in cfg.replay_windows]}


def check_coverage(summary, study, cfg):
    expected = set(cfg.seeds) if study == 'narma' else {(s, r) for s in cfg.seeds for r in cfg.rates}
    actual = [c['seed'] if study == 'narma' else (c['seed'], c['gamma']) for c in summary['results']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f'Incomplete or duplicated {study} cohort')
    wanted = expected_keys(study, cfg)
    for cell in summary['results']:
        keys = [(r['model'], r['window']) if study == 'narma'
                else (r['model'], r['window'], r['horizon']) for r in cell['rows']]
        if len(keys) != len(wanted) or set(keys) != wanted:
            raise ValueError(f'Incomplete or duplicated {study} settings')
        scored = [n-cfg.washout-(1 if study == 'narma' else max(cfg.horizons))
                  for n in cfg.block_lengths]
        if cell['scored_rows'] != scored:
            raise ValueError('Unexpected scored rows')
        if not cell['physical_check']['passed']:
            raise ValueError('Failed physical-state check')
        if study == 'narma':
            candidates = cell['candidate_rows']
            keys = [(r['model'], r['gamma']) for r in candidates]
            wanted_candidates = {(m, g) for m in ('qrc', 'qelm') for g in cfg.rates}
            if len(keys) != len(wanted_candidates) or set(keys) != wanted_candidates:
                raise ValueError('Incomplete or duplicated loss candidates')
            for model in ('qrc', 'qelm'):
                best = min((r for r in candidates if r['model'] == model),
                           key=lambda r: r['validation_mse'])
                selected = next(r for r in cell['rows'] if r['model'] == model)
                if cell['selection'][model] != best['gamma'] or selected != best:
                    raise ValueError('Selection is not the validation minimum')
            if any(r['gamma'] != cell['selection']['qrc'] for r in cell['rows'] if r['model'] == 'replay'):
                raise ValueError('History control changed the selected QRC dynamics')
        for row in cell['rows'] + cell.get('candidate_rows', []):
            if 'alpha' in row and row['alpha'] not in ALPHAS:
                raise ValueError('Undeclared ridge parameter')
            for key in ('validation_mse', 'validation_mse_normalized'):
                if key in row and (not np.isfinite(row[key]) or row[key] < 0):
                    raise ValueError('Invalid validation score')


def provenance(source):
    spec = replication.validate_spec(source)
    if spec['display'] != {'narma_seed': replication.NARMA_ROOTS[0],
                           'laser_seed': replication.LASER_ROOTS[0], 'scored_rows': [0, 96]}:
        raise ValueError('Predetermined display selection changed')
    for name, expected in spec['original_summaries'].items():
        if digest(ROOT/name/'summary.json') != expected:
            raise ValueError('Original cohort identity changed')
    studies = {}
    for study, cfg in replication.configs().items():
        directory = source/study
        summary = json.loads((directory/'summary.json').read_text())
        if summary['config'] != json_value(asdict(cfg)):
            raise ValueError('Fresh cohort configuration mismatch')
        check_coverage(summary, study, cfg)
        expected_files = {'run_specification.json'}
        if study == 'laser':
            expected_files.add('preprocessing.json')
        for cell in summary['results']:
            array_file = cell['array_file']
            expected_files.update((array_file, str(Path(array_file).with_suffix('.json'))))
        if set(summary['files_sha256']) != expected_files:
            raise ValueError('Missing or unexpected result-file provenance')
        for name, expected in summary['files_sha256'].items():
            if digest(directory/name) != expected:
                raise ValueError(f'Changed result file: {study}/{name}')
        for cell in summary['results']:
            recorded = json.loads((directory/Path(cell['array_file']).with_suffix('.json')).read_text())
            if recorded != cell:
                raise ValueError('Per-root result and summary disagree')
        run_spec = json.loads((directory/'run_specification.json').read_text())
        if run_spec['config'] != summary['config'] or run_spec['alphas'] != list(ALPHAS):
            raise ValueError('Study specification mismatch')
        for name, expected in run_spec['sources'].items():
            if digest(ROOT/name) != expected:
                raise ValueError(f'Frozen dependency changed: {name}')
        if study == 'laser' and run_spec['data_sha256'] != DATA_HASH:
            raise ValueError('Laser data identity changed')
        if not all(summary['checks'].values()):
            raise ValueError('Recorded study checks failed')
        studies[study] = summary
    return spec, studies


def check_score(row, prediction, target, training_sd):
    prediction = np.asarray(prediction)
    if prediction.shape != target.shape or not np.isfinite(prediction).all() or training_sd <= 0:
        raise ValueError('Invalid prediction shape, values, or training scale')
    rmse = float(np.linalg.norm(prediction-target)/np.sqrt(target.size))
    np.testing.assert_allclose([row['rmse'], row['nrmse']], [rmse, rmse/training_sd],
                               rtol=1e-12, atol=2e-13)
    return abs(rmse-row['rmse'])


def sampled_indices(length):
    return np.unique(np.linspace(0, length-1, 5, dtype=int))


def independent_replay(qrc, block, times, window, obs, snapshots):
    """Original sequential reservoir updates, not the batched replay helper."""
    return np.array([qrc.run(block[t-window+1:t+1], obs, n_virtual=snapshots)[-1]
                     for t in times])


def check_predictions(saved, key, features, picks, *, laser=False, comparison_key=None):
    predicted = parameter_prediction(saved, key, features)
    expected = saved[comparison_key or key][picks]
    np.testing.assert_allclose(predicted, expected, rtol=2e-8 if laser else 1e-9,
                               atol=2e-6 if laser else 1e-10)
    return float(np.max(np.abs(predicted-expected)))


def verify_narma(directory, summary, cfg):
    scores = predictions = 0
    score_difference = prediction_difference = 0.
    for cell in summary['results']:
        with np.load(directory/cell['array_file']) as saved:
            rng = np.random.default_rng(np.random.SeedSequence([cell['seed'], 7301]))
            targets = {}
            for label, length in zip(('train', 'validation', 'test'), cfg.block_lengths):
                raw = rng.uniform(size=length)
                np.testing.assert_array_equal(saved[f'{label}_raw_input'], raw)
                targets[label] = independent_target(raw, cfg.order, cfg.input_scale)[cfg.washout+1:]
            raw = saved['test_raw_input']
            times = np.arange(cfg.washout, len(raw)-1)
            np.testing.assert_array_equal(saved['test_indices'], times)
            np.testing.assert_allclose(saved['test_target'], targets['test'], rtol=1e-14, atol=1e-14)
            sd = float(np.std(targets['train']))
            np.testing.assert_allclose(saved['training_target_sd'], sd, rtol=1e-14)
            for candidate, rows in ((False, cell['rows']), (True, cell['candidate_rows'])):
                for row in rows:
                    key = (f"candidate_{row['model']}_gamma{row['gamma']:g}" if candidate else
                           f"{row['model']}_d{row['window']}" if row['model'] in ('replay', 'quadratic')
                           else row['model'])
                    score_difference = max(score_difference, check_score(row, saved[key], targets['test'], sd))
                    scores += 1
            np.testing.assert_allclose(saved['training_mean'], np.mean(targets['train']), rtol=1e-14)
            for model in ('qrc', 'qelm'):
                np.testing.assert_array_equal(saved[model], saved[f"candidate_{model}_gamma{cell['selection'][model]:g}"])
            suffix = raw[2*cfg.prefix:2*cfg.prefix+cfg.continuation]
            np.testing.assert_array_equal(saved['pair_common_input'], suffix)
            for index, label in enumerate(('a', 'b')):
                pair_raw = np.r_[raw[index*cfg.prefix:(index+1)*cfg.prefix], suffix]
                np.testing.assert_array_equal(saved[f'pair_raw_{label}'], pair_raw)
                expected = independent_target(pair_raw, cfg.order, cfg.input_scale)[cfg.prefix+1:]
                np.testing.assert_allclose(saved[f'pair_target_{label}'], expected, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(saved['pair_qelm_a'][cfg.window-1:], saved['pair_qelm_b'][cfg.window-1:],
                                       rtol=0, atol=1e-11)
            gap = abs(saved['pair_target_a'][cfg.window-1]-saved['pair_target_b'][cfg.window-1])
            np.testing.assert_allclose(cell['pair_gap_after_four_inputs'], gap, atol=1e-14)
            block = np.rint(raw*(cfg.levels-1))/(cfg.levels-1)
            picks = sampled_indices(len(times))
            qrc, _, obs, coupling = architecture(cell['seed'], cell['selection']['qrc'], cfg)
            _, qelm, _, _ = architecture(cell['seed'], cell['selection']['qelm'], cfg)
            np.testing.assert_array_equal(saved['coupling'], coupling)
            features = {'qrc': qrc.run(block, obs, n_virtual=cfg.snapshots)[times[picks]],
                        'qelm': qelm.features(delay_embed(block, cfg.window), obs,
                                              encoding_scheme='one_to_one')[times[picks]]}
            for d in cfg.replay_windows:
                features[f'replay_d{d}'] = independent_replay(qrc, block, times[picks], d, obs, cfg.snapshots)
            for d in cfg.quadratic_windows:
                x = block[times[picks, None]-np.arange(d)]
                left, right = np.triu_indices(d)
                features[f'quadratic_d{d}'] = np.column_stack((x, x[:, left]*x[:, right]))
            for key, x in features.items():
                prediction_difference = max(prediction_difference, check_predictions(saved, key, x, picks))
                np.testing.assert_allclose(saved[key+'_target_sd'], sd, rtol=1e-14)
                np.testing.assert_allclose(saved[key+'_target_mean'], np.mean(targets['train']), rtol=1e-14)
                predictions += len(picks)
        print(f'Verified NARMA root {cell["seed"]}', flush=True)
    return {'roots': len(cfg.seeds), 'score_rows_recomputed': scores,
            'predictions_reconstructed_at_fixed_rows': predictions,
            'maximum_score_recomputation_difference': score_difference,
            'maximum_reconstructed_prediction_difference': prediction_difference,
            'independent_recursion_rng_and_pair_targets_checked': True,
            'all_candidates_and_validation_selections_checked': True}


def laser_blocks(raw, cfg):
    """Independently reconstruct the frozen train-only preprocessing."""
    ends = np.r_[0, np.cumsum(cfg.block_lengths)]
    lo, hi = float(raw[:ends[1]].min()), float(raw[:ends[1]].max())
    blocks, metadata = [], []
    for left, right in zip(ends[:-1], ends[1:]):
        scaled = (raw[left:right]-lo)/(hi-lo)
        blocks.append(np.rint(np.clip(scaled, 0, 1)*(cfg.levels-1))/(cfg.levels-1))
        metadata.append({'start': int(left), 'stop': int(right),
                         'clipped': int(np.count_nonzero((scaled < 0) | (scaled > 1)))})
    return blocks, ends, {'training_min': lo, 'training_max': hi, 'blocks': metadata,
                         'levels': cfg.levels, 'unused_samples': len(raw)-int(ends[-1])}


def verify_laser(directory, summary, cfg, data):
    if digest(data) != DATA_HASH:
        raise ValueError('Unexpected measured laser source')
    raw = np.loadtxt(data, delimiter=',').reshape(-1)
    blocks, ends, preprocessing = laser_blocks(raw, cfg)
    if preprocessing != json.loads((directory/'preprocessing.json').read_text()):
        raise ValueError('Train-only laser preprocessing mismatch')
    origins = [np.arange(cfg.washout, n-max(cfg.horizons)) for n in cfg.block_lengths]
    picks = sampled_indices(len(origins[2]))
    scores = predictions = 0
    score_difference = prediction_difference = 0.
    for cell in summary['results']:
        with np.load(directory/cell['array_file']) as saved:
            np.testing.assert_array_equal(saved['test_indices'], origins[2])
            qrc, qelm, obs, coupling = architecture(cell['seed'], cell['gamma'], cfg)
            np.testing.assert_array_equal(saved['coupling'], coupling)
            features = {'qrc': qrc.run(blocks[2], obs, n_virtual=cfg.snapshots)[origins[2][picks]],
                        'qelm': qelm.features(delay_embed(blocks[2], cfg.window), obs,
                                              encoding_scheme='one_to_one')[origins[2][picks]]}
            for d in cfg.replay_windows:
                features[f'replay_d{d}'] = independent_replay(qrc, blocks[2], origins[2][picks], d, obs, cfg.snapshots)
            for horizon in cfg.horizons:
                target = raw[ends[2]+origins[2]+horizon]
                training = raw[origins[0]+horizon]
                sd = float(np.std(training))
                np.testing.assert_array_equal(saved[f'target_h{horizon}'], target)
                persistence = blocks[2][origins[2]]*(preprocessing['training_max']-preprocessing['training_min'])+preprocessing['training_min']
                np.testing.assert_array_equal(saved[f'persistence_h{horizon}'], persistence)
                for row in (r for r in cell['rows'] if r['horizon'] == horizon):
                    model, d = row['model'], row['window']
                    key = {'replay_refit': f'replay_d{d}', 'replay_frozen': f'frozen_d{d}'}.get(model, model)
                    score_difference = max(score_difference, check_score(row, saved[f'{key}_h{horizon}'], target, sd))
                    scores += 1
                for name, x in features.items():
                    key = f'{name}_h{horizon}'
                    prediction_difference = max(prediction_difference, check_predictions(saved, key, x, picks, laser=True))
                    np.testing.assert_allclose(saved[key+'_target_sd'], sd, rtol=1e-14)
                    np.testing.assert_allclose(saved[key+'_target_mean'], np.mean(training), rtol=1e-14)
                    predictions += len(picks)
                    if name.startswith('replay_d'):
                        frozen = name.replace('replay_', 'frozen_')+f'_h{horizon}'
                        prediction_difference = max(prediction_difference, check_predictions(
                            saved, f'qrc_h{horizon}', x, picks, laser=True, comparison_key=frozen))
                        predictions += len(picks)
        print(f'Verified laser root {cell["seed"]}, rate {cell["gamma"]}', flush=True)
    return {'roots': len(cfg.seeds), 'rate_cells': len(summary['results']),
            'score_rows_recomputed': scores, 'predictions_reconstructed_at_fixed_rows': predictions,
            'maximum_score_recomputation_difference': score_difference,
            'maximum_reconstructed_prediction_difference': prediction_difference,
            'all_true_targets_and_train_only_preprocessing_checked': True,
            'both_rates_both_horizons_all_replay_and_frozen_controls_retained': True,
            'data_sha256': digest(data), 'scope': 'Coupling variation conditional on one measured record'}


def paired(values, cfg):
    """Recalculate seed-wise differences without runner summary helpers."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    means = np.mean(values[rng.integers(0, len(values), size=(cfg.bootstrap_draws, len(values)))], axis=1)
    return {'n_seeds': len(values), 'differences': values.tolist(), 'mean': float(np.mean(values)),
            'median': float(np.median(values)), 'mean_ci95': np.percentile(means, [2.5, 97.5]).tolist(),
            'positive_seeds': int(np.sum(values > 0))}


def direction_gate(contrast):
    return {'positive_roots': contrast['positive_seeds'], 'total_roots': contrast['n_seeds'],
            'mean': contrast['mean'], 'mean_ci95': contrast['mean_ci95'],
            'passed': contrast['positive_seeds'] >= 20 and contrast['mean_ci95'][0] > 0}


def verify_directions(studies, cfgs):
    contrasts = {'narma': {}, 'laser': {}}
    differences = {key: [] for key in ('qelm_minus_qrc', 'replay_d4_minus_qrc', 'replay_d4_minus_d12')}
    for cell in sorted(studies['narma']['results'], key=lambda c: c['seed']):
        rows = {(r['model'], r['window']): r['nrmse'] for r in cell['rows']}
        differences['qelm_minus_qrc'].append(rows['qelm', 4]-rows['qrc', None])
        differences['replay_d4_minus_qrc'].append(rows['replay', 4]-rows['qrc', None])
        differences['replay_d4_minus_d12'].append(rows['replay', 4]-rows['replay', 12])
    contrasts['narma'] = {key: paired(values, cfgs['narma']) for key, values in differences.items()}
    for rate in cfgs['laser'].rates:
        cells = sorted((c for c in studies['laser']['results'] if c['gamma'] == rate), key=lambda c: c['seed'])
        for horizon in cfgs['laser'].horizons:
            differences = {'qelm_minus_qrc': [], 'restricted_d4_minus_qrc': []}
            for cell in cells:
                rows = {(r['model'], r['window']): r['nrmse'] for r in cell['rows'] if r['horizon'] == horizon}
                differences['qelm_minus_qrc'].append(rows['qelm', 4]-rows['qrc', None])
                differences['restricted_d4_minus_qrc'].append(rows['replay_refit', 4]-rows['qrc', None])
            contrasts['laser'][f'gamma{rate:g}_h{horizon}'] = {key: paired(v, cfgs['laser']) for key, v in differences.items()}
    for study in contrasts:
        if contrasts[study] != studies[study]['paired_contrasts']:
            raise ValueError('Independently recalculated paired contrast mismatch')
    primary = {'narma_qelm_minus_qrc': contrasts['narma']['qelm_minus_qrc'],
               'narma_replay4_minus_qrc': contrasts['narma']['replay_d4_minus_qrc'],
               'laser_fast_h1_qelm_minus_qrc': contrasts['laser']['gamma4_h1']['qelm_minus_qrc'],
               'laser_fast_h1_replay4_minus_qrc': contrasts['laser']['gamma4_h1']['restricted_d4_minus_qrc']}
    return {key: direction_gate(value) for key, value in primary.items()}


def verify(source, output):
    spec, studies = provenance(source)
    cfgs = replication.configs()
    checked = {'narma': verify_narma(source/'narma', studies['narma'], cfgs['narma']),
               'laser': verify_laser(source/'laser', studies['laser'], cfgs['laser'], ROOT/'data/santa_fe_laser/SF_Acont.dat')}
    gates = verify_directions(studies, cfgs)
    report = json.loads((source/'replication_report.json').read_text())
    expected_hashes = {name: digest(source/name/'summary.json') for name in studies}
    if report['direction_checks'] != gates or report['study_summaries_sha256'] != expected_hashes:
        raise ValueError('Replication report disagrees with independent verification')
    all_directions = all(g['passed'] for g in gates.values())
    if report['all_primary_directions_replicated'] != all_directions:
        raise ValueError('Incorrect overall direction check')
    if report['specification_sha256'] != digest(source/'replication_specification.json'):
        raise ValueError('Replication specification identity mismatch')
    result = {'verification_passed': True, 'studies': checked, 'direction_checks': gates,
              'all_primary_directions_replicated': all_directions,
              'specification_sha256': digest(source/'replication_specification.json'),
              'summary_sha256': expected_hashes, 'verifier_sha256': digest(Path(__file__)),
              'original_cohorts_kept_separate': spec['original_summaries'],
              'scope': 'Every stored test score and target; saved-readout/channel reconstruction sampled at five fixed rows per fitted model and cell. Direction checks are separate from verification success, not universal superiority or hardware validation.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.output), indent=2))
