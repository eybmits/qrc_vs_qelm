"""Prospective fresh-root contract and independently calculated verification."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import laser_forecast
import narma_memory
import task_memory_replication as replication
import verify_task_memory_replication as verify


def narma_summary():
    cfg = replace(replication.configs()['narma'], seeds=(921000,))
    candidates = [{'model': model, 'gamma': rate,
                   'window': 4 if model == 'qelm' else None,
                   'alpha': 1e-4, 'validation_mse': float(index+1),
                   'nrmse': float(10-index)}
                  for model in ('qrc', 'qelm') for index, rate in enumerate(cfg.rates)]
    selected = [next(row.copy() for row in candidates if row['model'] == model)
                for model in ('qrc', 'qelm')]
    selected += [{'model': model, 'window': d, 'gamma': cfg.rates[0] if model == 'replay' else None}
                 for model, d in verify.expected_keys('narma', cfg)
                 if model not in ('qrc', 'qelm')]
    cell = {'seed': cfg.seeds[0], 'rows': selected, 'candidate_rows': candidates,
            'selection': {'qrc': cfg.rates[0], 'qelm': cfg.rates[0]},
            'scored_rows': [n-cfg.washout-1 for n in cfg.block_lengths],
            'physical_check': {'passed': True}}
    return cfg, {'results': [cell]}


def test_fresh_configs_change_only_declared_roots_and_bootstrap():
    cfgs = replication.configs()
    for name, original in [('narma', narma_memory.NarmaConfig()), ('laser', laser_forecast.ForecastConfig())]:
        changed, old = asdict(cfgs[name]), asdict(original)
        differences = {key for key in old if old[key] != changed[key]}
        assert differences == {'seeds', 'bootstrap_seed'}
        assert len(cfgs[name].seeds) == 24
        assert not set(cfgs[name].seeds) & set(original.seeds)
    assert cfgs['narma'].seeds == tuple(range(921000, 921024))
    assert cfgs['laser'].seeds == tuple(range(922000, 922024))
    assert cfgs['laser'].rates == (4., .2) and cfgs['laser'].horizons == (1, 4)


def test_coverage_rejects_missing_duplicate_roots_and_candidates():
    cfg, summary = narma_summary()
    verify.check_coverage(summary, 'narma', cfg)
    missing = deepcopy(summary)
    missing['results'][0]['candidate_rows'].pop()
    with pytest.raises(ValueError, match='candidates'):
        verify.check_coverage(missing, 'narma', cfg)
    duplicated = deepcopy(summary)
    duplicated['results'].append(deepcopy(duplicated['results'][0]))
    with pytest.raises(ValueError, match='cohort'):
        verify.check_coverage(duplicated, 'narma', cfg)
    with pytest.raises(ValueError, match='cohort'):
        verify.check_coverage({'results': []}, 'narma', cfg)


def test_selection_is_validation_only_and_history_dynamics_stay_fixed():
    cfg, summary = narma_summary()
    # The selected candidate intentionally has the worst test error.
    verify.check_coverage(summary, 'narma', cfg)
    tampered = deepcopy(summary)
    cell = tampered['results'][0]
    cell['selection']['qrc'] = cfg.rates[-1]
    with pytest.raises(ValueError, match='validation minimum'):
        verify.check_coverage(tampered, 'narma', cfg)
    tampered = deepcopy(summary)
    next(row for row in tampered['results'][0]['rows'] if row['model'] == 'replay')['gamma'] = 4.
    with pytest.raises(ValueError, match='dynamics'):
        verify.check_coverage(tampered, 'narma', cfg)


def test_negative_and_secondary_laser_settings_are_required():
    cfg = replace(replication.configs()['laser'], seeds=(922000,))
    keys = verify.expected_keys('laser', cfg)
    assert len(keys) == 30
    rows = [{'model': m, 'window': d, 'horizon': h} for m, d, h in keys]
    cells = [{'seed': cfg.seeds[0], 'gamma': rate, 'rows': deepcopy(rows),
              'scored_rows': [n-cfg.washout-4 for n in cfg.block_lengths],
              'physical_check': {'passed': True}} for rate in cfg.rates]
    verify.check_coverage({'results': cells}, 'laser', cfg)
    with pytest.raises(ValueError, match='cohort'):
        verify.check_coverage({'results': cells[:1]}, 'laser', cfg)
    cells[1]['rows'] = [r for r in cells[1]['rows'] if r['horizon'] != 4]
    with pytest.raises(ValueError, match='settings'):
        verify.check_coverage({'results': cells}, 'laser', cfg)


def test_scores_are_bound_to_actual_targets_and_training_denominator():
    truth = np.array([.2, .4, .6])
    predicted = truth + np.array([.1, -.1, .1])
    assert verify.check_score({'rmse': .1, 'nrmse': .5}, predicted, truth, .2) < 1e-14
    with pytest.raises(AssertionError):
        verify.check_score({'rmse': .1, 'nrmse': .4}, predicted, truth, .2)
    with pytest.raises(ValueError, match='shape'):
        verify.check_score({'rmse': .1, 'nrmse': .5}, predicted[:2], truth, .2)
    with pytest.raises(ValueError, match='scale'):
        verify.check_score({'rmse': .1, 'nrmse': .5}, predicted, truth, 0.)


def test_saved_readout_reconstruction_and_fixed_sampling():
    x = np.array([[1., 2.], [3., 4.]])
    saved = {'qrc_mean': np.array([1., 1.]), 'qrc_scale': np.array([2., 1.]),
             'qrc_weights': np.array([.5, 2.]), 'qrc_intercept': .3,
             'qrc_target_sd': .2, 'qrc_target_mean': 3.}
    saved['qrc'] = np.array([3.46, 4.36])
    assert verify.check_predictions(saved, 'qrc', x, np.array([0, 1])) < 1e-14
    saved['qrc'][0] += .01
    with pytest.raises(AssertionError):
        verify.check_predictions(saved, 'qrc', x, np.array([0, 1]))
    assert verify.sampled_indices(1919).tolist() == [0, 479, 959, 1438, 1918]


def test_independent_laser_preprocessing_uses_training_only():
    cfg = replace(replication.configs()['laser'], block_lengths=(10, 10, 10))
    raw = np.arange(30, dtype=float)
    blocks, ends, info = verify.laser_blocks(raw, cfg)
    altered = raw.copy()
    altered[10:] *= 100
    other, _, new_info = verify.laser_blocks(altered, cfg)
    np.testing.assert_array_equal(blocks[0], other[0])
    assert ends.tolist() == [0, 10, 20, 30]
    assert info['training_min'] == new_info['training_min'] == 0
    assert info['training_max'] == new_info['training_max'] == 9
    assert info['blocks'][2]['clipped'] == 10


def test_independent_narma_recurrence_has_correct_lag_and_no_future_access():
    raw = np.zeros(24)
    target = verify.independent_target(raw, 10, .2)
    assert target[9] == 0 and target[10] == pytest.approx(.1)
    assert target[11] == pytest.approx(.1305)
    raw[0] = raw[9] = 1.
    changed = verify.independent_target(raw, 10, .2)
    assert changed[10] == pytest.approx(.16)
    raw[10:] = 1.
    assert verify.independent_target(raw, 10, .2)[10] == changed[10]


def test_direction_gate_requires_both_prespecified_conditions():
    good = {'positive_seeds': 20, 'n_seeds': 24, 'mean': .1, 'mean_ci95': [.01, .2]}
    assert verify.direction_gate(good)['passed']
    assert not verify.direction_gate({**good, 'positive_seeds': 19})['passed']
    assert not verify.direction_gate({**good, 'mean_ci95': [0., .2]})['passed']
    cfg = replication.configs()['narma']
    negative = verify.paired(np.linspace(-.3, -.1, 24), cfg)
    assert negative['positive_seeds'] == 0 and negative['mean_ci95'][1] < 0
    assert not verify.direction_gate(negative)['passed']
    assert negative == verify.paired(np.linspace(-.3, -.1, 24), cfg)
