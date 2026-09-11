"""Causal indexing, matched resources, and exact history-intervention checks."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]/'code'
sys.path.insert(0, str(CODE))
import laser_forecast as forecast


def tiny_config():
    return replace(forecast.ForecastConfig(), n_qubits=2, window=2, snapshots=2,
                   block_lengths=(24,24,24), washout=8, replay_windows=(1,2,4,8))


def test_actual_future_targets_are_unquantized_and_never_cross_splits():
    cfg = tiny_config()
    raw = np.arange(72, dtype=float)+.123
    blocks, indices, targets, info = forecast.forecast_blocks(raw, cfg)
    for block_id, t in enumerate(indices):
        assert t[0] == 8 and t[-1] == 19
        for h in cfg.horizons:
            assert np.max(t+h) < 24
            np.testing.assert_array_equal(targets[h][block_id], raw[block_id*24+t+h])
    assert info['training_min'] == pytest.approx(.123)
    assert info['training_max'] == pytest.approx(23.123)
    altered = raw.copy()
    altered[24:] *= 10
    new_blocks, _, _, new_info = forecast.forecast_blocks(altered, cfg)
    np.testing.assert_array_equal(blocks[0], new_blocks[0])
    assert info['training_max'] == new_info['training_max']


@pytest.mark.parametrize('rate',[.2,4.])
@pytest.mark.parametrize('window',[1,2,4,8])
def test_batch_replay_matches_step_by_step_reference(rate, window):
    cfg = tiny_config()
    qrc, _, obs, _ = forecast.architecture(918000, rate, cfg)
    block = np.rint(np.random.default_rng(33).uniform(size=24)*32)/32
    indices = np.array([8,10,16,23])
    actual = forecast.replay_features(qrc, block, indices, window, obs, cfg.snapshots)
    expected = np.array([qrc.run(block[t-window+1:t+1], obs, n_virtual=cfg.snapshots)[-1]
                         for t in indices])
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)


def test_all_features_are_causal_and_restricted_features_ignore_old_inputs():
    cfg = tiny_config()
    qrc, qelm, obs, _ = forecast.architecture(918001, .2, cfg)
    block = np.rint(np.random.default_rng(3).uniform(size=24)*32)/32
    altered = block.copy()
    altered[17:] = 1-altered[17:]
    np.testing.assert_array_equal(qrc.run(block,obs,n_virtual=2)[:17],
                                  qrc.run(altered,obs,n_virtual=2)[:17])
    q1 = qelm.features(forecast.delay_embed(block,2),obs,encoding_scheme='one_to_one')
    q2 = qelm.features(forecast.delay_embed(altered,2),obs,encoding_scheme='one_to_one')
    np.testing.assert_array_equal(q1[:17],q2[:17])
    actual = forecast.replay_features(qrc,block,[16],4,obs,2)
    changed_past = block.copy()
    changed_past[:13] = 1-changed_past[:13]
    changed_past[17:] = 0
    np.testing.assert_array_equal(actual,forecast.replay_features(qrc,changed_past,[16],4,obs,2))


def test_illegal_replay_windows_rejected():
    cfg = tiny_config()
    qrc, _, obs, _ = forecast.architecture(918000, .2, cfg)
    with pytest.raises(ValueError,match='outside'):
        forecast.replay_features(qrc,np.ones(10),[2],4,obs,2)


def test_training_target_scaling_and_fit_do_not_use_test_values():
    rng = np.random.default_rng(31)
    features = [rng.normal(size=(n,3)) for n in (64,24,32)]
    targets = [100+12*x[:,0] for x in features]
    one = forecast.fit_forecaster(features,targets)
    other = forecast.fit_forecaster(features,[targets[0],targets[1],targets[2]*1e6])
    np.testing.assert_array_equal(one[0].weights,other[0].weights)
    np.testing.assert_array_equal(one[3],other[3])
    assert one[1] == np.mean(targets[0])
    assert one[2] == np.std(targets[0])
    assert np.sqrt(np.mean((one[3]-targets[2])**2)) < 1e-6


def test_config_is_predeclared_and_primary_resources_match():
    cfg=forecast.ForecastConfig()
    assert cfg.horizons==(1,4) and cfg.replay_windows==(1,2,4,8,16,32)
    assert cfg.seeds==tuple(range(918000,918012))
    fast, qelm, obs, jf=forecast.architecture(cfg.seeds[0],4.,cfg)
    slow, _, _, js=forecast.architecture(cfg.seeds[0],.2,cfg)
    np.testing.assert_array_equal(jf,js)
    assert len(obs)*cfg.snapshots==48 and fast.n_qubits==qelm.n_qubits==4
    assert forecast.physical_check(slow,np.linspace(0,1,8))['passed']


def test_bootstrap_is_paired_reproducible_and_keeps_negative_effects():
    cfg=forecast.ForecastConfig()
    values=np.linspace(-.3,-.1,12)
    one=forecast.paired_summary(values,cfg)
    assert one==forecast.paired_summary(values,cfg)
    assert one['mean_ci95'][1]<0 and one['positive_seeds']==0
    with pytest.raises(ValueError,match='Incomplete'):
        forecast.summarize([],cfg)
