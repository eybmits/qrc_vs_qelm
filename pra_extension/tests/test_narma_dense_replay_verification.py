"""Independent extension audits reject leakage and altered numerical evidence."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import narma_memory
import verify_narma_dense_replay as verify
from laser_forecast import fit_forecaster, parameter_arrays, score_prediction


def fitted_example():
    rng = np.random.default_rng(880731)
    features = [rng.normal(loc=i, size=(n, 4)) for i, n in enumerate((70, 31, 29))]
    targets = [x @ np.array([.2, -.3, .1, .6]) + rng.normal(scale=.2, size=len(x)) for x in features]
    fit, mean, sd, prediction = fit_forecaster(features, targets)
    saved = {"replay_d4": prediction, **parameter_arrays("replay_d4", fit, mean, sd)}
    row = {"model": "replay", "window": 4, "gamma": .3, "features": 4,
           "alpha": fit.alpha, "validation_mse": fit.validation_mse,
           **score_prediction(prediction, targets[2], sd)}
    return features, targets, saved, row


def test_independent_rng_targets_quantization_and_split_indices_match_contract():
    cfg = replace(verify.replication.configs()["narma"], block_lengths=(87, 70, 79), washout=48)
    independent = verify.independent_data(921007, cfg)
    original = narma_memory.make_data(921007, cfg)
    for group, expected_group in zip(independent, original, strict=True):
        for actual, expected in zip(group, expected_group, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    for n, indices in zip(cfg.block_lengths, independent[2], strict=True):
        np.testing.assert_array_equal(indices, np.arange(cfg.washout, n-1))
    altered = replace(cfg, block_lengths=(87, 70, 89))
    changed = verify.independent_data(921007, altered)
    for index in (0, 1):
        np.testing.assert_array_equal(independent[0][index], changed[0][index])
        np.testing.assert_array_equal(independent[3][index], changed[3][index])


def test_complete_score_check_rejects_wrong_scale_and_single_corrupted_prediction():
    _, targets, saved, row = fitted_example()
    assert verify.check_score(row, saved["replay_d4"], targets[2], targets[0].std()) < 1e-14
    with pytest.raises(AssertionError):
        verify.check_score(row, saved["replay_d4"], targets[2], targets[2].std())
    changed = saved["replay_d4"].copy()
    changed[-1] += .01
    with pytest.raises(AssertionError):
        verify.check_score(row, changed, targets[2], targets[0].std())


def test_fit_audit_checks_training_scaling_validation_minimum_and_normal_equations():
    features, targets, saved, row = fitted_example()
    audit = verify.check_fit(saved, "replay_d4", row, features, targets)
    assert audit["ridge_backward_error"] < 1e-14
    assert len(audit["candidate_validation_scores"]) == len(verify.ALPHAS)
    assert verify.check_prediction(saved, "replay_d4", features[2]) < 1e-14
    corrupted = dict(saved, replay_d4_mean=np.mean(np.vstack(features), axis=0))
    with pytest.raises(AssertionError):
        verify.check_fit(corrupted, "replay_d4", row, features, targets)
    changed_alpha = next(a for a in verify.ALPHAS if a != row["alpha"])
    corrupted = dict(saved, replay_d4_alpha=np.asarray(changed_alpha))
    with pytest.raises(ValueError, match="validation minimum"):
        verify.check_fit(corrupted, "replay_d4", dict(row, alpha=changed_alpha), features, targets)
    corrupted = dict(saved, replay_d4_weights=saved["replay_d4_weights"] + 1.)
    with pytest.raises((AssertionError, ValueError)):
        verify.check_fit(corrupted, "replay_d4", row, features, targets)


def test_readout_training_and_validation_audit_has_no_test_data_dependency():
    features, targets, saved, row = fitted_example()
    original = verify.check_fit(saved, "replay_d4", row, features, targets)
    changed_features = features[:2] + [features[2] * 1e6]
    changed_targets = targets[:2] + [targets[2] + 1e6]
    assert verify.check_fit(saved, "replay_d4", row, changed_features, changed_targets) == original


def test_archive_tolerance_never_relaxes_same_run_scores_or_discrete_settings():
    _, targets, saved, row = fitted_example()
    changed = dict(row, nrmse=row["nrmse"] + verify.ARCHIVE_ATOL / 2)
    assert verify.check_archive(changed, row)["nrmse"] > 0
    with pytest.raises(AssertionError):
        verify.check_score(changed, saved["replay_d4"], targets[2], targets[0].std())
    with pytest.raises(AssertionError):
        verify.check_archive(dict(row, nrmse=row["nrmse"] + 2 * verify.ARCHIVE_ATOL), row)
    for key, value in (("alpha", .123), ("window", 5), ("gamma", 1.), ("features", 5)):
        with pytest.raises(ValueError, match="Archived"):
            verify.check_archive(dict(row, **{key: value}), row)
