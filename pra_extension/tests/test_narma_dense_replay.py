"""Dense replay preserves causal windows, physics, and the archived cohort."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import narma_dense_replay as dense
from laser_demonstration import architecture
from laser_forecast import replay_features


def test_shared_prefix_features_equal_independent_replays():
    cfg = replace(dense.replication.configs()["narma"], n_qubits=2, window=2,
                  snapshots=2, levels=5)
    qrc, _, observables, _ = architecture(814201, .3, cfg)
    block = np.random.default_rng(814202).choice(np.linspace(0, 1, 5), size=19)
    indices = np.array([5, 8, 12, 18])
    for window, features in dense.replay_feature_grid(qrc, block, indices, 6, observables, 2):
        original = replay_features(qrc, block, indices, window, observables, 2)
        direct = np.asarray([qrc.run(block[t-window+1:t+1], observables, n_virtual=2)[-1]
                             for t in indices])
        np.testing.assert_allclose(features, original, atol=2e-13, rtol=2e-12)
        np.testing.assert_allclose(features, direct, atol=2e-13, rtol=2e-12)


def test_dense_replay_ignores_older_inputs_and_future_inputs():
    cfg = replace(dense.replication.configs()["narma"], n_qubits=2, window=2,
                  snapshots=2, levels=5)
    qrc, _, obs, _ = architecture(814203, .2, cfg)
    block = np.random.default_rng(814204).choice(np.linspace(0, 1, 5), size=20)
    changed = block.copy()
    changed[:8] = 1 - changed[:8]
    changed[13:] = 1 - changed[13:]
    one = dict(dense.replay_feature_grid(qrc, block, np.array([12]), 5, obs, 2))
    two = dict(dense.replay_feature_grid(qrc, changed, np.array([12]), 5, obs, 2))
    for window in one:
        np.testing.assert_allclose(one[window], two[window], atol=1e-14, rtol=0)
    with pytest.raises(ValueError, match="Invalid replay"):
        next(dense.replay_feature_grid(qrc, block, np.array([3]), 5, obs, 2))


def test_archived_identity_and_anchor_guards():
    parent, cfg = dense.load_parent()
    assert cfg.seeds == tuple(range(921000, 921024))
    cell = parent["study_summaries"]["narma"]["results"][0]
    row = next(r for r in cell["rows"] if r["model"] == "replay")
    assert max(dense.compare_anchor(row, row).values()) == 0
    for key, value in (("gamma", 4.), ("alpha", 123.), ("features", 49)):
        with pytest.raises(ValueError):
            dense.compare_anchor(dict(row) | {key: value}, row)
    with pytest.raises(AssertionError):
        dense.compare_anchor(dict(row) | {"nrmse": row["nrmse"] + .001}, row)


def test_full_cohort_aggregation_rejects_missing_or_duplicate_rows():
    rows = [{"seed": seed, "window": d, "nrmse": (seed - 921000 + d) / 100}
            for seed in dense.replication.NARMA_ROOTS for d in dense.WINDOWS]
    summary = dense.aggregate_rows(rows)
    assert len(summary) == 48
    np.testing.assert_allclose([summary[0][k] for k in ("p10", "median", "p90")],
                               np.quantile(np.arange(1, 25) / 100, [.1, .5, .9]))
    with pytest.raises(ValueError, match="complete"):
        dense.aggregate_rows(rows[:-1])
    with pytest.raises(ValueError, match="complete"):
        dense.aggregate_rows(rows[:-1] + [rows[0]])


def test_portable_source_preserves_grid_anchors_and_selected_qrc():
    payload = dense.load_source()
    assert len(payload["rows"]) == 1152
    assert [r["window"] for r in payload["aggregates"]] == list(range(1, 49))
    assert all(r["n"] == 24 for r in payload["aggregates"])
    assert payload["verification"]["archived_window_comparisons"] == 144
    assert payload["verification"]["continuous_qrc_comparisons"] == 24
    assert payload["specification"]["numerical_reproducibility"]["first96_target_max_difference"] == 0
    changed = deepcopy(payload)
    changed["rows"][0]["gamma"] = 99
    with pytest.raises(ValueError, match="selection"):
        dense.validate_payload(changed)
    changed = deepcopy(payload)
    changed["rows"][0]["alpha"] = 123
    with pytest.raises(ValueError, match="selection"):
        dense.validate_payload(changed)
