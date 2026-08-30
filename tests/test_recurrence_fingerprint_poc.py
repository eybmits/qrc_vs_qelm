from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "recurrence_fingerprint_poc.py"

spec = importlib.util.spec_from_file_location("recurrence_fingerprint_poc", SCRIPT)
poc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = poc
spec.loader.exec_module(poc)


@pytest.fixture(scope="module")
def tiny_result():
    return poc.run(poc.PROFILES["tiny"])


def test_profile_validation_rejects_ambiguous_suffix_grids() -> None:
    profile = poc.PROFILES["tiny"]
    with pytest.raises(ValueError, match="unique and sorted"):
        replace(profile, suffix_lengths=(0, 2, 1, 3))
    with pytest.raises(ValueError, match="contain every declared window"):
        replace(profile, suffix_lengths=(0, 1, 2, 4, 6))
    with pytest.raises(ValueError, match="branch_length"):
        replace(profile, branch_length=1)


def test_tiny_physics_has_qelm_cutoffs_and_reset_sensitive_qrc_tail(tiny_result) -> None:
    scores, _, indices, verdict = tiny_result
    exact = scores[(scores["shots"] == poc.INFINITY)]

    qrc_carry = exact[
        (exact["model"] == poc.QRC_MODEL) & (exact["condition"] == "carry")
    ]
    qrc_tail = qrc_carry[qrc_carry["suffix_length"] >= 3]
    assert float(qrc_tail["exact_signal_mse"].mean()) > 1e-6

    qrc_reset = exact[
        (exact["model"] == poc.QRC_MODEL)
        & (exact["condition"] == "ideal_quantum_reset")
    ]
    assert np.array_equal(qrc_reset["exact_signal_mse"].to_numpy(), np.zeros(len(qrc_reset)))

    for model in ("spatial_qelm_d2", "replay_qelm_d3"):
        model_rows = exact[
            (exact["model"] == model) & (exact["condition"] == "carry")
        ]
        window = int(model_rows["declared_window"].iloc[0])
        before = model_rows[model_rows["suffix_length"] < window]
        after = model_rows[model_rows["suffix_length"] >= window]
        assert float(before["exact_signal_mse"].max()) > 1e-8
        assert np.array_equal(after["exact_signal_mse"].to_numpy(), np.zeros(len(after)))

        reset = exact[
            (exact["model"] == model)
            & (exact["condition"] == "ideal_quantum_reset")
        ].sort_values(["suffix_length", "coupling_seed", "pair_id"])
        carry = model_rows.sort_values(["suffix_length", "coupling_seed", "pair_id"])
        assert np.array_equal(
            carry["exact_signal_mse"].to_numpy(),
            reset["exact_signal_mse"].to_numpy(),
        )

    exact_classes = indices[indices["shots"] == poc.INFINITY].set_index("model")
    assert exact_classes.loc[poc.QRC_MODEL, "classification"] == "RECURRENT_MEMORY_DETECTED"
    assert exact_classes.loc["spatial_qelm_d2", "classification"] == "FINITE_WINDOW_COMPATIBLE"
    assert exact_classes.loc["replay_qelm_d3", "classification"] == "FINITE_WINDOW_COMPATIBLE"
    assert bool(exact_classes.loc["spatial_qelm_d2", "reset_removed"]) is False
    assert bool(exact_classes.loc["replay_qelm_d3", "reset_removed"]) is False
    assert verdict["exact_pass"] is True


def test_cross_batch_estimator_is_deterministic_and_untruncated() -> None:
    x_a = np.array([0.8, -0.2, 0.1])
    x_b = np.array([0.3, -0.1, -0.4])
    first = poc._cross_batch_estimate(
        x_a,
        x_b,
        tier=100,
        seed=123,
        parts=(1, 2, 3),
    )
    second = poc._cross_batch_estimate(
        x_a,
        x_b,
        tier=100,
        seed=123,
        parts=(1, 2, 3),
    )
    assert first == second
    exact, raw = poc._cross_batch_estimate(
        x_a,
        x_b,
        tier=None,
        seed=123,
        parts=(1, 2, 3),
    )
    assert exact == pytest.approx(np.mean((x_a - x_b) ** 2))
    assert raw == exact


def test_exact_binomial_pauli_sampler_is_bounded_and_unbiased() -> None:
    expected = np.array([-0.8, 0.0, 0.65])
    rng = np.random.default_rng(991)
    sampled = np.stack(
        [poc._sample_pauli_expectations(expected, 400, rng) for _ in range(4_000)]
    )
    assert np.all(sampled >= -1.0)
    assert np.all(sampled <= 1.0)
    assert np.allclose(sampled.mean(axis=0), expected, atol=0.01)
    assert np.allclose(sampled * 400 / 2, np.round(sampled * 400 / 2))


def test_outputs_are_pilot_only_and_well_formed(tiny_result, tmp_path: Path) -> None:
    scores, summary, indices, verdict = tiny_result
    paths = poc._write_outputs(
        out_dir=tmp_path,
        profile=poc.PROFILES["tiny"],
        scores=scores,
        summary=summary,
        indices=indices,
        verdict={**verdict, "dirty_tree_recorded": True},
        allow_dirty=True,
    )

    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    payload = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert payload["pilot_only"] is True
    assert payload["confirmatory_use"] == "forbidden"
    assert payload["confirmatory_seeds_touched"] is False
    assert payload["verdict"]["exact_pass"] is True
    assert payload["shot_model"].startswith("independent exact-binomial")
    assert set(payload["source_sha256"]) == {"runner", "tests"}

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest_payload = manifest["core"]["payload"]
    assert manifest_payload["pilot_only"] is True
    assert manifest_payload["confirmatory_use"] == "forbidden"
    assert manifest_payload["confirmatory_seeds_touched"] is False

    reloaded = pd.read_csv(paths["scores"])
    assert len(reloaded) == len(scores)
    assert set(reloaded["condition"]) == {"carry", "ideal_quantum_reset"}
