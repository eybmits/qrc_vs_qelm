from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrc_vs_qelm.recurrence_fingerprint import (
    FingerprintLabel,
    FingerprintRule,
    classify_blind,
    estimate_cross_batch_signal_mse,
    sample_pauli_expectations,
)


SUFFIXES = (0, 1, 2, 3, 4, 5, 6, 8, 12, 16)
RULE = FingerprintRule(
    max_window=5,
    baseline_floor=1e-10,
    equivalence_ratio=0.03,
    liveness_ratio=0.08,
    recurrent_tail_ratio=0.08,
    reset_max_ratio=0.03,
    reset_contrast_ratio=0.08,
    tail_suffixes=(5, 6, 8, 12),
)
FINITE_D2 = (1.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
FINITE_D3 = (1.0, 0.45, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
RECURRENT = (1.0, 0.72, 0.54, 0.40, 0.35, 0.30, 0.24, 0.20, 0.16, 0.0)
ZEROS = (0.0,) * len(SUFFIXES)


def _scores(
    *,
    blind_id: str,
    carry_ratios: tuple[float, ...],
    reset_ratios: tuple[float, ...],
    shots_per_batch: str = "infinity",
    measurement_seed: int = 0,
    histories_per_physics: tuple[int, ...] = (12,),
) -> pd.DataFrame:
    if len(carry_ratios) != len(reset_ratios) or len(carry_ratios) != len(SUFFIXES):
        raise ValueError("synthetic curves must match the registered suffix grid")
    rows: list[dict[str, object]] = []
    for physics_id, n_histories in enumerate(histories_per_physics):
        for history_id in range(n_histories):
            scale = 1.0 + 0.01 * (physics_id - 1) + 0.001 * (
                history_id - (n_histories - 1) / 2
            )
            for condition, ratios in (
                ("carry", carry_ratios),
                ("ideal_quantum_reset", reset_ratios),
            ):
                for suffix_length, ratio in zip(SUFFIXES, ratios, strict=True):
                    signal = scale * float(ratio)
                    rows.append(
                        {
                            "blind_id": blind_id,
                            "physics_id": f"physics-{physics_id}",
                            "history_id": history_id,
                            "condition": condition,
                            "suffix_length": suffix_length,
                            "shots_per_batch": shots_per_batch,
                            "measurement_seed": measurement_seed,
                            "cross_batch_signal_mse": signal,
                            # This audit-only oracle column must never enter the
                            # architecture-blind decision.
                            "exact_signal_mse": 10_000.0 + signal,
                        }
                    )
    return pd.DataFrame(rows)


def _classify(scores: pd.DataFrame, *, seed: int = 123) -> pd.DataFrame:
    return classify_blind(scores, RULE, n_boot=300, bootstrap_seed=seed)


def test_smallest_finite_cutoff_is_inferred_without_declared_window() -> None:
    scores = _scores(
        blind_id="blind-a",
        carry_ratios=FINITE_D3,
        reset_ratios=FINITE_D3,
    )

    result = _classify(scores).iloc[0]

    assert result["classification"] == "FINITE_WINDOW_COMPATIBLE_d3"
    assert result["inferred_cutoff"] == 3
    assert bool(result["cutoff_compatible"]) is True
    assert result["inferred_cutoff_liveness_ratio_ci_low"] > RULE.liveness_ratio
    assert (
        result["inferred_cutoff_max_abs_ratio_ci_high"]
        < RULE.equivalence_ratio
    )


def test_sustained_tail_with_stable_reset_contrast_is_recurrent() -> None:
    scores = _scores(
        blind_id="blind-b",
        carry_ratios=RECURRENT,
        reset_ratios=ZEROS,
    )

    result = _classify(scores).iloc[0]

    assert result["classification"] == FingerprintLabel.RECURRENT_RESET_SENSITIVE
    assert pd.isna(result["inferred_cutoff"])
    assert bool(result["sustained_tail"]) is True
    assert bool(result["stable_reset_contrast"]) is True
    assert (
        result["recurrent_tail_mean_ratio_ci_low"]
        > RULE.recurrent_tail_ratio
    )
    assert result["reset_tail_max_abs_ratio_ci_high"] < RULE.reset_max_ratio
    assert result["reset_contrast_ratio_ci_low"] > RULE.reset_contrast_ratio


def test_suffix_outside_registered_tail_does_not_veto_recurrent_label() -> None:
    # L=16 is deliberately zero, but the locked recurrent tail is {5, 6, 8, 12}.
    assert RECURRENT[-1] == 0.0
    result = _classify(
        _scores(blind_id="blind-fixed-tail", carry_ratios=RECURRENT, reset_ratios=ZEROS)
    ).iloc[0]

    assert result["classification"] == FingerprintLabel.RECURRENT_RESET_SENSITIVE


def test_transient_tail_is_indeterminate() -> None:
    carry = (1.0, 0.6, 0.4, 0.2, 0.15, 0.20, 0.0, 0.0, 0.0, 0.0)
    result = _classify(
        _scores(blind_id="blind-indeterminate", carry_ratios=carry, reset_ratios=ZEROS)
    ).iloc[0]

    assert result["classification"] == FingerprintLabel.INDETERMINATE
    assert pd.isna(result["inferred_cutoff"])


def test_sustained_tail_that_survives_reset_is_reported_separately() -> None:
    reset = tuple(0.8 * value for value in RECURRENT)
    result = _classify(
        _scores(blind_id="blind-reset-noop", carry_ratios=RECURRENT, reset_ratios=reset)
    ).iloc[0]

    assert result["classification"] == FingerprintLabel.RECURRENT_RESET_INSENSITIVE
    assert bool(result["reset_tail_persists"]) is True


def test_zero_baseline_is_explicitly_low_signal() -> None:
    result = _classify(
        _scores(blind_id="blind-low", carry_ratios=ZEROS, reset_ratios=ZEROS)
    ).iloc[0]

    assert result["classification"] == FingerprintLabel.INDETERMINATE_LOW_SIGNAL
    assert bool(result["low_signal"]) is True
    assert bool(result["baseline_live"]) is False


def test_one_deterministic_label_per_blind_shot_seed_group() -> None:
    finite = _scores(
        blind_id="blind-finite",
        carry_ratios=FINITE_D2,
        reset_ratios=FINITE_D2,
        shots_per_batch="10000",
        measurement_seed=7,
    )
    recurrent = _scores(
        blind_id="blind-recurrent",
        carry_ratios=RECURRENT,
        reset_ratios=ZEROS,
        shots_per_batch="10000",
        measurement_seed=8,
    )
    combined = pd.concat([finite, recurrent], ignore_index=True)

    first = _classify(combined, seed=991)
    shuffled = combined.sample(frac=1.0, random_state=1234).reset_index(drop=True)
    second = _classify(shuffled, seed=991)

    assert len(first) == 2
    assert set(first["physics_id"]) == {"physics-0"}
    assert not first.duplicated(
        ["blind_id", "shots_per_batch", "measurement_seed"]
    ).any()
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_bootstrap_statistics_do_not_depend_on_case_pseudonym() -> None:
    first_scores = _scores(
        blind_id="blind-config-a",
        carry_ratios=RECURRENT,
        reset_ratios=ZEROS,
        shots_per_batch="10000",
        measurement_seed=7,
    )
    second_scores = _scores(
        blind_id="blind-config-b",
        carry_ratios=RECURRENT,
        reset_ratios=ZEROS,
        shots_per_batch="10000",
        measurement_seed=7,
    )

    result = _classify(
        pd.concat([first_scores, second_scores], ignore_index=True),
        seed=991,
    ).sort_values("blind_id", kind="mergesort")
    left = result.iloc[0].drop(labels=["blind_id"])
    right = result.iloc[1].drop(labels=["blind_id"])

    pd.testing.assert_series_equal(left, right, check_names=False, check_exact=True)


def test_a_blind_prediction_is_scoped_to_exactly_one_physics_system() -> None:
    mixed = _scores(
        blind_id="blind-mixed",
        carry_ratios=FINITE_D2,
        reset_ratios=FINITE_D2,
        histories_per_physics=(2, 2),
    )

    with pytest.raises(ValueError, match="exactly one physics_id"):
        _classify(mixed)


@pytest.mark.parametrize(
    "forbidden",
    [
        "model",
        "Declared_Window",
        "TRUE_LABEL",
        "architecture",
        "curve_model",
        "source_model",
        "adapter_type",
    ],
)
def test_forbidden_label_columns_are_rejected(forbidden: str) -> None:
    scores = _scores(
        blind_id="blind-c",
        carry_ratios=FINITE_D2,
        reset_ratios=FINITE_D2,
    )
    scores[forbidden] = "leak"

    with pytest.raises(ValueError, match="forbidden label columns"):
        _classify(scores)


def test_exact_oracle_column_is_ignored() -> None:
    scores = _scores(
        blind_id="blind-oracle",
        carry_ratios=RECURRENT,
        reset_ratios=ZEROS,
    )
    changed = scores.copy()
    changed["exact_signal_mse"] = -scores["exact_signal_mse"]

    pd.testing.assert_frame_equal(_classify(scores), _classify(changed))


def test_incomplete_or_duplicate_paired_trajectory_is_rejected() -> None:
    scores = _scores(
        blind_id="blind-d",
        carry_ratios=FINITE_D2,
        reset_ratios=FINITE_D2,
    )
    incomplete = scores.drop(
        scores[
            (scores["physics_id"] == "physics-0")
            & (scores["history_id"] == 0)
            & (scores["condition"] == "ideal_quantum_reset")
            & (scores["suffix_length"] == 6)
        ].index
    )
    duplicate = pd.concat([scores, scores.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="incomplete paired fingerprint trajectory"):
        _classify(incomplete)
    with pytest.raises(ValueError, match="duplicate rows"):
        _classify(duplicate)


def test_shot_helpers_are_seeded_bounded_and_do_not_truncate_signal() -> None:
    expected = np.array([-0.8, 0.0, 0.65])
    first = sample_pauli_expectations(
        expected, 400, rng=np.random.default_rng(991)
    )
    second = sample_pauli_expectations(
        expected, 400, rng=np.random.default_rng(991)
    )
    assert np.array_equal(first, second)
    assert np.all(first >= -1.0)
    assert np.all(first <= 1.0)
    assert np.allclose(first * 400 / 2, np.round(first * 400 / 2))

    x_a = np.array([0.8, -0.2, 0.1])
    x_b = np.array([0.3, -0.1, -0.4])
    noisy_a = estimate_cross_batch_signal_mse(
        x_a, x_b, 100, rng=np.random.default_rng(123)
    )
    noisy_b = estimate_cross_batch_signal_mse(
        x_a, x_b, 100, rng=np.random.default_rng(123)
    )
    assert noisy_a == noisy_b

    exact = estimate_cross_batch_signal_mse(
        x_a, x_b, None, rng=np.random.default_rng(123)
    )
    assert exact.signal_mse == pytest.approx(np.mean((x_a - x_b) ** 2))
    assert exact.raw_two_batch_mse == exact.signal_mse

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        sample_pauli_expectations([1.01], 100, rng=np.random.default_rng(1))
