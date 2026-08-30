import hashlib
import json
from pathlib import Path
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from qrc_vs_qelm.canonical_gates import (
    _sha256_json,
    evaluate_canonical_gates,
    load_strict_json,
    verify_canonical_manifest,
    verify_pytest_junit,
)
from qrc_vs_qelm.canonical_emulation import classify_emulation_regime
from qrc_vs_qelm.canonical_resources import expected_observable_contract


MODELS = (
    ("rr_same_identity", None),
    ("constant", None),
    ("linear_delay", None),
    ("poly2_delay", None),
    ("ptd_parallel", 1),
    ("ptd_parallel", 2),
    ("ptd_parallel", 4),
    ("classical_tanh", 1),
    ("classical_tanh", 2),
    ("classical_tanh", 4),
)


def _profile(*, confirmatory: bool = True) -> dict:
    return {
        "study": "canonical_qrc_to_parallel_qelm_emulation",
        "profile": "confirmatory" if confirmatory else "calibration",
        "confirmatory": confirmatory,
        "roots": [{"root_id": f"root{index:02d}"} for index in range(1, 13)],
        "teacher_n_qubits": [2, 3],
        "teacher_dissipators": ["local_loss"],
        "teacher_gammas": [0.0, 0.2, 0.5, 1.0, 2.0, 4.0],
        "teacher": {
            "dt": 0.5,
            "gammas": [0.0, 0.2, 0.5, 1.0, 2.0, 4.0],
            "target_observables": {"axes": ["x", "y", "z"], "max_weight": 2},
        },
        "student": {
            "n_qubits": 6,
            "observables": {"axes": ["x", "y", "z"], "max_weight": 1},
        },
        "windows": [1, 2, 4, 6],
        "snapshot_budgets": [1, 2, 4],
        "max_snapshots": 4,
        "suffix_groups": 32,
        "prefix_branches": 4,
        "min_target_variance": 1e-8,
        "tolerance_nrmse": 0.10,
        "decision": {
            "tolerance_nrmse": 0.10,
            "min_target_variance": 1e-8,
            "min_root_successes": 10,
            "roots_total": 12,
            "min_spearman": 0.80,
            "min_g2_live_systems_per_family": 5,
            "min_window_gain": 0.20,
            "min_feature_gain": 0.20,
            "min_best_classical_gain": 0.10,
            "min_scaling_r_squared": 0.85,
            "g4_gammas": [2.0, 4.0],
            "g5_gamma": 4.0,
        },
    }


def _history(gamma: float, window: int) -> float:
    if gamma == 0.0:
        return {1: 0.40, 2: 0.25, 4: 0.12, 6: 0.06}[int(window)]
    return float(0.5 * np.exp(-0.4 * gamma * 0.5 * window))


def _stream_nrmse(model: str, budget: int | None, gamma: float, window: int) -> float:
    if model == "rr_same_identity":
        return float(0.05 + _history(gamma, window))
    if model == "ptd_parallel" and budget == 4 and gamma in {2.0, 4.0}:
        return {1: 1.0, 2: 0.82, 4: 0.65, 6: 0.50}[int(window)]
    if gamma == 4.0 and window == 6:
        if model == "linear_delay":
            return 0.90
        if model == "poly2_delay":
            return 0.75
        if model == "classical_tanh" and budget == 4:
            return 0.80
    return 0.70


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    profile = _profile()
    student_contract = expected_observable_contract(
        profile["student"]["n_qubits"],
        profile["student"]["observables"],
    )
    stream_rows: list[dict] = []
    suffix_rows: list[dict] = []
    system_rows: list[dict] = []
    for root in profile["roots"]:
        root_id = root["root_id"]
        for teacher_n in profile["teacher_n_qubits"]:
            target_contract = expected_observable_contract(
                teacher_n,
                profile["teacher"]["target_observables"],
            )
            for gamma in profile["teacher_gammas"]:
                system_id = f"{root_id}_n{teacher_n}_g{gamma:.1f}"
                teacher_seed = 10_000 + teacher_n
                system_rows.append(
                    {
                        "root_id": root_id,
                        "system_id": system_id,
                        "teacher_n_qubits": teacher_n,
                        "teacher_dissipator": "local_loss",
                        "teacher_gamma": gamma,
                        "teacher_coupling_seed": teacher_seed,
                        "target_dim": target_contract["count"],
                        "target_observables_sha256": target_contract[
                            "names_sha256"
                        ],
                        "target_variance": 1.0,
                        "target_live": True,
                    }
                )
                for window in profile["windows"]:
                    history = _history(gamma, window)
                    for model, budget in MODELS:
                        stream_nrmse = _stream_nrmse(model, budget, gamma, window)
                        base = {
                            "root_id": root_id,
                            "system_id": system_id,
                            "teacher_n_qubits": teacher_n,
                            "teacher_dissipator": "local_loss",
                            "teacher_gamma": gamma,
                            "teacher_coupling_seed": teacher_seed,
                            "window": window,
                            "model": model,
                            "snapshot_budget": budget,
                            "target_observables_sha256": target_contract[
                                "names_sha256"
                            ],
                            "student_n_qubits": profile["student"]["n_qubits"],
                            "student_observables_sha256": student_contract[
                                "names_sha256"
                            ],
                        }
                        if model == "rr_same_identity":
                            feature_width = target_contract["count"]
                        elif model == "constant":
                            feature_width = 0
                        elif model == "linear_delay":
                            feature_width = window
                        elif model == "poly2_delay":
                            feature_width = window + window * (window + 1) // 2
                        else:
                            feature_width = int(budget) * student_contract["count"]
                        stream_rows.append(
                            {
                                **base,
                                "feature_budget": feature_width,
                                "feature_dim": feature_width,
                                "mse": stream_nrmse**2,
                                "rmse": stream_nrmse,
                                "nrmse": stream_nrmse,
                                "mean_row_rms": stream_nrmse,
                                "median_row_rms": stream_nrmse,
                                "max_row_rms": stream_nrmse,
                                "alpha": (
                                    1e-6 if model not in {"rr_same_identity", "constant"} else None
                                ),
                                "intercept_only_selected": (
                                    False if model not in {"rr_same_identity", "constant"} else None
                                ),
                                "validation_nrmse": (
                                    stream_nrmse
                                    if model not in {"rr_same_identity", "constant"}
                                    else None
                                ),
                                "feature_rank": (
                                    feature_width
                                    if model not in {"rr_same_identity", "constant"}
                                    else None
                                ),
                                "effective_rank": (
                                    float(feature_width)
                                    if model not in {"rr_same_identity", "constant"}
                                    else None
                                ),
                                "condition_number": (
                                    1.0
                                    if model not in {"rr_same_identity", "constant"}
                                    else None
                                ),
                                "target_aligned_fraction": (
                                    0.5
                                    if model not in {"rr_same_identity", "constant"}
                                    else None
                                ),
                            }
                        )
                        representation = 0.0 if model == "rr_same_identity" else 0.40
                        if model == "ptd_parallel" and gamma == 4.0 and window == 6:
                            representation = {1: 0.50, 2: 0.40, 4: 0.30}[int(budget)]
                        history_mse = history**2
                        representation_mse = representation**2
                        total_mse = history_mse + representation_mse
                        total_nrmse = float(np.hypot(history, representation))
                        oracle_nrmse = history
                        suffix_rows.append(
                            {
                                **base,
                                "feature_budget": feature_width,
                                "n_suffix_groups": profile["suffix_groups"],
                                "branches_per_group": profile["prefix_branches"],
                                "n_outputs": target_contract["count"],
                                "history_mse": history_mse,
                                "representation_mse": representation_mse,
                                "total_mse": total_mse,
                                "history_nrmse": history,
                                "representation_nrmse": representation,
                                "total_nrmse": total_nrmse,
                                "history_fraction": (
                                    history_mse / total_mse if total_mse > 0.0 else 0.0
                                ),
                                "oracle_nrmse": oracle_nrmse,
                                "regime": classify_emulation_regime(
                                    history_nrmse=history,
                                    student_nrmse=total_nrmse,
                                    oracle_nrmse=oracle_nrmse,
                                    target_variance=1.0,
                                    tolerance=profile["tolerance_nrmse"],
                                    min_target_variance=profile["min_target_variance"],
                                ),
                                "mean_group_diameter": 0.15,
                                "target_variance": 1.0,
                                "identity_error": 0.0,
                                "prediction_disagreement": 0.0,
                                "max_group_diameter": 0.2,
                                "half_max_group_diameter": 0.1,
                                "rms_pair_half_bound": 0.08,
                                "repeated_suffix_rmse": 0.40,
                                "worst_branch_rmse": 0.50,
                                "predictions_agree": True,
                                "rms_bound_slack": 0.32,
                                "diameter_bound_slack": 0.40,
                                "rms_half_bound_satisfied": True,
                                "diameter_half_bound_satisfied": True,
                                "suffix_bounds_satisfied": True,
                            }
                        )
    return (
        pd.DataFrame(stream_rows),
        pd.DataFrame(suffix_rows),
        pd.DataFrame(system_rows),
        profile,
    )


def _set_stream_nrmse(
    stream: pd.DataFrame,
    mask: pd.Series,
    value: float | Callable[[pd.Series], float],
) -> None:
    rows = stream.loc[mask]
    updated = (
        rows.apply(value, axis=1).to_numpy(dtype=float)
        if callable(value)
        else np.full(len(rows), float(value), dtype=float)
    )
    stream.loc[mask, "nrmse"] = updated
    stream.loc[mask, "mse"] = updated**2
    stream.loc[mask, "rmse"] = updated
    stream.loc[mask, "mean_row_rms"] = updated
    stream.loc[mask, "median_row_rms"] = updated
    stream.loc[mask, "max_row_rms"] = updated


def _normalized_from_mse(mse: np.ndarray, variance: np.ndarray) -> np.ndarray:
    output = np.empty_like(mse, dtype=float)
    live = variance > 0.0
    output[live] = np.sqrt(mse[live] / variance[live])
    output[(~live) & (mse > 0.0)] = np.inf
    output[(~live) & (mse == 0.0)] = 0.0
    return output


def _refresh_suffix_groups(
    suffix: pd.DataFrame,
    selected_keys: pd.DataFrame,
    profile: dict,
) -> None:
    key_columns = ["root_id", "system_id", "window"]
    for key in selected_keys[key_columns].drop_duplicates().itertuples(index=False):
        group_mask = (
            (suffix["root_id"].astype(str) == str(key.root_id))
            & (suffix["system_id"].astype(str) == str(key.system_id))
            & (suffix["window"].astype(int) == int(key.window))
        )
        variance = suffix.loc[group_mask, "target_variance"].to_numpy(dtype=float)
        history_mse = suffix.loc[group_mask, "history_mse"].to_numpy(dtype=float)
        representation_mse = suffix.loc[
            group_mask, "representation_mse"
        ].to_numpy(dtype=float)
        total_mse = history_mse + representation_mse
        suffix.loc[group_mask, "total_mse"] = total_mse
        suffix.loc[group_mask, "identity_error"] = 0.0
        suffix.loc[group_mask, "history_nrmse"] = _normalized_from_mse(
            history_mse,
            variance,
        )
        suffix.loc[group_mask, "representation_nrmse"] = _normalized_from_mse(
            representation_mse,
            variance,
        )
        suffix.loc[group_mask, "total_nrmse"] = _normalized_from_mse(
            total_mse,
            variance,
        )
        fraction = np.zeros_like(total_mse)
        positive = total_mse > 0.0
        fraction[positive] = history_mse[positive] / total_mse[positive]
        suffix.loc[group_mask, "history_fraction"] = fraction
        oracle_row = suffix.loc[
            group_mask
            & (suffix["model"] == "rr_same_identity")
            & suffix["snapshot_budget"].isna()
        ]
        assert len(oracle_row) == 1
        oracle_nrmse = float(oracle_row.iloc[0]["total_nrmse"])
        suffix.loc[group_mask, "oracle_nrmse"] = oracle_nrmse
        for index in suffix.index[group_mask]:
            row = suffix.loc[index]
            suffix.loc[index, "regime"] = classify_emulation_regime(
                history_nrmse=float(row["history_nrmse"]),
                student_nrmse=float(row["total_nrmse"]),
                oracle_nrmse=float(row["oracle_nrmse"]),
                target_variance=float(row["target_variance"]),
                tolerance=profile["tolerance_nrmse"],
                min_target_variance=profile["min_target_variance"],
            )


def _set_suffix_history(
    suffix: pd.DataFrame,
    mask: pd.Series,
    value: float | Callable[[pd.Series], float],
    profile: dict,
) -> None:
    keys = suffix.loc[mask, ["root_id", "system_id", "window"]].drop_duplicates()
    for key in keys.itertuples(index=False):
        group_mask = (
            (suffix["root_id"].astype(str) == str(key.root_id))
            & (suffix["system_id"].astype(str) == str(key.system_id))
            & (suffix["window"].astype(int) == int(key.window))
        )
        row = suffix.loc[group_mask].iloc[0]
        updated = float(value(row) if callable(value) else value)
        variance = float(row["target_variance"])
        suffix.loc[group_mask, "history_mse"] = updated**2 * variance
    _refresh_suffix_groups(suffix, keys, profile)


def _set_suffix_representation(
    suffix: pd.DataFrame,
    mask: pd.Series,
    value: float,
    profile: dict,
) -> None:
    variance = suffix.loc[mask, "target_variance"].to_numpy(dtype=float)
    suffix.loc[mask, "representation_mse"] = float(value) ** 2 * variance
    keys = suffix.loc[mask, ["root_id", "system_id", "window"]]
    _refresh_suffix_groups(suffix, keys, profile)


def _make_suffix_groups_degenerate(
    suffix: pd.DataFrame,
    mask: pd.Series,
    profile: dict,
) -> None:
    keys = suffix.loc[mask, ["root_id", "system_id", "window"]].drop_duplicates()
    for key in keys.itertuples(index=False):
        group_mask = (
            (suffix["root_id"].astype(str) == str(key.root_id))
            & (suffix["system_id"].astype(str) == str(key.system_id))
            & (suffix["window"].astype(int) == int(key.window))
        )
        suffix.loc[group_mask, "target_variance"] = 0.0
    _refresh_suffix_groups(suffix, keys, profile)


def _evidence(**overrides: bool) -> dict[str, bool]:
    evidence = {
        "tests_passed": True,
        "strict_json": True,
        "manifest_hashes_valid": True,
        "freeze_identity_matches": True,
        "seed_namespaces_disjoint": True,
        "duplicate_smoke_deterministic": True,
    }
    evidence.update(overrides)
    return evidence


def test_all_primary_gates_secondary_safeguard_and_p6_pass():
    stream, suffix, systems, profile = _tables()

    report, contrasts = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    assert all(gate["passed"] for gate in report["primary_gates"].values())
    assert report["secondary_qelm_vs_best_classical"]["passed"]
    assert report["descriptive_qelm_vs_tanh"]["passed"]
    assert report["secondary_p6_forgetting_scaling"]["passed"]
    assert report["decision"] == "CONFIRMS_FINITE_RESOURCE_FRONTIER_QELM_SPECIFIC"
    assert set(contrasts["gate_id"]) == {
        "G2",
        "G4",
        "G5",
        "SECONDARY_TANH_DESCRIPTIVE",
        "SECONDARY_BEST_CLASSICAL",
        "P6",
    }
    assert contrasts.loc[contrasts["gate_id"] == "P6", "passed"].all()


def test_best_classical_gate_prevents_tanh_only_quantum_specific_label():
    stream, suffix, systems, profile = _tables()
    mask = (
        (stream["model"] == "poly2_delay")
        & (stream["teacher_gamma"] == 4.0)
        & (stream["window"] == 6)
    )
    _set_stream_nrmse(stream, mask, 0.40)

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    assert report["descriptive_qelm_vs_tanh"]["passed"]
    assert not report["secondary_qelm_vs_best_classical"]["passed"]
    assert report["decision"] == "CONFIRMS_FINITE_RESOURCE_FRONTIER"


def test_g2_fails_closed_when_three_roots_have_one_undefined_live_correlation():
    stream, suffix, systems, profile = _tables()
    mask = (
        stream["root_id"].isin(["root01", "root02", "root03"])
        & (stream["model"] == "rr_same_identity")
        & (stream["teacher_n_qubits"] == 2)
        & (stream["teacher_gamma"] == 0.0)
    )
    _set_stream_nrmse(stream, mask, 0.25)

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    assert report["primary_gates"]["G2"]["successful_roots"] == 9
    assert report["primary_gates"]["G2"]["root_median_spearman"]["root01"] is None
    assert report["decision"] == "NO_GO"


def test_g2_and_g6_report_partial_defined_root_medians_and_intervals():
    stream, suffix, systems, profile = _tables()
    mask = (
        stream["root_id"].isin(["root01", "root02"])
        & (stream["model"] == "rr_same_identity")
        & (stream["teacher_n_qubits"] == 2)
        & (stream["teacher_gamma"] == 0.0)
    )
    _set_stream_nrmse(stream, mask, 0.25)

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    g2 = report["primary_gates"]["G2"]
    assert g2["defined_roots"] == 10
    assert g2["median_root_median_spearman"] is not None
    assert g2["median_root_median_spearman_bootstrap_95_ci"] is not None
    g6_n2 = report["primary_gates"]["G6"]["families"]["2"]
    assert g6_n2["g2_defined_roots"] == 10
    assert g6_n2["g2_median_root_median_spearman"] is not None
    assert g6_n2["g2_median_root_median_spearman_bootstrap_95_ci"] is not None


def test_g2_requires_five_fully_live_systems_per_family():
    stream, suffix, systems, profile = _tables()
    mask = (
        suffix["root_id"].isin(["root01", "root02", "root03"])
        & (suffix["teacher_n_qubits"] == 2)
        & (suffix["teacher_gamma"].isin([0.0, 0.2, 0.5]))
    )
    _make_suffix_groups_degenerate(suffix, mask, profile)

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    assert report["primary_gates"]["G2"]["eligible_system_counts_by_root_family"][
        "root01"
    ]["2"] == 3
    assert report["primary_gates"]["G2"]["successful_roots"] == 9
    assert not report["primary_gates"]["G2"]["passed"]
    assert report["decision"] == "NO_GO"


def test_g1_uses_production_prediction_disagreement():
    stream, suffix, systems, profile = _tables()
    suffix.loc[suffix.index[0], "prediction_disagreement"] = 1e-4
    suffix.loc[suffix.index[0], "predictions_agree"] = False

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    assert not report["primary_gates"]["G1"]["passed"]
    assert report["decision"] == "NO_GO"


def test_g1_recomputes_the_reported_mse_decomposition():
    stream, suffix, systems, profile = _tables()
    suffix.loc[suffix.index[0], "total_mse"] += 0.01

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    assert not report["primary_gates"]["G1"]["passed"]
    assert report["primary_gates"]["G1"][
        "max_recomputed_identity_error"
    ] == pytest.approx(0.01)
    assert report["decision"] == "NO_GO"


def test_g6_only_failure_uses_unresolved_family_replication_label():
    stream, suffix, systems, profile = _tables()
    mask = (
        suffix["root_id"].isin(["root01", "root02", "root03"])
        & (suffix["model"] == "ptd_parallel")
        & (suffix["teacher_n_qubits"] == 3)
        & (suffix["teacher_gamma"] == 4.0)
        & (suffix["window"] == 6)
        & (suffix["snapshot_budget"] == 4)
    )
    _set_suffix_representation(suffix, mask, 0.60, profile)

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    assert report["primary_gates"]["G5"]["passed"]
    assert not report["primary_gates"]["G6"]["passed"]
    assert report["decision"] == "UNRESOLVED_FAMILY_REPLICATION"


def test_feature_and_quantum_specific_gates_require_history_feasible_cells():
    stream, suffix, systems, profile = _tables()
    mask = (
        (suffix["model"] == "rr_same_identity")
        & (suffix["teacher_gamma"] == 4.0)
        & (suffix["window"] == 6)
        & (suffix["root_id"].isin(["root01", "root02", "root03"]))
    )
    _set_suffix_history(suffix, mask, 0.20, profile)

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    assert not report["primary_gates"]["G5"]["passed"]
    assert not report["secondary_qelm_vs_best_classical"]["passed"]
    assert report["primary_gates"]["G5"][
        "history_feasible_root_family_cells"
    ] == 18
    assert report["decision"] == "SUPPORTS_HISTORY_ONLY"


def test_window_gain_requires_a_registered_history_transition():
    stream, suffix, systems, profile = _tables()
    mask = (
        (suffix["model"] == "rr_same_identity")
        & (suffix["teacher_gamma"] == 2.0)
        & (suffix["window"] == 6)
        & (suffix["root_id"].isin(["root01", "root02", "root03"]))
    )
    _set_suffix_history(suffix, mask, 0.20, profile)

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    assert report["primary_gates"]["G4"]["defined_roots"] == 9
    assert not report["primary_gates"]["G4"]["passed"]
    assert report["decision"] == "SUPPORTS_HISTORY_ONLY"


def test_p6_family_failure_is_reported_but_does_not_change_primary_decision():
    stream, suffix, systems, profile = _tables()
    roots = ["root01", "root02", "root03"]
    suffix_mask = (
        suffix["root_id"].isin(roots)
        & (suffix["teacher_n_qubits"] == 3)
        & (suffix["teacher_gamma"] > 0.0)
        & (suffix["teacher_gamma"] < 2.0)
    )
    _set_suffix_history(
        suffix,
        suffix_mask,
        lambda row: float(
            0.02
            * np.exp(
                0.1 * float(row["teacher_gamma"]) * 0.5 * int(row["window"])
            )
        ),
        profile,
    )
    stream_mask = (
        stream["root_id"].isin(roots)
        & (stream["teacher_n_qubits"] == 3)
        & (stream["teacher_gamma"] > 0.0)
        & (stream["teacher_gamma"] < 2.0)
        & (stream["model"] == "rr_same_identity")
    )
    _set_stream_nrmse(
        stream,
        stream_mask,
        lambda row: float(
            0.05
            + 0.02
            * np.exp(
                0.1 * float(row["teacher_gamma"]) * 0.5 * int(row["window"])
            )
        ),
    )

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    assert report["primary_gates"]["G2"]["passed"]
    assert not report["secondary_p6_forgetting_scaling"]["passed"]
    assert report["secondary_p6_forgetting_scaling"]["families"]["3"][
        "successful_roots"
    ] == 9
    assert report["decision"] == "CONFIRMS_FINITE_RESOURCE_FRONTIER_QELM_SPECIFIC"


def test_p6_reports_effect_size_with_ten_of_twelve_defined_roots():
    stream, suffix, systems, profile = _tables()
    mask = (
        (suffix["root_id"] == "root01")
        & (suffix["teacher_n_qubits"] == 2)
        & (suffix["teacher_gamma"] == 0.2)
        & (suffix["window"] == 1)
        & (suffix["model"] == "rr_same_identity")
    )
    _set_suffix_history(suffix, mask, 0.0, profile)

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    family = report["secondary_p6_forgetting_scaling"]["families"]["2"]
    assert family["passed"]
    assert family["defined_roots"] == 11
    assert family["median_slope"] < 0.0
    assert len(family["median_slope_bootstrap_95_ci"]) == 2


def test_gate_thresholds_are_sourced_from_frozen_profile_decision():
    stream, suffix, systems, profile = _tables()
    profile["decision"]["min_best_classical_gain"] = 0.40

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    assert not report["secondary_qelm_vs_best_classical"]["passed"]
    assert report["decision"] == "CONFIRMS_FINITE_RESOURCE_FRONTIER"


def test_missing_g0_evidence_fails_closed():
    stream, suffix, systems, profile = _tables()

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(tests_passed=False),
    )

    assert not report["primary_gates"]["G0"]["passed"]
    assert report["decision"] == "NO_GO"


def test_g0_fails_closed_on_resource_width_observable_or_panel_mismatch():
    stream, suffix, systems, profile = _tables()

    bad_stream = stream.copy()
    bad_stream.loc[bad_stream.index[0], "feature_budget"] += 1
    report, _ = evaluate_canonical_gates(
        bad_stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )
    assert not report["primary_gates"]["G0"]["coverage"]["checks"][
        "stream_cells_complete"
    ]

    bad_systems = systems.copy()
    bad_systems.loc[bad_systems.index[0], "target_observables_sha256"] = "0" * 64
    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        bad_systems,
        profile,
        g0_evidence=_evidence(),
    )
    assert not report["primary_gates"]["G0"]["coverage"]["checks"][
        "observable_contracts"
    ]

    bad_suffix = suffix.copy()
    bad_suffix.loc[bad_suffix.index[0], "n_suffix_groups"] -= 1
    report, _ = evaluate_canonical_gates(
        stream,
        bad_suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )
    assert not report["primary_gates"]["G0"]["coverage"]["checks"][
        "suffix_cells_complete"
    ]


def test_g0_scientific_schema_fails_closed_on_metric_corruption():
    expected_checks = (
        "stream_core_metrics_finite_nonnegative",
        "stream_nrmse_matches_system_variance",
        "stream_learned_rank_domain",
        "suffix_regime_domain",
        "suffix_regime_matches_frozen_classification",
        "suffix_normalized_metric_identities",
        "suffix_history_fraction_identity",
        "suffix_student_independent_invariance",
        "suffix_witness_geometry_consistent",
        "suffix_boolean_domain",
    )
    for expected_check in expected_checks:
        stream, suffix, systems, profile = _tables()
        if expected_check == "stream_core_metrics_finite_nonnegative":
            stream.loc[stream.index[0], "mse"] = -1.0
        elif expected_check == "stream_nrmse_matches_system_variance":
            stream.loc[stream.index[0], "nrmse"] += 0.1
        elif expected_check == "stream_learned_rank_domain":
            row = stream.index[stream["model"] == "linear_delay"][0]
            stream.loc[row, "feature_rank"] = 1e9
        elif expected_check == "suffix_regime_domain":
            suffix.loc[suffix.index[0], "regime"] = "NOT_A_REGIME"
        elif expected_check == "suffix_regime_matches_frozen_classification":
            suffix.loc[suffix.index[0], "regime"] = "UNRESOLVED"
        elif expected_check == "suffix_normalized_metric_identities":
            suffix.loc[suffix.index[0], "history_nrmse"] += 0.1
        elif expected_check == "suffix_history_fraction_identity":
            suffix.loc[suffix.index[0], "history_fraction"] = 0.5
        elif expected_check == "suffix_student_independent_invariance":
            suffix.loc[suffix.index[0], "mean_group_diameter"] = 0.16
        elif expected_check == "suffix_witness_geometry_consistent":
            suffix.loc[suffix.index[0], "rms_bound_slack"] = 0.0
        elif expected_check == "suffix_boolean_domain":
            suffix["predictions_agree"] = suffix["predictions_agree"].astype(object)
            suffix.loc[suffix.index[0], "predictions_agree"] = "maybe"
        else:  # pragma: no cover - protects future table extensions
            raise AssertionError(expected_check)
        report, _ = evaluate_canonical_gates(
            stream,
            suffix,
            systems,
            profile,
            g0_evidence=_evidence(),
        )
        schema = report["primary_gates"]["G0"]["coverage"][
            "scientific_schema"
        ]
        assert not schema["passed"]
        assert not schema["checks"][expected_check]
        assert not report["primary_gates"]["G0"]["passed"]
        assert report["decision"] == "NO_GO"


def test_g0_schema_allows_registered_exactly_degenerate_targets():
    stream, suffix, systems, profile = _tables()
    system_id = str(systems.iloc[0]["system_id"])
    systems.loc[systems["system_id"] == system_id, "target_variance"] = 0.0
    systems.loc[systems["system_id"] == system_id, "target_live"] = False

    stream_mask = stream["system_id"].astype(str) == system_id
    stream.loc[stream_mask, "nrmse"] = np.inf
    learned_mask = stream_mask & stream["model"].astype(str).isin(
        {"linear_delay", "poly2_delay", "ptd_parallel", "classical_tanh"}
    )
    stream.loc[learned_mask, "validation_nrmse"] = np.inf

    suffix_mask = suffix["system_id"].astype(str) == system_id
    _make_suffix_groups_degenerate(suffix, suffix_mask, profile)

    report, _ = evaluate_canonical_gates(
        stream,
        suffix,
        systems,
        profile,
        g0_evidence=_evidence(),
    )

    schema = report["primary_gates"]["G0"]["coverage"]["scientific_schema"]
    assert schema["passed"]


def test_required_scientific_column_is_fail_closed():
    stream, suffix, systems, profile = _tables()
    with pytest.raises(ValueError, match="mean_row_rms"):
        evaluate_canonical_gates(
            stream.drop(columns=["mean_row_rms"]),
            suffix,
            systems,
            profile,
            g0_evidence=_evidence(),
        )


def test_strict_json_rejects_nan_and_duplicate_keys(tmp_path: Path):
    invalid_number = tmp_path / "nan.json"
    invalid_number.write_text('{"value": NaN}\n', encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="non-standard"):
        load_strict_json(invalid_number)
    with pytest.raises(ValueError, match="duplicate"):
        load_strict_json(duplicate)


def _manifest_for(results: Path) -> dict:
    artifact_hashes = {
        name: hashlib.sha256((results / name).read_bytes()).hexdigest()
        for name in (
            "stream_metrics.csv",
            "suffix_decomposition.csv",
            "system_diagnostics.csv",
            "canonical_summary.json",
            "computational_resources.json",
        )
    }
    material_payloads = {
        "configuration": ("config.yaml", "config\n"),
        "runner": ("runner.py", "runner\n"),
        "source:src/qrc_vs_qelm/canonical_study.py": (
            "canonical_study.py",
            "module\n",
        ),
    }
    files = {}
    for label, (relative, payload) in material_payloads.items():
        path = results / relative
        path.write_text(payload, encoding="utf-8")
        files[label] = {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    material = {
        "files": files,
        "file_count": len(files),
        "identity_sha256": _sha256_json(
            {label: entry["sha256"] for label, entry in files.items()}
        ),
    }
    reproducibility = {
        "schema_version": 1,
        "capture_phase": "before_simulation",
        "material": material,
        "environment": {},
        "git": {},
    }
    reproducibility["identity_sha256"] = _sha256_json(reproducibility)
    return {
        "schema_version": 1,
        "artifacts": artifact_hashes,
        "config_sha256": files["configuration"]["sha256"],
        "module_sha256": files[
            "source:src/qrc_vs_qelm/canonical_study.py"
        ]["sha256"],
        "runner_sha256": files["runner"]["sha256"],
        "reproducibility": reproducibility,
    }


def test_manifest_verifier_checks_hash_chain_and_artifact_bytes(tmp_path: Path):
    for name in (
        "stream_metrics.csv",
        "suffix_decomposition.csv",
        "system_diagnostics.csv",
        "canonical_summary.json",
        "computational_resources.json",
    ):
        payload = (
            '{"schema_version": 1, "decision_bearing": false}\n'
            if name == "computational_resources.json"
            else f"content for {name}\n"
        )
        (tmp_path / name).write_text(payload, encoding="utf-8")
    manifest = _manifest_for(tmp_path)

    assert verify_canonical_manifest(
        tmp_path,
        manifest,
        material_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )["passed"]
    alternate = tmp_path / "alternate.yaml"
    alternate.write_text("different config\n", encoding="utf-8")
    config_audit = verify_canonical_manifest(
        tmp_path,
        manifest,
        config_path=alternate,
    )
    assert not config_audit["passed"]
    assert not config_audit["checks"]["selected_config_matches"]
    (tmp_path / "stream_metrics.csv").write_text("changed\n", encoding="utf-8")
    audit = verify_canonical_manifest(tmp_path, manifest)
    assert not audit["passed"]
    assert not audit["checks"]["artifact_hashes"]


def test_pytest_junit_verifier_requires_cases_and_zero_failures(tmp_path: Path):
    material_hash = "a" * 64
    config_hash = "b" * 64
    properties = (
        '<properties>'
        f'<property name="canonical_material_identity_sha256" value="{material_hash}"/>'
        f'<property name="canonical_config_sha256" value="{config_hash}"/>'
        '</properties>'
    )
    passed = tmp_path / "passed.xml"
    passed.write_text(
        '<testsuite tests="2" failures="0" errors="0">'
        f'{properties}'
        '<testcase name="a"/><testcase name="b"/></testsuite>',
        encoding="utf-8",
    )
    failed = tmp_path / "failed.xml"
    failed.write_text(
        '<testsuite tests="1" failures="1" errors="0">'
        '<testcase name="a"><failure/></testcase></testsuite>',
        encoding="utf-8",
    )

    arguments = {
        "expected_material_identity_sha256": material_hash,
        "expected_config_sha256": config_hash,
    }
    assert verify_pytest_junit(
        passed,
        expected_testcases=2,
        **arguments,
    )["passed"]
    assert not verify_pytest_junit(
        passed,
        expected_testcases=3,
        **arguments,
    )["passed"]
    assert not verify_pytest_junit(
        failed,
        expected_testcases=1,
        **arguments,
    )["passed"]
    stale = verify_pytest_junit(
        passed,
        expected_testcases=2,
        expected_material_identity_sha256="c" * 64,
        expected_config_sha256=config_hash,
    )
    assert not stale["passed"]
    assert not stale["binding_matches"]
