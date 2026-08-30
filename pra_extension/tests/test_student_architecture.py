from __future__ import annotations

import pandas as pd
import pytest

from analyze_student_architecture import analyze, validate_architecture_results
from student_architecture import (
    ARCHITECTURES,
    default_architecture_config,
    edge_coefficient_l2,
    matched_architecture_couplings,
    run_architecture_study,
    validate_architecture_config,
)


def test_smoke_architecture_study_is_complete_and_norm_matched():
    config = default_architecture_config("smoke")
    validate_architecture_config(config)
    results = run_architecture_study(config)
    validate_architecture_results(results, "smoke")
    assert len(results) == 15
    qelm = results[results.student_model.eq("qelm")]
    for _, rows in qelm.groupby("snapshot_budget"):
        assert len(rows) == len(ARCHITECTURES)
        assert rows.coupling_edge_l2.max() - rows.coupling_edge_l2.min() < 1e-12
    summary, selected = analyze(results, "smoke")
    assert summary["audit"]["test_targets_unused_for_model_selection"] is True
    assert len(selected) == 1


def test_dense_is_exact_canonical_draw_and_sparse_graphs_are_matched():
    config = default_architecture_config("smoke")
    root = config.roots[0]
    dense, factor, dense_norm = matched_architecture_couplings(
        root=root,
        architecture="dense",
        n_qubits=config.student_n_qubits,
        coupling_scale=config.coupling_scale,
    )
    assert factor == 1.0
    assert edge_coefficient_l2(dense) == pytest.approx(dense_norm)
    for architecture in ARCHITECTURES[1:]:
        sparse, scaling, norm = matched_architecture_couplings(
            root=root,
            architecture=architecture,
            n_qubits=config.student_n_qubits,
            coupling_scale=config.coupling_scale,
            dense_reference_norm=dense_norm,
        )
        assert scaling > 0.0
        assert edge_coefficient_l2(sparse) == pytest.approx(dense_norm)
        assert norm == pytest.approx(dense_norm)


def test_strict_validator_rejects_profile_and_norm_mutations():
    results = run_architecture_study(default_architecture_config("smoke"))
    missing = results.iloc[:-1].copy()
    with pytest.raises(ValueError, match="Cartesian profile mismatch"):
        validate_architecture_results(missing, "smoke")
    changed = results.copy()
    index = changed.index[changed.student_architecture.eq("chain")][0]
    changed.loc[index, "coupling_edge_l2"] *= 1.01
    with pytest.raises(ValueError, match="coupling norm mismatch"):
        validate_architecture_results(changed, "smoke")


def test_full_artifact_if_present_is_strictly_valid():
    path = (
        "pra_extension/outputs/student_architecture_full_exploratory_v1_12roots/"
        "student_architecture_results.csv"
    )
    try:
        results = pd.read_csv(path)
    except FileNotFoundError:
        pytest.skip("full exploratory artifact is not present")
    validate_architecture_results(results, "full")
