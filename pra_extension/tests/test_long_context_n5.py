"""Regressions for the separately frozen N5 follow-up; no fresh roots are simulated."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_long_context_n5 import (  # noqa: E402
    paired_contrasts, summarize_cells, validate_tables,
)
from long_context_n5_common import (  # noqa: E402
    CONFIG, GAMMAS, PROFILE, ROOTS, TEST_FILES, WINDOWS, configuration,
    material_manifest, median_bootstrap, seed_audit, verify_freeze,
    validate_test_evidence,
    test_evidence as bind_test_evidence,
)
from run_long_context_n5 import run  # noqa: E402
from qrc_vs_qelm.canonical_emulation import classify_emulation_regime  # noqa: E402
from qrc_vs_qelm.canonical_resources import observable_names_sha256  # noqa: E402
from qrc_vs_qelm.core.readout import pauli_observables  # noqa: E402


def synthetic_tables():
    """Deterministic algebraic fixtures, never a follow-up target simulation."""
    stream, suffix, systems = [], [], []
    observable_hash = observable_names_sha256(f"{axis}{site}" for axis in "xyz" for site in range(10))
    target_hash = observable_names_sha256(item.name for item in pauli_observables(5, max_weight=2))
    specs = [("rr_same_identity", None, 45), ("constant", None, 0),
             ("linear_delay", None, None), ("poly2_delay", None, None),
             *[(model, snapshot, snapshot * 30) for model in ("ptd_parallel", "classical_tanh") for snapshot in (1, 2, 4)]]
    for index, root in enumerate(ROOTS, 1):
        for gamma in GAMMAS:
            system_id = f"{root}_n5_local_loss_g{gamma:.3f}".replace(".", "p")
            systems.append({"root_id": root, "teacher_n_qubits": 5, "teacher_gamma": gamma,
                            "system_id": system_id, "target_variance": .5, "target_dim": 45, "target_live": True,
                            "teacher_dissipator": "local_loss", "teacher_coupling_seed": 96_010_000 + index,
                            "target_observables_sha256": target_hash})
            for window in WINDOWS:
                history = (.30 + index * .001) / window / gamma
                replay = float(np.sqrt(history**2 + .001**2))
                for model, snapshot, budget in specs:
                    feature_budget = budget if budget is not None else window if model == "linear_delay" else window + window * (window + 1) // 2
                    representation = .001 if model == "rr_same_identity" else .08 / (snapshot or 1)
                    total = float(np.sqrt(history**2 + representation**2))
                    base = {"root_id": root, "teacher_n_qubits": 5, "teacher_gamma": gamma,
                            "window": window, "system_id": system_id, "teacher_dissipator": "local_loss",
                            "teacher_coupling_seed": 96_010_000 + index, "target_observables_sha256": target_hash,
                            "student_n_qubits": 10, "student_observables_sha256": observable_hash,
                            "model": model, "feature_budget": feature_budget, "snapshot_budget": snapshot}
                    stream.append({**base, "mse": total**2 * .5, "rmse": total * np.sqrt(.5),
                                   "nrmse": total, "validation_nrmse": total})
                    suffix.append({**base, "history_mse": history**2 * .5,
                                   "representation_mse": representation**2 * .5, "total_mse": total**2 * .5,
                                   "target_variance": .5, "history_nrmse": history,
                                   "representation_nrmse": representation, "total_nrmse": total,
                                   "identity_error": 0., "prediction_disagreement": 0.,
                                   "repeated_suffix_rmse": total * np.sqrt(.5), "worst_branch_rmse": total * np.sqrt(.5),
                                   "rms_pair_half_bound": 0., "max_group_diameter": 0., "half_max_group_diameter": 0.,
                                   "rms_bound_slack": total * np.sqrt(.5), "diameter_bound_slack": total * np.sqrt(.5),
                                   "predictions_agree": True, "rms_half_bound_satisfied": True,
                                   "diameter_half_bound_satisfied": True, "suffix_bounds_satisfied": True,
                                   "regime": classify_emulation_regime(history_nrmse=history, student_nrmse=total,
                                                                       oracle_nrmse=replay, target_variance=.5,
                                                                       tolerance=.1, min_target_variance=1e-8)})
    return pd.DataFrame(stream), pd.DataFrame(suffix), pd.DataFrame(systems)


def test_long_context_configuration_keeps_entire_window_and_fresh_roots():
    profile = configuration()
    assert profile["student"]["n_qubits"] == max(WINDOWS) == 10
    assert tuple(profile["teacher_n_qubits"]) == (5,)
    assert tuple(root["root_id"] for root in profile["roots"]) == ROOTS
    audit = seed_audit()
    assert audit["passed"] and audit["seed_count"] == 105


@pytest.mark.parametrize("change", ("six_qubits", "dropped_window", "fewer_roots", "changed_rows"))
def test_long_context_configuration_rejects_scope_drift(change, tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    if change == "six_qubits":
        raw["common"]["student"]["n_qubits"] = 6
    elif change == "dropped_window":
        raw["common"]["student"]["windows"] = [2, 4, 6, 8]
    elif change == "fewer_roots":
        raw["profiles"][PROFILE]["roots"].pop()
    else:
        raw["profiles"][PROFILE]["train_rows"] = 320
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        configuration(path)


def test_numerical_material_excludes_changing_figures_and_manuscripts():
    manifest = material_manifest()
    paths = set(manifest["files"])
    assert "scripts/long_context_qelm_backend.py" in paths
    assert "src/qrc_vs_qelm/canonical_study.py" in paths
    assert "LONG_CONTEXT_N5_PROTOCOL.md" in paths
    assert not any(path.startswith("paper/") or "figure" in path for path in paths)
    assert "configs/canonical_emulation_long_n5_freeze.json" not in paths
    assert len(manifest["identity_sha256"]) == 64


def test_fresh_execution_requires_freeze_and_never_overwrites(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_freeze(freeze=tmp_path / "missing.json")
    (tmp_path / "owned.txt").write_text("preserve")
    with pytest.raises(ValueError, match="nonempty"):
        run(PROFILE, tmp_path)
    assert (tmp_path / "owned.txt").read_text() == "preserve"


def test_test_evidence_requires_every_numerical_suite(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text("<testsuites><testsuite><testcase classname='unrelated' name='x'/></testsuite></testsuites>")
    with pytest.raises(ValueError, match="missing required suite"):
        bind_test_evidence(report)
    cases = "".join(f"<testcase classname='{Path(path).stem}' name='fixture'/>" for path in TEST_FILES)
    report.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    evidence = bind_test_evidence(report)
    assert evidence["case_count"] == len(TEST_FILES)
    assert "material" not in evidence  # Parsing alone cannot certify tested material.
    with pytest.raises(ValueError, match="bound to executed"):
        validate_test_evidence(evidence, material_manifest())
    report.write_text("<testsuites><testsuite><testcase classname='x'><failure/></testcase></testsuite></testsuites>")
    with pytest.raises(ValueError, match="successful"):
        bind_test_evidence(report)


def test_freeze_rejects_reusing_one_smoke_execution(tmp_path, monkeypatch):
    import long_context_n5_common as common

    monkeypatch.setattr(common, "configuration", lambda config: None)
    monkeypatch.setattr(common, "material_manifest", lambda config: {})
    monkeypatch.setattr(common, "seed_audit", lambda config: {})
    monkeypatch.setattr(common, "read_json", lambda path: {})
    monkeypatch.setattr(common, "validate_test_evidence", lambda evidence, material: None)
    with pytest.raises(ValueError, match="distinct execution directories"):
        common.create_freeze(tmp_path, tmp_path, tmp_path / "tests.json", freeze=tmp_path / "freeze.json")


def test_synthetic_full_grid_validates_and_all_controls_are_retained():
    stream, suffix, systems = synthetic_tables()
    audit = validate_tables(stream, suffix, systems)
    assert audit["passed"] and audit["root_cells"] == 120
    assert audit["qelm_root_budget_cells"] == 360
    assert audit["stream_rows"] == audit["suffix_rows"] == 1200
    cells = summarize_cells(suffix)
    assert len(cells) == 10 and cells["n_roots"].eq(12).all()
    assert cells["student_n_qubits"].eq(10).all()


@pytest.mark.parametrize("change", ("missing_row", "bad_regime", "bad_nrmse", "bad_identity", "bad_budget", "dead_target",
                                    "bad_prediction", "bad_rmse", "bad_target_hash", "bad_system_id", "bad_slack"))
def test_table_validation_fails_closed_on_corrupt_or_selected_results(change):
    stream, suffix, systems = synthetic_tables()
    if change == "missing_row":
        suffix = suffix.iloc[:-1]
    elif change == "bad_regime":
        suffix.loc[0, "regime"] = "EMULATED"
    elif change == "bad_nrmse":
        suffix.loc[0, "history_nrmse"] += .02
    elif change == "bad_identity":
        suffix.loc[0, "total_mse"] += .01
    elif change == "bad_budget":
        stream.loc[stream["feature_budget"].eq(120), "feature_budget"] = 72
    elif change == "dead_target":
        systems.loc[0, "target_live"] = False
    elif change == "bad_prediction":
        suffix.loc[0, "prediction_disagreement"] = 999
    elif change == "bad_rmse":
        stream.loc[0, "rmse"] = 777
    elif change == "bad_target_hash":
        suffix.loc[0, "target_observables_sha256"] = "0" * 64
    elif change == "bad_system_id":
        systems.loc[0, "system_id"] = "wrong"
    else:
        suffix.loc[0, "rms_bound_slack"] = -1
    with pytest.raises(ValueError):
        validate_tables(stream, suffix, systems)


def test_prespecified_contrasts_resample_twelve_roots_not_rows():
    stream, suffix, _ = synthetic_tables()
    history, classical, summaries = paired_contrasts(suffix, stream)
    assert len(history) == 36 and len(classical) == 120
    assert history.groupby("contrast")["root_id"].nunique().eq(12).all()
    assert classical.groupby(["teacher_gamma", "window"])["root_id"].nunique().eq(12).all()
    assert set(summaries["paired_history"]) == {
        "history_gamma1_d2_to_d10", "history_gamma4_d2_to_d10", "history_gamma1_d6_to_d10",
    }
    for summary in summaries["paired_history"].values():
        assert summary["n_roots"] == 12 and summary["bootstrap_unit"] == "root"
        assert summary["bootstrap_draws"] == 10_000
    assert median_bootstrap(np.arange(12), 123) == median_bootstrap(np.arange(12), 123)
    with pytest.raises(ValueError, match="twelve"):
        median_bootstrap(np.arange(120), 123)


def test_robust_history_summary_is_not_a_median_threshold():
    _, suffix, _ = synthetic_tables()
    cell = suffix["teacher_gamma"].eq(1) & suffix["window"].eq(4)
    insufficient_roots = set(ROOTS[-4:])
    suffix.loc[cell & suffix["root_id"].isin(insufficient_roots), "history_nrmse"] = .2
    suffix.loc[cell & suffix["root_id"].isin(insufficient_roots), "regime"] = "HISTORY_IMPOSSIBLE"
    selected = summarize_cells(suffix).query("teacher_gamma == 1 and window == 4").iloc[0]
    assert selected["median_history_nrmse"] < .10
    assert selected["history_sufficient_n"] == 8
    assert selected["display_outcome"] == "HISTORY_NOT_ROBUST"
