from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from analyze_finite_k_history_calibration import (  # noqa: E402
    DEFAULT_GATE_REPORT,
    DEFAULT_PHASE_SOURCE,
    DEFAULT_SUFFIX,
    analyze,
    load_strict_json,
    sha256_file,
    validate_suffix_frame,
    write_summary,
)


def _hashes() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in (DEFAULT_SUFFIX, DEFAULT_GATE_REPORT, DEFAULT_PHASE_SOURCE)
    }


def test_frozen_companion_reproduces_labels_gates_and_bias_factor(tmp_path: Path) -> None:
    before = _hashes()
    summary = analyze()
    first = write_summary(summary, tmp_path / "first")
    second = write_summary(analyze(), tmp_path / "second")

    assert first.read_bytes() == second.read_bytes()
    assert len(bytes.fromhex(sha256_file(first))) == 32
    assert load_strict_json(first) == summary
    assert _hashes() == before
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_summary(summary, first.parent)

    calibration = summary["calibration"]
    assert calibration["branches_per_suffix_group"] == 4
    assert calibration["history_mse_numerator_factor"] == pytest.approx(4.0 / 3.0)
    assert calibration["finite_K_downward_bias_fraction"] == pytest.approx(0.25)
    assert calibration[
        "history_nrmse_factor_holding_registered_target_variance"
    ] == pytest.approx(1.1547005383792515)
    assert calibration["equivalent_registered_scale_tolerance_nrmse"] == pytest.approx(
        0.08660254037844388
    )

    root_labels = summary["root_level_phase_labels"]
    assert root_labels["rows"] == 1_728
    assert root_labels["changed_rows"] == 39
    assert root_labels["transition_counts"] == {
        "UNRESOLVED->HISTORY_IMPOSSIBLE": 39
    }
    assert root_labels["registered_counts"] == {
        "DEGENERATE_TARGET": 0,
        "HISTORY_IMPOSSIBLE": 1_335,
        "EMULATED": 188,
        "REPRESENTATION_LIMITED": 72,
        "UNRESOLVED": 133,
    }
    assert root_labels["calibrated_counts"] == {
        "DEGENERATE_TARGET": 0,
        "HISTORY_IMPOSSIBLE": 1_374,
        "EMULATED": 188,
        "REPRESENTATION_LIMITED": 72,
        "UNRESOLVED": 94,
    }
    assert summary["all_model_row_label_sensitivity"]["changed_rows"] == 130

    phase = summary["modal_phase_map"]
    assert phase["composition_or_agreement_changed_cells"] == 12
    assert phase["modal_label_changed_cells"] == 4
    assert phase["registered_counts"] == {
        "DEGENERATE_TARGET": 0,
        "HISTORY_IMPOSSIBLE": 108,
        "EMULATED": 17,
        "REPRESENTATION_LIMITED": 7,
        "UNRESOLVED": 10,
        "MIXED": 2,
    }
    assert phase["calibrated_counts"] == {
        "DEGENERATE_TARGET": 0,
        "HISTORY_IMPOSSIBLE": 111,
        "EMULATED": 18,
        "REPRESENTATION_LIMITED": 7,
        "UNRESOLVED": 7,
        "MIXED": 1,
    }
    observed_modal_changes = {
        (
            item["teacher_n_qubits"],
            item["teacher_gamma"],
            item["window"],
            item["feature_budget"],
            item["registered_modal_regime"],
            item["calibrated_modal_regime"],
        )
        for item in phase["changed_cells"]
        if item["modal_label_changed"]
    }
    assert observed_modal_changes == {
        (2, 4.0, 2, 18, "MIXED", "EMULATED"),
        (3, 1.0, 6, 18, "UNRESOLVED", "HISTORY_IMPOSSIBLE"),
        (3, 1.0, 6, 36, "UNRESOLVED", "HISTORY_IMPOSSIBLE"),
        (3, 1.0, 6, 72, "UNRESOLVED", "HISTORY_IMPOSSIBLE"),
    }

    gates = summary["gate_sensitivity"]
    assert gates["decision_invariant"] is True
    assert all(gates["primary_gate_status_after_history_calibration"].values())
    assert gates["G3"]["families"]["2"]["successful_roots"] == 12
    assert gates["G3"]["families"]["3"]["successful_roots"] == 12
    assert gates["G4"]["calibrated_transition_cells"] == 48
    assert gates["G4"]["calibrated_transition_roots"] == 12
    assert gates["G4"][
        "maximum_calibrated_long_window_history_nrmse"
    ] == pytest.approx(0.09656989428879294)
    assert gates["G5"]["calibrated_history_feasible_root_family_cells"] == 24
    assert gates["G5"]["maximum_calibrated_history_nrmse"] == pytest.approx(
        0.010367797948263173
    )
    assert gates["G5"]["registered_empirical_representation_gain"][
        "median_root_gain"
    ] == pytest.approx(0.6726414464648288)
    assert gates["G6"]["registered_empirical_representation_gain_by_family"]["2"][
        "median_root_gain"
    ] == pytest.approx(0.661282098534065)
    assert gates["G6"]["registered_empirical_representation_gain_by_family"]["3"][
        "median_root_gain"
    ] == pytest.approx(0.6884823496693037)

    boundary = summary["decomposition_boundary"]
    assert boundary["registered_exact_identity_retained"] is True
    assert boundary["numerator_only_calibration_is_an_exact_empirical_decomposition"] is False
    assert boundary["negative_counterfactual_residual_rows_all_models"] == 832
    assert boundary["negative_counterfactual_residual_rows_in_fixed_G5_contrast"] == 0
    assert boundary["fixed_G5_contrast_rows"] == 48


@pytest.mark.parametrize(
    ("source", "keyword"),
    [
        (DEFAULT_SUFFIX, "suffix_decomposition.csv"),
        (DEFAULT_GATE_REPORT, "canonical_gate_report.json"),
        (DEFAULT_PHASE_SOURCE, "fig04_regime_map_source.csv"),
    ],
)
def test_frozen_companion_rejects_any_source_byte_mutation(
    tmp_path: Path,
    source: Path,
    keyword: str,
) -> None:
    suffix = tmp_path / DEFAULT_SUFFIX.name
    gate = tmp_path / DEFAULT_GATE_REPORT.name
    phase = tmp_path / DEFAULT_PHASE_SOURCE.name
    for original, target in (
        (DEFAULT_SUFFIX, suffix),
        (DEFAULT_GATE_REPORT, gate),
        (DEFAULT_PHASE_SOURCE, phase),
    ):
        shutil.copyfile(original, target)
    target = {suffix.name: suffix, gate.name: gate, phase.name: phase}[source.name]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=f"{keyword} frozen SHA-256 mismatch"):
        analyze(suffix, gate, phase)


def test_strict_json_and_semantic_validation_fail_closed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON numeric constant"):
        load_strict_json(nonfinite)

    suffix = pd.read_csv(DEFAULT_SUFFIX)
    suffix.loc[0, "history_mse"] *= 1.01
    with pytest.raises(ValueError, match="history quantities change"):
        validate_suffix_frame(
            suffix,
            tolerance=0.1,
            min_target_variance=1e-8,
        )
