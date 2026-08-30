import importlib.util
import json
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_canonical_emulation_figures.py"
CONFIG = ROOT / "configs" / "canonical_emulation.yaml"

spec = importlib.util.spec_from_file_location("render_canonical_emulation_figures", SCRIPT)
render = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = render
spec.loader.exec_module(render)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _sha_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _write_synthetic_study(
    path: Path,
    *,
    gate_report: bool = False,
    degenerate_target: bool = False,
) -> None:
    roots = ("root01", "root02")
    families = (2, 3)
    gammas = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)
    windows = (1, 2, 4, 6)
    systems = []
    stream_rows = []
    suffix_rows = []
    for root_index, root_id in enumerate(roots):
        for n_qubits in families:
            for gamma in gammas:
                system_id = f"{root_id}_n{n_qubits}_g{gamma:g}"
                degenerate_system = bool(
                    degenerate_target and n_qubits == 2 and gamma == 0.0
                )
                systems.append(
                    {
                        "root_id": root_id,
                        "system_id": system_id,
                        "teacher_n_qubits": n_qubits,
                        "teacher_gamma": gamma,
                        "target_live": not degenerate_system,
                    }
                )
                for window in windows:
                    if degenerate_system:
                        history_nrmse = 0.0
                    elif gamma == 0.0:
                        history_nrmse = 0.60 + 0.002 * window + 0.003 * root_index
                    else:
                        history_nrmse = (0.42 + 0.01 * root_index) * np.exp(
                            -0.62 * gamma * 0.5 * window
                        )
                    history_mse = history_nrmse**2
                    model_budgets = {
                        "ptd_parallel": ((1, 18), (2, 36), (4, 72)),
                        "classical_tanh": ((1, 18), (2, 36), (4, 72)),
                        "poly2_delay": ((None, 27),),
                        "linear_delay": ((None, 6),),
                        "rr_same_identity": ((None, 9),),
                    }
                    for model, budgets in model_budgets.items():
                        for snapshot_budget, feature_budget in budgets:
                            if model == "ptd_parallel":
                                representation_nrmse = 0.22 * np.sqrt(18 / feature_budget)
                            elif model == "classical_tanh":
                                representation_nrmse = 0.27 * np.sqrt(18 / feature_budget)
                            elif model == "poly2_delay":
                                representation_nrmse = 0.035
                            elif model == "rr_same_identity":
                                representation_nrmse = 0.01
                            else:
                                representation_nrmse = 0.14
                            representation_nrmse *= 1.0 + 0.02 * root_index
                            if degenerate_system:
                                representation_nrmse = (
                                    np.inf
                                    if model == "ptd_parallel" and snapshot_budget != 1
                                    else 0.0
                                )
                            representation_mse = representation_nrmse**2
                            total_mse = history_mse + representation_mse
                            total_nrmse = float(np.sqrt(total_mse))
                            if degenerate_system:
                                regime = "DEGENERATE_TARGET"
                            elif history_nrmse > 0.10:
                                regime = "HISTORY_IMPOSSIBLE"
                            elif model == "ptd_parallel" and total_nrmse <= 0.10:
                                regime = "EMULATED"
                            elif model == "ptd_parallel":
                                regime = "REPRESENTATION_LIMITED"
                            else:
                                regime = "UNRESOLVED"
                            if degenerate_system:
                                stream_nrmse = 0.0
                            elif gamma == 4.0 and window == 6:
                                stream_nrmse = {
                                    "ptd_parallel": 0.18,
                                    "classical_tanh": 0.21,
                                    "poly2_delay": 0.08,
                                    "linear_delay": 0.15,
                                    "rr_same_identity": 0.02,
                                }[model] * (1.0 + 0.03 * root_index)
                            else:
                                stream_nrmse = 0.55 + 0.01 * root_index
                            stream_rows.append(
                                {
                                    "root_id": root_id,
                                    "system_id": system_id,
                                    "teacher_n_qubits": n_qubits,
                                    "teacher_gamma": gamma,
                                    "window": window,
                                    "model": model,
                                    "snapshot_budget": snapshot_budget,
                                    "feature_budget": feature_budget,
                                    "nrmse": stream_nrmse,
                                }
                            )
                            suffix_rows.append(
                                {
                                    "root_id": root_id,
                                    "system_id": system_id,
                                    "teacher_n_qubits": n_qubits,
                                    "teacher_gamma": gamma,
                                    "window": window,
                                    "model": model,
                                    "snapshot_budget": snapshot_budget,
                                    "feature_budget": feature_budget,
                                    "history_mse": history_mse,
                                    "representation_mse": representation_mse,
                                    "total_mse": total_mse,
                                    "target_variance": 0.0 if degenerate_system else 1.0,
                                    "identity_error": 0.0,
                                    "history_nrmse": history_nrmse,
                                    "representation_nrmse": representation_nrmse,
                                    "total_nrmse": total_nrmse,
                                    "oracle_nrmse": history_nrmse,
                                    "regime": regime,
                                }
                            )

    path.mkdir(parents=True)
    pd.DataFrame(stream_rows).to_csv(path / "stream_metrics.csv", index=False)
    pd.DataFrame(suffix_rows).to_csv(path / "suffix_decomposition.csv", index=False)
    pd.DataFrame(systems).to_csv(path / "system_diagnostics.csv", index=False)
    summary = {
        "schema_version": 1,
        "study": "canonical_qrc_to_parallel_qelm_emulation",
        "profile": "calibration",
        "confirmatory": False,
        "status": "CALIBRATION_ONLY_CONFIRMATORY_SEALED",
        "decision": "CALIBRATION_ONLY",
        "counts": {"roots": len(roots), "systems": len(systems)},
    }
    (path / "canonical_summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / "computational_resources.json").write_text(
        json.dumps(
            {"schema_version": 1, "decision_bearing": False},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact_names = (
        "stream_metrics.csv",
        "suffix_decomposition.csv",
        "system_diagnostics.csv",
        "canonical_summary.json",
        "computational_resources.json",
    )
    material_paths = {
        "configuration": CONFIG,
        "runner": ROOT / "scripts" / "run_canonical_emulation.py",
        "source:src/qrc_vs_qelm/canonical_study.py": (
            ROOT / "src" / "qrc_vs_qelm" / "canonical_study.py"
        ),
    }
    material_files = {
        label: {
            "path": str(source.relative_to(ROOT)),
            "sha256": _sha(source),
            "size_bytes": source.stat().st_size,
        }
        for label, source in material_paths.items()
    }
    material = {
        "files": material_files,
        "file_count": len(material_files),
        "identity_sha256": _sha_json(
            {label: entry["sha256"] for label, entry in material_files.items()}
        ),
    }
    reproducibility = {
        "schema_version": 1,
        "capture_phase": "before_simulation",
        "material": material,
        "environment": {},
        "git": {},
    }
    reproducibility["identity_sha256"] = _sha_json(reproducibility)
    manifest = {
        "schema_version": 1,
        "study": summary["study"],
        "profile": summary["profile"],
        "config_sha256": material_files["configuration"]["sha256"],
        "module_sha256": material_files[
            "source:src/qrc_vs_qelm/canonical_study.py"
        ]["sha256"],
        "runner_sha256": material_files["runner"]["sha256"],
        "reproducibility": reproducibility,
        "artifacts": {name: _sha(path / name) for name in artifact_names},
    }
    (path / "canonical_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if gate_report:
        canonical_names = (*artifact_names, "canonical_manifest.json")
        report = {
            "schema_version": 1,
            "study": summary["study"],
            "profile": summary["profile"],
            "confirmatory": False,
            "decision": "CALIBRATION_ONLY",
            "secondary_qelm_vs_best_classical": {
                "fixed_teacher_gamma": 4.0,
                "fixed_window": 6,
                "snapshot_budget": 4,
            },
            "evaluation_provenance": {
                "config_sha256": _sha(CONFIG),
                "input_artifact_sha256": {
                    name: _sha(path / name) for name in canonical_names
                },
            },
        }
        (path / "canonical_gate_report.json").write_text(
            json.dumps(report, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def test_canonical_figure_sources_preserve_contracts(tmp_path):
    results = tmp_path / "results"
    _write_synthetic_study(results)
    tables = render.load_study_tables(results, CONFIG)

    history = render.history_frontier_source(tables)
    surface = render.qelm_error_surface_source(tables)
    decomposition = render.decomposition_source(tables)
    regimes = render.regime_map_source(tables)
    controls = render.matched_control_source(tables)
    predictions = render.window_prediction_source(tables)
    scaling = render.dissipation_window_scaling_source(tables)

    assert len(history) == 2 * 6 * 4
    assert len(surface) == 2 * 6 * 4 * 3
    assert len(decomposition) == 2 * 6 * 4
    assert decomposition["aggregation_identity_error"].max() < 1e-12
    assert len(regimes) == 2 * 6 * 4 * 3
    assert set(controls["model"]) == {
        "ptd_parallel",
        "classical_tanh",
        "poly2_delay",
        "linear_delay",
    }
    assert len(controls) == 2 * 2 * 4
    assert set(
        controls.loc[
            controls["model"].isin(("ptd_parallel", "classical_tanh")),
            "feature_budget",
        ]
    ) == {72}
    assert set(
        controls.loc[
            controls["model"].isin(("ptd_parallel", "classical_tanh")),
            "snapshot_budget",
        ]
    ) == {4.0}
    assert set(controls["estimand"]) == {"held_out_stream_nrmse"}
    assert controls["secondary_gate_eligible"].all()
    assert set(
        controls.loc[controls["best_registered_classical"], "model"]
    ) == {"poly2_delay"}
    assert np.allclose(
        controls.loc[controls["model"] == "poly2_delay", "stream_nrmse"],
        [0.08, 0.08, 0.0824, 0.0824],
    )
    nondecaying = predictions[np.isclose(predictions["teacher_gamma"], 0.0)]
    assert (~nondecaying["predicted_finite"]).all()
    assert nondecaying["observed_right_censored"].all()
    assert set(scaling["teacher_gamma"]) == {0.2, 0.5, 1.0, 2.0, 4.0}
    assert set(scaling["root_fit_n_points"]) == {20}
    per_root_fit = scaling.drop_duplicates(["root_id", "teacher_n_qubits"])
    assert len(per_root_fit) == 2 * 2
    assert (per_root_fit["root_fit_slope"] < 0.0).all()
    assert (per_root_fit["root_fit_r_squared"] > 0.99).all()
    assert set(per_root_fit["root_fit_min_r_squared"]) == {
        tables.min_scaling_r_squared
    }
    assert scaling["prospective_secondary_prediction"].all()


def test_rendered_canonical_figures_are_deterministic_and_read_only(tmp_path):
    results = tmp_path / "results"
    output = tmp_path / "figures"
    _write_synthetic_study(results, gate_report=True)
    input_paths = sorted(results.iterdir())
    before = {path.name: _sha(path) for path in input_paths}

    tables = render.load_study_tables(results, CONFIG)
    first = render.render_all(tables, output)
    second = render.render_all(tables, output)

    assert first["outputs"] == second["outputs"]
    assert first["status_label"] == "CALIBRATION ONLY • CONFIRMATORY SEALED"
    assert {path.name: _sha(path) for path in input_paths} == before
    for index in range(1, 8):
        stem = next(output.glob(f"fig{index:02d}_*.pdf")).stem
        assert (output / f"{stem}.pdf").stat().st_size > 10_000
        assert (output / f"{stem}.png").stat().st_size > 10_000
        assert (output / f"{stem}_source.csv").stat().st_size > 100
    manifest = json.loads((output / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["confirmatory"] is False
    assert len([name for name in manifest["outputs"] if name.endswith(".pdf")]) == 7
    assert manifest["primary_figures"] == 6
    assert manifest["supplemental_figures"] == 1
    assert "figure_manifest.json" not in manifest["outputs"]
    assert manifest["renderer_sha256"] == _sha(SCRIPT)
    assert manifest["config_sha256"] == _sha(CONFIG)
    for name, digest in manifest["input_artifacts"].items():
        assert digest == _sha(results / name)
    for name, digest in manifest["outputs"].items():
        assert digest == _sha(output / name)
    for source_path in output.glob("fig*_source.csv"):
        source_with_provenance = pd.read_csv(source_path)
        assert set(source_with_provenance["provenance_renderer_sha256"]) == {
            _sha(SCRIPT)
        }
        assert set(source_with_provenance["provenance_config_sha256"]) == {
            _sha(CONFIG)
        }
        for name, digest in manifest["input_artifacts"].items():
            safe = name.replace(".", "_").replace("-", "_")
            assert set(
                source_with_provenance[f"provenance_input_{safe}_sha256"]
            ) == {digest}
    source = pd.read_csv(output / "fig05_matched_controls_source.csv")
    assert set(source["provenance_renderer_sha256"]) == {_sha(SCRIPT)}
    assert set(source["provenance_config_sha256"]) == {_sha(CONFIG)}
    assert set(source["provenance_decision"]) == {"CALIBRATION_ONLY"}
    assert set(source["estimand"]) == {"held_out_stream_nrmse"}


def test_manifest_config_and_gate_report_mismatches_fail_closed(tmp_path):
    results = tmp_path / "results"
    _write_synthetic_study(results, gate_report=True)

    alternate = tmp_path / "alternate.yaml"
    alternate.write_text(CONFIG.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest verification failed"):
        render.load_study_tables(results, alternate)

    report_path = results / "canonical_gate_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["evaluation_provenance"]["input_artifact_sha256"][
        "stream_metrics.csv"
    ] = "0" * 64
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gate report input hash mismatch"):
        render.load_study_tables(results, CONFIG)

    report_path.unlink()
    with (results / "suffix_decomposition.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="manifest verification failed"):
        render.load_study_tables(results, CONFIG)


def test_render_fails_before_writing_if_loaded_input_drifts(tmp_path):
    results = tmp_path / "results"
    output = tmp_path / "figures"
    _write_synthetic_study(results)
    tables = render.load_study_tables(results, CONFIG)

    with (results / "stream_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="input changed after load"):
        render.render_all(tables, output)
    assert not output.exists()


def test_confirmatory_status_uses_gate_report_not_pending_summary():
    summary = {"confirmatory": True, "decision": "PENDING_GATES"}
    report = {"decision": "SUPPORTS_HISTORY_ONLY"}
    decision, source, status = render._decision_and_status(summary, report)

    assert decision == "SUPPORTS_HISTORY_ONLY"
    assert source == "canonical_gate_report.json"
    assert status == "CONFIRMATORY OUTPUT • SUPPORTS HISTORY ONLY"


def test_figures_handle_infeasible_degenerate_and_zero_history(tmp_path):
    results = tmp_path / "results"
    _write_synthetic_study(results)
    tables = render.load_study_tables(results, CONFIG)
    suffix = tables.suffix.copy()

    zero_history = (
        (suffix["model"] == "rr_same_identity")
        & (suffix["root_id"] == "root01")
        & (suffix["teacher_n_qubits"] == 2)
        & np.isclose(suffix["teacher_gamma"], 0.2)
        & (suffix["window"] == 2)
    )
    suffix.loc[zero_history, ["history_nrmse", "history_mse"]] = 0.0

    nonfinite_history = (
        (suffix["model"] == "rr_same_identity")
        & (suffix["root_id"] == "root01")
        & (suffix["teacher_n_qubits"] == 3)
        & np.isclose(suffix["teacher_gamma"], 0.5)
        & (suffix["window"] == 4)
    )
    suffix.loc[nonfinite_history, "history_nrmse"] = np.nan

    incomplete_history = (
        (suffix["model"] == "rr_same_identity")
        & (suffix["root_id"] == "root02")
        & (suffix["teacher_n_qubits"] == 2)
        & np.isclose(suffix["teacher_gamma"], 0.5)
        & (suffix["window"] == 2)
    )
    suffix = suffix.loc[~incomplete_history].copy()

    degenerate = (
        (suffix["model"] == "rr_same_identity")
        & (suffix["root_id"] == "root02")
        & (suffix["teacher_n_qubits"] == 3)
        & (suffix["teacher_gamma"] > 0.0)
    )
    suffix.loc[degenerate, "target_variance"] = 0.0

    infeasible_fixed = (
        (suffix["model"] == "rr_same_identity")
        & (suffix["root_id"] == "root01")
        & (suffix["teacher_n_qubits"] == 3)
        & np.isclose(suffix["teacher_gamma"], tables.fixed_fast_gamma)
        & (suffix["window"] == tables.fixed_fast_window)
    )
    suffix.loc[infeasible_fixed, "history_nrmse"] = 2.0 * tables.tolerance_nrmse
    modified = replace(tables, suffix=suffix)

    predictions = render.window_prediction_source(modified)
    zero_row = predictions[
        (predictions["root_id"] == "root01")
        & (predictions["teacher_n_qubits"] == 2)
        & np.isclose(predictions["teacher_gamma"], 0.2)
    ].iloc[0]
    assert zero_row["fit_status"] == "NONPOSITIVE_HISTORY_TRACE"
    assert not bool(zero_row["predicted_finite"])
    assert int(zero_row["trace_positive_finite_n"]) == 3
    assert int(zero_row["fit_n_points"]) == 0
    assert bool(zero_row["registered_window_trace_complete"])

    nonfinite_row = predictions[
        (predictions["root_id"] == "root01")
        & (predictions["teacher_n_qubits"] == 3)
        & np.isclose(predictions["teacher_gamma"], 0.5)
    ].iloc[0]
    assert nonfinite_row["fit_status"] == "NONFINITE_HISTORY_TRACE"
    assert int(nonfinite_row["trace_positive_finite_n"]) == 3
    assert int(nonfinite_row["fit_n_points"]) == 0

    incomplete_row = predictions[
        (predictions["root_id"] == "root02")
        & (predictions["teacher_n_qubits"] == 2)
        & np.isclose(predictions["teacher_gamma"], 0.5)
    ].iloc[0]
    assert incomplete_row["fit_status"] == "INCOMPLETE_REGISTERED_WINDOW_TRACE"
    assert int(incomplete_row["trace_n_points"]) == 3
    assert int(incomplete_row["fit_n_points"]) == 0

    valid_row = predictions[
        (predictions["root_id"] == "root02")
        & (predictions["teacher_n_qubits"] == 2)
        & np.isclose(predictions["teacher_gamma"], 1.0)
    ].iloc[0]
    assert valid_row["fit_status"] == "FIT"
    assert int(valid_row["trace_n_points"]) == 4
    assert int(valid_row["fit_n_points"]) == 4
    assert bool(valid_row["predicted_finite"])
    assert int(valid_row["predicted_window"]) >= 1
    fitted_rows = predictions[
        predictions["fit_status"].isin(("FIT", "NON_DECAYING_FIT"))
    ]
    assert set(fitted_rows["fit_n_points"]) == {4}
    assert fitted_rows["registered_window_trace_complete"].all()
    assert "predicted_left_censored" not in predictions
    assert "predicted_plot_x" not in predictions

    degenerate_rows = predictions[
        (predictions["root_id"] == "root02")
        & (predictions["teacher_n_qubits"] == 3)
        & (predictions["teacher_gamma"] > 0.0)
    ]
    assert set(degenerate_rows["fit_status"]) == {"DEGENERATE_TARGET"}
    assert not degenerate_rows["observed_right_censored"].any()

    window_output = tmp_path / "window.pdf"
    render.render_window_prediction(predictions, modified, window_output)
    assert window_output.stat().st_size > 10_000

    controls = render.matched_control_source(modified)
    excluded_controls = controls[
        (controls["root_id"] == "root01")
        & (controls["teacher_n_qubits"] == 3)
    ]
    assert not excluded_controls["secondary_gate_eligible"].any()
    assert excluded_controls["qelm_gain_over_best_classical"].isna().all()
    assert not excluded_controls["best_registered_classical"].any()

    scaling = render.dissipation_window_scaling_source(modified)
    zero_fit = scaling[
        (scaling["root_id"] == "root01")
        & (scaling["teacher_n_qubits"] == 2)
    ].iloc[0]
    assert zero_fit["root_fit_status"] == "NONPOSITIVE_OR_NONFINITE_HISTORY"
    assert not bool(zero_fit["root_fit_defined"])
    degenerate_fit = scaling[
        (scaling["root_id"] == "root02")
        & (scaling["teacher_n_qubits"] == 3)
    ].iloc[0]
    assert degenerate_fit["root_fit_status"] == "INCOMPLETE_LIVE_GRID"
    assert not bool(degenerate_fit["root_fit_defined"])
    assert scaling["root_fit_min_r_squared"].eq(
        tables.min_scaling_r_squared
    ).all()


def test_material_drift_and_unfrozen_gamma_style_fail_closed(tmp_path, monkeypatch):
    results = tmp_path / "results"
    _write_synthetic_study(results)

    manifest_path = results / "canonical_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runner = manifest["reproducibility"]["material"]["files"]["runner"]
    runner["sha256"] = "f" * 64
    manifest["runner_sha256"] = runner["sha256"]
    material = manifest["reproducibility"]["material"]
    material["identity_sha256"] = _sha_json(
        {label: entry["sha256"] for label, entry in material["files"].items()}
    )
    reproducibility = manifest["reproducibility"]
    without_identity = dict(reproducibility)
    without_identity.pop("identity_sha256", None)
    reproducibility["identity_sha256"] = _sha_json(without_identity)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest verification failed"):
        render.load_study_tables(results, CONFIG)

    style_results = tmp_path / "style_results"
    _write_synthetic_study(style_results)
    monkeypatch.delitem(render.GAMMA_COLORS, 4.0)
    with pytest.raises(ValueError, match="frozen gamma style maps"):
        render.load_study_tables(style_results, CONFIG)


def test_degenerate_suffix_targets_are_counted_but_excluded_from_error_plots(
    tmp_path,
):
    results = tmp_path / "degenerate_results"
    output = tmp_path / "degenerate_figures"
    _write_synthetic_study(results, degenerate_target=True)

    tables = render.load_study_tables(results, CONFIG)
    dead_ptd = tables.suffix[
        (tables.suffix["model"] == "ptd_parallel")
        & (tables.suffix["teacher_n_qubits"] == 2)
        & np.isclose(tables.suffix["teacher_gamma"], 0.0)
    ]
    assert dead_ptd["target_variance"].eq(0.0).all()
    assert (dead_ptd["total_nrmse"] == 0.0).any()
    assert np.isinf(dead_ptd["total_nrmse"]).any()

    history = render.history_frontier_source(tables)
    dead_history = history[
        (history["teacher_n_qubits"] == 2)
        & np.isclose(history["teacher_gamma"], 0.0)
    ]
    assert dead_history["suffix_live_n"].eq(0).all()
    assert dead_history["suffix_dead_n"].eq(2).all()
    assert dead_history["median"].isna().all()

    surface = render.qelm_error_surface_source(tables)
    dead_surface = surface[
        (surface["teacher_n_qubits"] == 2)
        & np.isclose(surface["teacher_gamma"], 0.0)
    ]
    assert dead_surface["suffix_live_n"].eq(0).all()
    assert dead_surface["suffix_dead_n"].eq(2).all()
    assert dead_surface["median_total_nrmse"].isna().all()

    decomposition = render.decomposition_source(tables)
    dead_decomposition = decomposition[
        (decomposition["teacher_n_qubits"] == 2)
        & np.isclose(decomposition["teacher_gamma"], 0.0)
    ]
    assert dead_decomposition["suffix_live_n"].eq(0).all()
    assert dead_decomposition["suffix_dead_n"].eq(2).all()
    assert dead_decomposition["stacked_total_mse"].isna().all()

    regimes = render.regime_map_source(tables)
    dead_regimes = regimes[
        (regimes["teacher_n_qubits"] == 2)
        & np.isclose(regimes["teacher_gamma"], 0.0)
    ]
    assert set(dead_regimes["modal_regime"]) == {"DEGENERATE_TARGET"}
    assert dead_regimes["agreement_n"].eq(2).all()
    assert dead_regimes["n_roots"].eq(2).all()

    manifest = render.render_all(tables, output)
    assert manifest["outputs"]["fig02_qelm_error_surface.pdf"] == _sha(
        output / "fig02_qelm_error_surface.pdf"
    )

    for label, invalid_value, message in (
        ("nonfinite", np.inf, "must be finite"),
        ("nonpositive", 0.0, "must be positive"),
    ):
        invalid_results = tmp_path / f"invalid_live_{label}"
        _write_synthetic_study(invalid_results)
        suffix_path = invalid_results / "suffix_decomposition.csv"
        invalid_suffix = pd.read_csv(suffix_path)
        row_index = invalid_suffix.index[
            invalid_suffix["model"].eq("ptd_parallel")
        ][0]
        invalid_suffix.loc[row_index, "total_nrmse"] = invalid_value
        invalid_suffix.to_csv(suffix_path, index=False)
        manifest_path = invalid_results / "canonical_manifest.json"
        invalid_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        invalid_manifest["artifacts"][suffix_path.name] = _sha(suffix_path)
        manifest_path.write_text(
            json.dumps(invalid_manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=message):
            render.load_study_tables(invalid_results, CONFIG)
