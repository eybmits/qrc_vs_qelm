"""Fail-closed confirmatory gates for canonical QRC-to-QELM emulation."""

from __future__ import annotations

from hashlib import sha256
from itertools import product
import json
from math import comb
from pathlib import Path
import re
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qrc_vs_qelm.canonical_emulation import classify_emulation_regime
from qrc_vs_qelm.canonical_resources import expected_observable_contract


IDENTITY_ATOL = 1e-12
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260828
DETERMINISTIC_SMOKE_ARTIFACTS = (
    "stream_metrics.csv",
    "suffix_decomposition.csv",
    "system_diagnostics.csv",
    "canonical_summary.json",
)
REQUIRED_ARTIFACTS = (
    *DETERMINISTIC_SMOKE_ARTIFACTS,
    "computational_resources.json",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
STREAM_REQUIRED_COLUMNS = frozenset(
    {
        "root_id",
        "system_id",
        "teacher_n_qubits",
        "teacher_dissipator",
        "teacher_gamma",
        "teacher_coupling_seed",
        "window",
        "model",
        "snapshot_budget",
        "feature_budget",
        "feature_dim",
        "target_observables_sha256",
        "student_n_qubits",
        "student_observables_sha256",
        "mse",
        "rmse",
        "nrmse",
        "mean_row_rms",
        "median_row_rms",
        "max_row_rms",
        "alpha",
        "intercept_only_selected",
        "validation_nrmse",
        "feature_rank",
        "effective_rank",
        "condition_number",
        "target_aligned_fraction",
    }
)
SUFFIX_REQUIRED_COLUMNS = frozenset(
    {
        "root_id",
        "system_id",
        "teacher_n_qubits",
        "teacher_dissipator",
        "teacher_gamma",
        "teacher_coupling_seed",
        "window",
        "model",
        "snapshot_budget",
        "feature_budget",
        "target_observables_sha256",
        "student_n_qubits",
        "student_observables_sha256",
        "n_suffix_groups",
        "branches_per_group",
        "n_outputs",
        "history_mse",
        "representation_mse",
        "total_mse",
        "target_variance",
        "identity_error",
        "history_nrmse",
        "representation_nrmse",
        "total_nrmse",
        "history_fraction",
        "oracle_nrmse",
        "regime",
        "mean_group_diameter",
        "max_group_diameter",
        "half_max_group_diameter",
        "rms_pair_half_bound",
        "prediction_disagreement",
        "repeated_suffix_rmse",
        "worst_branch_rmse",
        "rms_bound_slack",
        "diameter_bound_slack",
        "predictions_agree",
        "rms_half_bound_satisfied",
        "diameter_half_bound_satisfied",
        "suffix_bounds_satisfied",
    }
)
SYSTEM_REQUIRED_COLUMNS = frozenset(
    {
        "root_id",
        "system_id",
        "teacher_n_qubits",
        "teacher_dissipator",
        "teacher_gamma",
        "teacher_coupling_seed",
        "target_dim",
        "target_observables_sha256",
        "target_variance",
        "target_live",
    }
)
LEARNED_MODELS = frozenset(
    {"linear_delay", "poly2_delay", "ptd_parallel", "classical_tanh"}
)
DIRECT_MODELS = frozenset({"rr_same_identity", "constant"})
ALLOWED_REGIMES = frozenset(
    {
        "DEGENERATE_TARGET",
        "HISTORY_IMPOSSIBLE",
        "EMULATED",
        "REPRESENTATION_LIMITED",
        "UNRESOLVED",
    }
)
SUFFIX_BOOLEAN_COLUMNS = (
    "predictions_agree",
    "rms_half_bound_satisfied",
    "diameter_half_bound_satisfied",
    "suffix_bounds_satisfied",
)


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load a JSON object while rejecting extensions and duplicate keys."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            output[key] = value
        return output

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_canonical_artifacts(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load the canonical tables plus strict summary and manifest objects."""

    results = Path(results_dir)
    paths = {name: results / name for name in REQUIRED_ARTIFACTS}
    paths["canonical_manifest.json"] = results / "canonical_manifest.json"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonical artifacts: {missing}")
    summary = load_strict_json(paths["canonical_summary.json"])
    resources = load_strict_json(paths["computational_resources.json"])
    if resources.get("schema_version") != 1 or resources.get("decision_bearing") is not False:
        raise ValueError("invalid computational resource artifact contract")
    manifest = load_strict_json(paths["canonical_manifest.json"])
    return (
        pd.read_csv(paths["stream_metrics.csv"]),
        pd.read_csv(paths["suffix_decomposition.csv"]),
        pd.read_csv(paths["system_diagnostics.csv"]),
        summary,
        manifest,
    )


def verify_canonical_manifest(
    results_dir: Path,
    manifest: Mapping[str, Any],
    *,
    material_root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Verify artifact bytes and the internal provenance hash chain."""

    results = Path(results_dir).resolve()
    checks: dict[str, bool] = {
        "manifest_schema": manifest.get("schema_version") == 1,
    }
    artifacts = manifest.get("artifacts")
    checks["artifact_inventory"] = bool(
        isinstance(artifacts, dict) and set(REQUIRED_ARTIFACTS) <= set(artifacts)
    )
    artifact_details: dict[str, Any] = {}
    if isinstance(artifacts, dict):
        for name, expected in artifacts.items():
            safe_name = isinstance(name, str) and Path(name).name == name
            expected_hash = str(expected)
            path = results / str(name)
            valid_hash = bool(SHA256_PATTERN.fullmatch(expected_hash))
            matches = bool(
                safe_name
                and valid_hash
                and path.is_file()
                and _sha256_file(path) == expected_hash
            )
            artifact_details[str(name)] = {
                "recorded_sha256": expected_hash,
                "matches": matches,
            }
        checks["artifact_hashes"] = bool(
            artifact_details and all(item["matches"] for item in artifact_details.values())
        )
    else:
        checks["artifact_hashes"] = False

    reproducibility = manifest.get("reproducibility")
    checks["reproducibility_schema"] = bool(
        isinstance(reproducibility, dict)
        and reproducibility.get("schema_version") == 1
        and reproducibility.get("capture_phase") == "before_simulation"
    )
    material = reproducibility.get("material") if isinstance(reproducibility, dict) else None
    files = material.get("files") if isinstance(material, dict) else None
    material_valid = isinstance(files, dict) and bool(files)
    recorded_hashes: dict[str, str] = {}
    if isinstance(files, dict):
        for label, entry in files.items():
            if not isinstance(entry, dict):
                material_valid = False
                continue
            digest = str(entry.get("sha256", ""))
            size = entry.get("size_bytes")
            path = entry.get("path")
            if not (
                SHA256_PATTERN.fullmatch(digest)
                and isinstance(size, int)
                and not isinstance(size, bool)
                and size >= 0
                and isinstance(path, str)
                and bool(path)
            ):
                material_valid = False
            recorded_hashes[str(label)] = digest
    checks["material_entries"] = bool(material_valid)
    checks["material_identity"] = bool(
        material_valid
        and str(material.get("identity_sha256", "")) == _sha256_json(recorded_hashes)
    )
    recorded_count = material.get("file_count") if isinstance(material, dict) else None
    checks["material_count"] = bool(
        material_valid
        and isinstance(recorded_count, int)
        and not isinstance(recorded_count, bool)
        and recorded_count == len(recorded_hashes)
    )
    current_material: dict[str, bool] = {}
    if material_root is not None and isinstance(files, dict):
        root = Path(material_root).resolve()
        for label, entry in files.items():
            matches = False
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                relative = Path(entry["path"])
                if not relative.is_absolute():
                    candidate = (root / relative).resolve()
                    try:
                        candidate.relative_to(root)
                    except ValueError:
                        pass
                    else:
                        matches = bool(
                            candidate.is_file()
                            and candidate.stat().st_size == entry.get("size_bytes")
                            and _sha256_file(candidate) == entry.get("sha256")
                        )
            current_material[str(label)] = matches
        checks["current_material_matches"] = bool(
            current_material and all(current_material.values())
        )
    if config_path is not None:
        config = Path(config_path)
        checks["selected_config_matches"] = bool(
            config.is_file()
            and _sha256_file(config) == manifest.get("config_sha256")
        )
    if isinstance(reproducibility, dict):
        without_identity = dict(reproducibility)
        recorded_identity = str(without_identity.pop("identity_sha256", ""))
        checks["reproducibility_identity"] = bool(
            SHA256_PATTERN.fullmatch(recorded_identity)
            and recorded_identity == _sha256_json(without_identity)
        )
    else:
        checks["reproducibility_identity"] = False
    checks["top_level_hash_links"] = bool(
        material_valid
        and manifest.get("config_sha256")
        == recorded_hashes.get("configuration")
        and manifest.get("module_sha256")
        == recorded_hashes.get("source:src/qrc_vs_qelm/canonical_study.py")
        and manifest.get("runner_sha256") == recorded_hashes.get("runner")
    )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "artifacts": artifact_details,
        "current_material_matches": current_material,
        "material_identity_sha256": (
            material.get("identity_sha256") if isinstance(material, dict) else None
        ),
    }


def verify_duplicate_smoke_runs(
    first_dir: Path,
    second_dir: Path,
    *,
    expected_material_identity_sha256: str,
) -> dict[str, Any]:
    """Check two smoke runs byte-for-byte on every scientific artifact."""

    first_dir = Path(first_dir).resolve()
    second_dir = Path(second_dir).resolve()
    distinct_directories = first_dir != second_dir
    _, _, _, first_summary, first_manifest = load_canonical_artifacts(first_dir)
    _, _, _, second_summary, second_manifest = load_canonical_artifacts(second_dir)
    first_audit = verify_canonical_manifest(first_dir, first_manifest)
    second_audit = verify_canonical_manifest(second_dir, second_manifest)
    first_artifacts = first_manifest.get("artifacts", {})
    second_artifacts = second_manifest.get("artifacts", {})
    matches = {
        name: bool(first_artifacts.get(name) == second_artifacts.get(name))
        for name in DETERMINISTIC_SMOKE_ARTIFACTS
    }
    same_material = bool(
        first_audit.get("material_identity_sha256")
        == second_audit.get("material_identity_sha256")
    )
    material_identity = first_audit.get("material_identity_sha256")
    matches_expected_material = bool(
        SHA256_PATTERN.fullmatch(str(expected_material_identity_sha256))
        and material_identity == str(expected_material_identity_sha256)
    )
    passed = bool(
        first_summary.get("profile") == "smoke"
        and second_summary.get("profile") == "smoke"
        and distinct_directories
        and first_audit["passed"]
        and second_audit["passed"]
        and same_material
        and matches_expected_material
        and all(matches.values())
    )
    return {
        "passed": passed,
        "distinct_directories": distinct_directories,
        "same_material": same_material,
        "material_identity_sha256": material_identity,
        "expected_material_identity_sha256": expected_material_identity_sha256,
        "matches_expected_material": matches_expected_material,
        "artifact_hashes_match": matches,
        "first_manifest_valid": bool(first_audit["passed"]),
        "second_manifest_valid": bool(second_audit["passed"]),
        "manifest_sha256": {
            "first": _sha256_file(first_dir / "canonical_manifest.json"),
            "second": _sha256_file(second_dir / "canonical_manifest.json"),
        },
        "resource_sha256": {
            "first": _sha256_file(first_dir / "computational_resources.json"),
            "second": _sha256_file(second_dir / "computational_resources.json"),
        },
    }


def verify_pytest_junit(
    path: Path,
    *,
    expected_testcases: int,
    expected_material_identity_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Validate a pytest JUnit record and bind it to the evaluated material."""

    junit = Path(path)
    root = ET.parse(junit).getroot()
    cases = root.findall(".//testcase")
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    count_matches = bool(
        int(expected_testcases) > 0
        and len(cases) == int(expected_testcases)
    )
    property_items = root.findall(".//property")
    properties: dict[str, str] = {}
    duplicate_properties: set[str] = set()
    for item in property_items:
        name = str(item.attrib.get("name", ""))
        value = str(item.attrib.get("value", ""))
        if name in properties:
            duplicate_properties.add(name)
        properties[name] = value
    recorded_material = properties.get("canonical_material_identity_sha256")
    recorded_config = properties.get("canonical_config_sha256")
    binding_matches = bool(
        not duplicate_properties
        and SHA256_PATTERN.fullmatch(str(expected_material_identity_sha256))
        and SHA256_PATTERN.fullmatch(str(expected_config_sha256))
        and recorded_material == str(expected_material_identity_sha256)
        and recorded_config == str(expected_config_sha256)
    )
    passed = bool(
        cases
        and failures == 0
        and errors == 0
        and count_matches
        and binding_matches
    )
    return {
        "passed": passed,
        "testcases": len(cases),
        "expected_full_suite_testcases": expected_testcases,
        "complete_suite_count_matches": count_matches,
        "binding_matches": binding_matches,
        "recorded_material_identity_sha256": recorded_material,
        "expected_material_identity_sha256": expected_material_identity_sha256,
        "recorded_config_sha256": recorded_config,
        "expected_config_sha256": expected_config_sha256,
        "duplicate_property_names": sorted(duplicate_properties),
        "failures": failures,
        "errors": errors,
        "sha256": _sha256_file(junit),
    }


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Return a numeric matrix, coercing malformed artifact cells to NaN."""

    return (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )


def _finite_nonnegative(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    values = _numeric(frame, columns)
    return bool(values.size and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _nonnegative_or_positive_infinity(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> bool:
    """Accept normalized errors for an exactly zero-variance target."""

    values = _numeric(frame, columns)
    return bool(
        values.size
        and not np.any(np.isnan(values))
        and np.all((np.isfinite(values) & (values >= 0.0)) | np.isposinf(values))
    )


def _integer_domain(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    minimum: int,
) -> bool:
    values = _numeric(frame, columns)
    return bool(
        values.size
        and np.all(np.isfinite(values))
        and np.all(values >= int(minimum))
        and np.all(values == np.floor(values))
    )


def _nullable_positive_integer(series: pd.Series) -> bool:
    present = series[~series.isna()]
    if present.empty:
        return True
    values = pd.to_numeric(present, errors="coerce").to_numpy(dtype=float)
    return bool(
        np.all(np.isfinite(values))
        and np.all(values >= 1.0)
        and np.all(values == np.floor(values))
    )


def _nonempty_strings(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    for column in columns:
        values = frame[column]
        if values.isna().any() or not values.astype(str).str.strip().ne("").all():
            return False
    return True


def _sha256_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return bool(
        all(
            frame[column]
            .astype(str)
            .map(lambda value: bool(SHA256_PATTERN.fullmatch(value)))
            .all()
            for column in columns
        )
    )


def _boolean_domain(series: pd.Series, label: str) -> bool:
    try:
        _bool_values(series, label)
    except ValueError:
        return False
    return True


def _bool_values_fail_closed(series: pd.Series, label: str) -> np.ndarray:
    """Parse artifact booleans, mapping a malformed column to all-false."""

    try:
        return _bool_values(series, label)
    except ValueError:
        return np.zeros(len(series), dtype=bool)


def _normalized_metric_identity(
    mse: np.ndarray,
    variance: np.ndarray,
    observed: np.ndarray,
) -> bool:
    """Check the canonical sqrt(MSE/variance) rule, including zero variance."""

    squared = np.asarray(mse, dtype=float)
    scale = np.asarray(variance, dtype=float)
    actual = np.asarray(observed, dtype=float)
    if not (
        squared.shape == scale.shape == actual.shape
        and squared.size
        and np.all(np.isfinite(squared))
        and np.all(squared >= 0.0)
        and np.all(np.isfinite(scale))
        and np.all(scale >= 0.0)
        and not np.any(np.isnan(actual))
    ):
        return False
    expected = np.empty_like(squared)
    live = scale > 0.0
    expected[live] = np.sqrt(squared[live] / scale[live])
    dead_nonzero = (~live) & (squared > 0.0)
    expected[dead_nonzero] = np.inf
    expected[(~live) & (~dead_nonzero)] = 0.0
    return bool(
        np.all(
            np.isclose(actual, expected, rtol=1e-12, atol=IDENTITY_ATOL)
            | (np.isposinf(actual) & np.isposinf(expected))
        )
    )


def _stream_nrmse_identity(
    stream: pd.DataFrame,
    systems: pd.DataFrame,
) -> bool:
    keys = ["root_id", "system_id"]
    if systems.duplicated(keys).any():
        return False
    variance_by_system = {
        (str(row.root_id), str(row.system_id)): float(row.target_variance)
        for row in systems.itertuples(index=False)
    }
    try:
        variance = np.asarray(
            [
                variance_by_system[(str(row.root_id), str(row.system_id))]
                for row in stream.itertuples(index=False)
            ],
            dtype=float,
        )
    except KeyError:
        return False
    return _normalized_metric_identity(
        _numeric(stream, ("mse",))[:, 0],
        variance,
        _numeric(stream, ("nrmse",))[:, 0],
    )


def _suffix_normalized_metric_identities(suffix: pd.DataFrame) -> bool:
    variance = _numeric(suffix, ("target_variance",))[:, 0]
    return bool(
        all(
            _normalized_metric_identity(
                _numeric(suffix, (mse_column,))[:, 0],
                variance,
                _numeric(suffix, (nrmse_column,))[:, 0],
            )
            for mse_column, nrmse_column in (
                ("history_mse", "history_nrmse"),
                ("representation_mse", "representation_nrmse"),
                ("total_mse", "total_nrmse"),
            )
        )
    )


def _history_fraction_identity(suffix: pd.DataFrame) -> bool:
    history = _numeric(suffix, ("history_mse",))[:, 0]
    total = _numeric(suffix, ("total_mse",))[:, 0]
    observed = _numeric(suffix, ("history_fraction",))[:, 0]
    if not (
        history.size
        and np.all(np.isfinite(history))
        and np.all(history >= 0.0)
        and np.all(np.isfinite(total))
        and np.all(total >= 0.0)
        and np.all(np.isfinite(observed))
    ):
        return False
    expected = np.zeros_like(history)
    positive = total > 0.0
    expected[positive] = history[positive] / total[positive]
    return bool(
        np.allclose(observed, expected, rtol=1e-12, atol=IDENTITY_ATOL)
    )


def _suffix_regimes_match(
    suffix: pd.DataFrame,
    *,
    tolerance: float,
    min_target_variance: float,
) -> bool:
    expected: list[str] = []
    try:
        for row in suffix.itertuples(index=False):
            expected.append(
                classify_emulation_regime(
                    history_nrmse=float(row.history_nrmse),
                    student_nrmse=float(row.total_nrmse),
                    oracle_nrmse=float(row.oracle_nrmse),
                    target_variance=float(row.target_variance),
                    tolerance=float(tolerance),
                    min_target_variance=float(min_target_variance),
                )
            )
    except (TypeError, ValueError):
        return False
    return bool(
        np.array_equal(
            suffix["regime"].astype(str).to_numpy(),
            np.asarray(expected, dtype=object),
        )
    )


def _suffix_student_independent_invariance(suffix: pd.DataFrame) -> bool:
    """Check quantities fixed by a teacher suffix ensemble across students."""

    invariant_columns = (
        "history_mse",
        "history_nrmse",
        "target_variance",
        "oracle_nrmse",
        "mean_group_diameter",
        "max_group_diameter",
        "half_max_group_diameter",
        "rms_pair_half_bound",
    )
    keys = ["root_id", "system_id", "window"]
    if suffix.empty:
        return False
    for _, group in suffix.groupby(keys, sort=False):
        values = _numeric(group, invariant_columns)
        if not values.size or np.any(np.isnan(values)):
            return False
        reference = values[0]
        equal = np.isclose(
            values,
            reference[None, :],
            rtol=0.0,
            atol=0.0,
        ) | (np.isposinf(values) & np.isposinf(reference[None, :]))
        if not np.all(equal):
            return False
        oracle_rows = group[
            (group["model"].astype(str) == "rr_same_identity")
            & group["snapshot_budget"].isna()
        ]
        if len(oracle_rows) != 1:
            return False
        oracle_total = float(oracle_rows.iloc[0]["total_nrmse"])
        recorded_oracle = float(reference[3])
        if not (
            np.isclose(
                recorded_oracle,
                oracle_total,
                rtol=1e-12,
                atol=IDENTITY_ATOL,
            )
            or (np.isposinf(recorded_oracle) and np.isposinf(oracle_total))
        ):
            return False
    return True


def _scientific_schema_report(
    stream: pd.DataFrame,
    suffix: pd.DataFrame,
    systems: pd.DataFrame,
    *,
    tolerance: float,
    min_target_variance: float,
) -> dict[str, Any]:
    """Fail-closed domains for every scientific quantity used by the gates."""

    learned = stream[stream["model"].astype(str).isin(LEARNED_MODELS)]
    direct = stream[stream["model"].astype(str).isin(DIRECT_MODELS)]

    stream_finite_core = (
        "mse",
        "rmse",
        "mean_row_rms",
        "median_row_rms",
        "max_row_rms",
    )
    stream_core_values = _numeric(stream, stream_finite_core)
    stream_rmse = _numeric(stream, ("rmse",))[:, 0]
    stream_mse = _numeric(stream, ("mse",))[:, 0]
    mean_row = _numeric(stream, ("mean_row_rms",))[:, 0]
    median_row = _numeric(stream, ("median_row_rms",))[:, 0]
    max_row = _numeric(stream, ("max_row_rms",))[:, 0]

    learned_alpha = _numeric(learned, ("alpha",))[:, 0]
    learned_validation = _numeric(learned, ("validation_nrmse",))[:, 0]
    learned_rank = _numeric(learned, ("feature_rank",))[:, 0]
    learned_width = _numeric(learned, ("feature_dim",))[:, 0]
    learned_effective_rank = _numeric(learned, ("effective_rank",))[:, 0]
    learned_condition = _numeric(learned, ("condition_number",))[:, 0]
    learned_alignment = _numeric(learned, ("target_aligned_fraction",))[:, 0]
    learned_bool_ok = _boolean_domain(
        learned["intercept_only_selected"],
        "intercept_only_selected",
    )
    learned_intercept = (
        _bool_values_fail_closed(
            learned["intercept_only_selected"],
            "intercept_only_selected",
        )
        if len(learned)
        else np.asarray([], dtype=bool)
    )
    alpha_domain = bool(
        len(learned)
        and np.all(
            (np.isfinite(learned_alpha) & (learned_alpha >= 0.0))
            | np.isposinf(learned_alpha)
        )
    )
    intercept_consistent = bool(
        learned_bool_ok
        and len(learned_intercept) == len(learned_alpha)
        and np.array_equal(learned_intercept, np.isposinf(learned_alpha))
    )
    direct_diagnostic_columns = (
        "alpha",
        "intercept_only_selected",
        "validation_nrmse",
        "feature_rank",
        "effective_rank",
        "condition_number",
        "target_aligned_fraction",
    )

    suffix_finite_risks = (
        "history_mse",
        "representation_mse",
        "total_mse",
        "target_variance",
    )
    suffix_normalized_risks = (
        "history_nrmse",
        "representation_nrmse",
        "total_nrmse",
        "oracle_nrmse",
    )
    suffix_witness_nonnegative = (
        "mean_group_diameter",
        "max_group_diameter",
        "half_max_group_diameter",
        "rms_pair_half_bound",
        "prediction_disagreement",
        "repeated_suffix_rmse",
        "worst_branch_rmse",
    )
    suffix_signed = _numeric(
        suffix,
        ("identity_error", "rms_bound_slack", "diameter_bound_slack"),
    )
    history_fraction = _numeric(suffix, ("history_fraction",))[:, 0]
    mean_diameter = _numeric(suffix, ("mean_group_diameter",))[:, 0]
    max_diameter = _numeric(suffix, ("max_group_diameter",))[:, 0]
    half_diameter = _numeric(suffix, ("half_max_group_diameter",))[:, 0]
    rms_pair_bound = _numeric(suffix, ("rms_pair_half_bound",))[:, 0]
    repeated_rmse = _numeric(suffix, ("repeated_suffix_rmse",))[:, 0]
    worst_branch_rmse = _numeric(suffix, ("worst_branch_rmse",))[:, 0]
    rms_slack = _numeric(suffix, ("rms_bound_slack",))[:, 0]
    diameter_slack = _numeric(suffix, ("diameter_bound_slack",))[:, 0]

    suffix_booleans_ok = bool(
        all(
            _boolean_domain(suffix[column], column)
            for column in SUFFIX_BOOLEAN_COLUMNS
        )
    )
    parsed_booleans = {
        column: _bool_values_fail_closed(suffix[column], column)
        for column in SUFFIX_BOOLEAN_COLUMNS
    }
    expected_agreement = (
        _numeric(suffix, ("prediction_disagreement",))[:, 0] <= IDENTITY_ATOL
    )
    expected_rms = expected_agreement & (rms_slack >= -IDENTITY_ATOL)
    expected_diameter = expected_agreement & (
        diameter_slack >= -IDENTITY_ATOL
    )

    system_variance = _numeric(systems, ("target_variance",))[:, 0]
    checks = {
        "stream_identifiers_nonempty": _nonempty_strings(
            stream,
            ("root_id", "system_id", "teacher_dissipator", "model"),
        ),
        "stream_hashes_valid": _sha256_columns(
            stream,
            ("target_observables_sha256", "student_observables_sha256"),
        ),
        "stream_integer_domains": _integer_domain(
            stream,
            (
                "teacher_n_qubits",
                "teacher_coupling_seed",
                "window",
                "student_n_qubits",
                "feature_budget",
                "feature_dim",
            ),
            minimum=0,
        )
        and bool(
            (_numeric(stream, ("teacher_n_qubits", "window", "student_n_qubits")) >= 1)
            .all()
        )
        and _nullable_positive_integer(stream["snapshot_budget"]),
        "stream_gamma_domain": _finite_nonnegative(stream, ("teacher_gamma",)),
        "stream_models_known": bool(
            set(stream["model"].astype(str)) == LEARNED_MODELS | DIRECT_MODELS
        ),
        "stream_core_metrics_finite_nonnegative": bool(
            stream_core_values.size
            and np.all(np.isfinite(stream_core_values))
            and np.all(stream_core_values >= 0.0)
            and _nonnegative_or_positive_infinity(stream, ("nrmse",))
        ),
        "stream_rmse_matches_mse": bool(
            np.allclose(
                stream_rmse,
                np.sqrt(np.maximum(stream_mse, 0.0)),
                rtol=1e-12,
                atol=1e-12,
            )
        ),
        "stream_nrmse_matches_system_variance": _stream_nrmse_identity(
            stream,
            systems,
        ),
        "stream_row_rms_ordered": bool(
            np.all(mean_row <= max_row + IDENTITY_ATOL)
            and np.all(median_row <= max_row + IDENTITY_ATOL)
        ),
        "stream_learned_rows_present": bool(len(learned) and len(direct)),
        "stream_learned_alpha_domain": alpha_domain,
        "stream_learned_intercept_flag_consistent": intercept_consistent,
        "stream_learned_validation_finite_nonnegative": bool(
            learned_validation.size
            and not np.any(np.isnan(learned_validation))
            and np.all(
                (np.isfinite(learned_validation) & (learned_validation >= 0.0))
                | np.isposinf(learned_validation)
            )
        ),
        "stream_learned_rank_domain": bool(
            learned_rank.size
            and np.all(np.isfinite(learned_rank))
            and np.all(learned_rank == np.floor(learned_rank))
            and np.all(learned_rank >= 0.0)
            and np.all(learned_rank <= learned_width)
            and np.all(np.isfinite(learned_effective_rank))
            and np.all(learned_effective_rank >= 0.0)
            and np.all(learned_effective_rank <= learned_width + IDENTITY_ATOL)
        ),
        "stream_learned_condition_domain": bool(
            learned_condition.size
            and not np.any(np.isnan(learned_condition))
            and np.all(
                (np.isfinite(learned_condition) & (learned_condition >= 1.0))
                | np.isposinf(learned_condition)
            )
        ),
        "stream_learned_alignment_domain": bool(
            learned_alignment.size
            and np.all(np.isfinite(learned_alignment))
            and np.all(learned_alignment >= 0.0)
            and np.all(learned_alignment <= 1.0)
        ),
        "stream_direct_diagnostics_null": bool(
            len(direct)
            and direct.loc[:, list(direct_diagnostic_columns)].isna().all().all()
        ),
        "suffix_identifiers_nonempty": _nonempty_strings(
            suffix,
            ("root_id", "system_id", "teacher_dissipator", "model", "regime"),
        ),
        "suffix_hashes_valid": _sha256_columns(
            suffix,
            ("target_observables_sha256", "student_observables_sha256"),
        ),
        "suffix_integer_domains": _integer_domain(
            suffix,
            (
                "teacher_n_qubits",
                "teacher_coupling_seed",
                "window",
                "student_n_qubits",
                "feature_budget",
            ),
            minimum=0,
        )
        and bool(
            (_numeric(suffix, ("teacher_n_qubits", "window", "student_n_qubits")) >= 1)
            .all()
        )
        and _integer_domain(
            suffix,
            ("n_suffix_groups", "branches_per_group", "n_outputs"),
            minimum=1,
        )
        and _nullable_positive_integer(suffix["snapshot_budget"]),
        "suffix_gamma_domain": _finite_nonnegative(suffix, ("teacher_gamma",)),
        "suffix_models_known": bool(
            set(suffix["model"].astype(str)) == LEARNED_MODELS | DIRECT_MODELS
        ),
        "suffix_risk_metrics_finite_nonnegative": _finite_nonnegative(
            suffix,
            suffix_finite_risks,
        )
        and _nonnegative_or_positive_infinity(
            suffix,
            suffix_normalized_risks,
        ),
        "suffix_signed_metrics_finite": bool(
            suffix_signed.size and np.all(np.isfinite(suffix_signed))
        ),
        "suffix_history_fraction_domain": bool(
            history_fraction.size
            and np.all(np.isfinite(history_fraction))
            and np.all(history_fraction >= -IDENTITY_ATOL)
            and np.all(history_fraction <= 1.0 + IDENTITY_ATOL)
        ),
        "suffix_normalized_metric_identities": (
            _suffix_normalized_metric_identities(suffix)
        ),
        "suffix_history_fraction_identity": _history_fraction_identity(suffix),
        "suffix_regime_domain": bool(
            set(suffix["regime"].astype(str)) <= ALLOWED_REGIMES
            and len(suffix)
        ),
        "suffix_regime_matches_frozen_classification": _suffix_regimes_match(
            suffix,
            tolerance=float(tolerance),
            min_target_variance=float(min_target_variance),
        ),
        "suffix_student_independent_invariance": (
            _suffix_student_independent_invariance(suffix)
        ),
        "suffix_witness_metrics_finite_nonnegative": _finite_nonnegative(
            suffix,
            suffix_witness_nonnegative,
        ),
        "suffix_witness_geometry_consistent": bool(
            np.all(mean_diameter <= max_diameter + IDENTITY_ATOL)
            and np.all(rms_pair_bound <= half_diameter + IDENTITY_ATOL)
            and np.allclose(
                half_diameter,
                0.5 * max_diameter,
                rtol=1e-12,
                atol=IDENTITY_ATOL,
            )
            and np.allclose(
                rms_slack,
                repeated_rmse - rms_pair_bound,
                rtol=1e-12,
                atol=IDENTITY_ATOL,
            )
            and np.allclose(
                diameter_slack,
                worst_branch_rmse - half_diameter,
                rtol=1e-12,
                atol=IDENTITY_ATOL,
            )
        ),
        "suffix_boolean_domain": suffix_booleans_ok,
        "suffix_boolean_semantics": bool(
            suffix_booleans_ok
            and np.array_equal(parsed_booleans["predictions_agree"], expected_agreement)
            and np.array_equal(
                parsed_booleans["rms_half_bound_satisfied"],
                expected_rms,
            )
            and np.array_equal(
                parsed_booleans["diameter_half_bound_satisfied"],
                expected_diameter,
            )
            and np.array_equal(
                parsed_booleans["suffix_bounds_satisfied"],
                expected_rms & expected_diameter,
            )
        ),
        "systems_identifiers_nonempty": _nonempty_strings(
            systems,
            ("root_id", "system_id", "teacher_dissipator"),
        ),
        "systems_hashes_valid": _sha256_columns(
            systems,
            ("target_observables_sha256",),
        ),
        "systems_integer_domains": _integer_domain(
            systems,
            ("teacher_n_qubits", "teacher_coupling_seed", "target_dim"),
            minimum=0,
        )
        and bool(
            (_numeric(systems, ("teacher_n_qubits", "target_dim")) >= 1).all()
        ),
        "systems_gamma_domain": _finite_nonnegative(systems, ("teacher_gamma",)),
        "systems_target_variance_finite_nonnegative": bool(
            system_variance.size
            and np.all(np.isfinite(system_variance))
            and np.all(system_variance >= 0.0)
        ),
        "systems_target_live_boolean": _boolean_domain(
            systems["target_live"],
            "target_live",
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
    }


def _gate_decision(profile: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "tolerance_nrmse",
        "min_target_variance",
        "min_root_successes",
        "roots_total",
        "min_spearman",
        "min_g2_live_systems_per_family",
        "min_window_gain",
        "min_feature_gain",
        "min_best_classical_gain",
        "min_scaling_r_squared",
        "g4_gammas",
        "g5_gamma",
    }
    raw = profile.get("decision")
    if not isinstance(raw, Mapping):
        raise ValueError("profile must expose the frozen decision mapping")
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"decision mapping is missing keys: {missing}")
    if not isinstance(raw["g4_gammas"], (list, tuple)):
        raise ValueError("decision.g4_gammas must be a list")
    for key in (
        "min_root_successes",
        "roots_total",
        "min_g2_live_systems_per_family",
    ):
        if isinstance(raw[key], bool) or not isinstance(raw[key], (int, np.integer)):
            raise ValueError(f"decision.{key} must be an integer")
    decision = {
        key: float(raw[key])
        for key in (
            "tolerance_nrmse",
            "min_target_variance",
            "min_spearman",
            "min_window_gain",
            "min_feature_gain",
            "min_best_classical_gain",
            "min_scaling_r_squared",
            "g5_gamma",
        )
    }
    decision["min_root_successes"] = int(raw["min_root_successes"])
    decision["roots_total"] = int(raw["roots_total"])
    decision["min_g2_live_systems_per_family"] = int(
        raw["min_g2_live_systems_per_family"]
    )
    decision["g4_gammas"] = [float(value) for value in raw["g4_gammas"]]
    if (
        decision["roots_total"] < 1
        or not 1 <= decision["min_root_successes"] <= decision["roots_total"]
        or not 1
        <= decision["min_g2_live_systems_per_family"]
        <= len(profile["teacher"]["gammas"])
    ):
        raise ValueError("invalid frozen root thresholds")
    numeric = [
        value
        for key, value in decision.items()
        if key
        not in {
            "min_root_successes",
            "roots_total",
            "min_g2_live_systems_per_family",
            "g4_gammas",
        }
    ] + decision["g4_gammas"]
    if not numeric or not np.all(np.isfinite(np.asarray(numeric, dtype=float))):
        raise ValueError("decision contrasts must be finite")
    if decision["tolerance_nrmse"] <= 0.0 or decision["min_target_variance"] <= 0.0:
        raise ValueError("decision error thresholds must be positive")
    bounded = (
        "min_spearman",
        "min_window_gain",
        "min_feature_gain",
        "min_best_classical_gain",
        "min_scaling_r_squared",
    )
    if any(not 0.0 <= decision[key] <= 1.0 for key in bounded):
        raise ValueError("decision correlations, gains, and R-squared must be in [0, 1]")
    common_gammas = {float(value) for value in profile["teacher"]["gammas"]}
    if (
        not decision["g4_gammas"]
        or len(set(decision["g4_gammas"])) != len(decision["g4_gammas"])
        or not set(decision["g4_gammas"]) <= common_gammas
        or decision["g5_gamma"] not in common_gammas
    ):
        raise ValueError("frozen contrast gammas must belong to the teacher grid")
    return decision


def _bool_values(series: pd.Series, label: str) -> np.ndarray:
    values: list[bool] = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
            values.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            values.append(value.strip().lower() == "true")
        else:
            raise ValueError(f"{label} contains a non-boolean value: {value!r}")
    return np.asarray(values, dtype=bool)


def _budget_key(value: Any) -> int | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or int(numeric) != numeric:
        raise ValueError(f"invalid snapshot budget: {value!r}")
    return int(numeric)


def _expected_model_cells(snapshot_budgets: list[int]) -> set[tuple[str, int | None]]:
    cells: set[tuple[str, int | None]] = {
        ("rr_same_identity", None),
        ("constant", None),
        ("linear_delay", None),
        ("poly2_delay", None),
    }
    cells.update(
        (model, int(budget))
        for model in ("ptd_parallel", "classical_tanh")
        for budget in snapshot_budgets
    )
    return cells


def _coverage_report(
    stream: pd.DataFrame,
    suffix: pd.DataFrame,
    systems: pd.DataFrame,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    root_ids = [str(root["root_id"]) for root in profile["roots"]]
    teacher_ns = [int(value) for value in profile["teacher_n_qubits"]]
    dissipators = [str(value) for value in profile["teacher_dissipators"]]
    gammas = [float(value) for value in profile["teacher_gammas"]]
    windows = [int(value) for value in profile["windows"]]
    budgets = [int(value) for value in profile["snapshot_budgets"]]
    model_cells = _expected_model_cells(budgets)
    student_n = int(profile["student"]["n_qubits"])
    student_contract = expected_observable_contract(
        student_n,
        profile["student"]["observables"],
    )
    teacher_contracts = {
        int(n_qubits): expected_observable_contract(
            int(n_qubits),
            profile["teacher"]["target_observables"],
        )
        for n_qubits in teacher_ns
    }
    expected_systems = len(root_ids) * len(teacher_ns) * len(dissipators) * len(gammas)
    expected_metric_rows = expected_systems * len(windows) * len(model_cells)

    system_keys = {
        (
            str(row.root_id),
            int(row.teacher_n_qubits),
            str(row.teacher_dissipator),
            float(row.teacher_gamma),
        )
        for row in systems.itertuples(index=False)
    }
    expected_system_keys = set(product(root_ids, teacher_ns, dissipators, gammas))
    metric_key = ["root_id", "system_id", "window", "model", "snapshot_budget"]
    expected_metric_groups = {
        (str(row.root_id), str(row.system_id))
        for row in systems.itertuples(index=False)
    }
    system_metadata = {
        (str(row.root_id), str(row.system_id)): (
            int(row.teacher_n_qubits),
            str(row.teacher_dissipator),
            float(row.teacher_gamma),
            int(row.target_dim),
            str(row.target_observables_sha256),
            int(row.teacher_coupling_seed),
        )
        for row in systems.itertuples(index=False)
    }

    def expected_feature_width(
        model: str,
        budget: int | None,
        window: int,
        target_dim: int,
    ) -> int:
        if model == "rr_same_identity":
            return int(target_dim)
        if model == "constant":
            return 0
        if model == "linear_delay":
            return int(window)
        if model == "poly2_delay":
            return int(window + window * (window + 1) // 2)
        if model in {"ptd_parallel", "classical_tanh"} and budget is not None:
            return int(budget * student_contract["count"])
        raise ValueError(f"unexpected model/budget cell: {model!r}, {budget!r}")

    def complete_metric_cells(frame: pd.DataFrame, *, suffix_panel: bool) -> bool:
        if frame.duplicated(metric_key).any():
            return False
        observed_groups = {
            (str(root_id), str(system_id))
            for root_id, system_id in frame[["root_id", "system_id"]].itertuples(
                index=False, name=None
            )
        }
        if observed_groups != expected_metric_groups:
            return False
        for (root_id, system_id), group in frame.groupby(
            ["root_id", "system_id"], sort=False
        ):
            expected_metadata = system_metadata.get((str(root_id), str(system_id)))
            observed_metadata = {
                (
                    int(row.teacher_n_qubits),
                    str(row.teacher_dissipator),
                    float(row.teacher_gamma),
                    int(row.n_outputs if suffix_panel else row.feature_dim)
                    if str(row.model) == "rr_same_identity"
                    else expected_metadata[3],
                    str(row.target_observables_sha256),
                    int(row.teacher_coupling_seed),
                )
                for row in group.itertuples(index=False)
            }
            if expected_metadata is None or observed_metadata != {expected_metadata}:
                return False
            if set(group["student_n_qubits"].astype(int)) != {student_n}:
                return False
            if set(group["student_observables_sha256"].astype(str)) != {
                str(student_contract["names_sha256"])
            }:
                return False
            if set(int(value) for value in group["window"]) != set(windows):
                return False
            for window, rows in group.groupby("window", sort=False):
                observed = {
                    (str(row.model), _budget_key(row.snapshot_budget))
                    for row in rows.itertuples(index=False)
                }
                if observed != model_cells:
                    return False
                for row in rows.itertuples(index=False):
                    budget = _budget_key(row.snapshot_budget)
                    expected_width = expected_feature_width(
                        str(row.model),
                        budget,
                        int(window),
                        int(expected_metadata[3]),
                    )
                    if int(row.feature_budget) != expected_width:
                        return False
                    if not suffix_panel and int(row.feature_dim) != expected_width:
                        return False
                    if suffix_panel and (
                        int(row.n_suffix_groups) != int(profile["suffix_groups"])
                        or int(row.branches_per_group)
                        != int(profile["prefix_branches"])
                        or int(row.n_outputs) != int(expected_metadata[3])
                    ):
                        return False
        return True

    schema = _scientific_schema_report(
        stream,
        suffix,
        systems,
        tolerance=float(profile["decision"]["tolerance_nrmse"]),
        min_target_variance=float(
            profile["decision"]["min_target_variance"]
        ),
    )
    live_values = _bool_values_fail_closed(systems["target_live"], "target_live")
    target_variance = systems["target_variance"].to_numpy(dtype=float)
    system_contract_ok = True
    for row in systems.itertuples(index=False):
        contract = teacher_contracts[int(row.teacher_n_qubits)]
        system_contract_ok = bool(
            system_contract_ok
            and int(row.target_dim) == int(contract["count"])
            and str(row.target_observables_sha256)
            == str(contract["names_sha256"])
        )
    checks = {
        "root_ids": set(systems["root_id"].astype(str)) == set(root_ids),
        "system_count": len(systems) == expected_systems,
        "system_keys": system_keys == expected_system_keys,
        "system_ids_unique": not systems["system_id"].duplicated().any(),
        "observable_contracts": system_contract_ok,
        "target_variance_finite": bool(np.all(np.isfinite(target_variance))),
        "target_live_consistent": bool(
            np.array_equal(
                live_values,
                target_variance
                >= float(profile["decision"]["min_target_variance"]),
            )
        ),
        "stream_row_count": len(stream) == expected_metric_rows,
        "suffix_row_count": len(suffix) == expected_metric_rows,
        "stream_cells_complete": complete_metric_cells(stream, suffix_panel=False),
        "suffix_cells_complete": complete_metric_cells(suffix, suffix_panel=True),
        "scientific_schema": bool(schema["passed"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "scientific_schema": schema,
        "expected": {
            "roots": len(root_ids),
            "systems": expected_systems,
            "stream_rows": expected_metric_rows,
            "suffix_rows": expected_metric_rows,
        },
        "observed": {
            "roots": int(systems["root_id"].nunique()),
            "systems": int(len(systems)),
            "stream_rows": int(len(stream)),
            "suffix_rows": int(len(suffix)),
        },
    }


def _history_frame(suffix: pd.DataFrame) -> pd.DataFrame:
    frame = suffix.loc[
        suffix["model"] == "rr_same_identity",
        [
            "root_id",
            "system_id",
            "teacher_n_qubits",
            "teacher_dissipator",
            "teacher_gamma",
            "window",
            "history_nrmse",
            "oracle_nrmse",
            "target_variance",
        ],
    ].copy()
    key = ["root_id", "system_id", "window"]
    if frame.empty or frame.duplicated(key).any():
        raise ValueError("rr_same_identity suffix rows must be unique per system/window")
    return frame


def _spearman_by_root(
    history: pd.DataFrame,
    windows: list[int],
    roots: list[str],
    *,
    teacher_n: int | None = None,
    comparison_column: str = "stream_rr_nrmse",
    min_variance: float,
    min_live_systems_per_family: int,
) -> tuple[dict[str, float | None], dict[str, dict[str, int]]]:
    selected = history
    if teacher_n is not None:
        selected = selected[selected["teacher_n_qubits"] == int(teacher_n)]
    output: dict[str, float | None] = {str(root_id): None for root_id in roots}
    eligible_counts: dict[str, dict[str, int]] = {
        str(root_id): {} for root_id in roots
    }
    selected_families = sorted(set(selected["teacher_n_qubits"].astype(int)))
    for root_id in roots:
        root_rows = selected[selected["root_id"].astype(str) == str(root_id)]
        correlations: list[float] = []
        undefined = False
        for family in selected_families:
            family_rows = root_rows[
                root_rows["teacher_n_qubits"] == int(family)
            ]
            family_count = 0
            for _, group in family_rows.groupby("system_id", sort=True):
                ordered = group.sort_values("window")
                if ordered["window"].astype(int).tolist() != sorted(windows):
                    undefined = True
                    break
                variances = ordered["target_variance"].to_numpy(dtype=float)
                if not (
                    np.all(np.isfinite(variances))
                    and np.all(variances >= float(min_variance))
                ):
                    continue
                family_count += 1
                history_values = ordered["history_nrmse"].to_numpy(dtype=float)
                comparison_values = ordered[comparison_column].to_numpy(dtype=float)
                if (
                    not np.all(np.isfinite(history_values))
                    or not np.all(np.isfinite(comparison_values))
                    or np.ptp(history_values) == 0.0
                    or np.ptp(comparison_values) == 0.0
                ):
                    undefined = True
                    break
                statistic = float(
                    spearmanr(
                        history_values,
                        comparison_values,
                    ).statistic
                )
                if not np.isfinite(statistic):
                    undefined = True
                    break
                correlations.append(statistic)
            eligible_counts[str(root_id)][str(family)] = family_count
            if family_count < int(min_live_systems_per_family):
                undefined = True
            if undefined:
                break
        output[str(root_id)] = (
            None if undefined or not correlations else float(np.median(correlations))
        )
    return output, eligible_counts


def _bootstrap_median_ci(
    values: Mapping[str, float | None],
    *,
    seed: int,
) -> list[float] | None:
    numeric = np.asarray(
        [float(value) for value in values.values() if value is not None],
        dtype=float,
    )
    if len(numeric) != len(values) or len(numeric) == 0 or not np.all(np.isfinite(numeric)):
        return None
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        len(numeric),
        size=(BOOTSTRAP_REPLICATES, len(numeric)),
    )
    medians = np.median(numeric[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return [float(low), float(high)]


def _success_count(values: Mapping[str, float | None], threshold: float) -> int:
    return int(
        sum(value is not None and float(value) >= float(threshold) for value in values.values())
    )


def _p6_by_root_n(
    history: pd.DataFrame,
    roots: list[str],
    teacher_ns: list[int],
    gammas: list[float],
    windows: list[int],
    teacher_dt: float,
    min_r_squared: float,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Fit the frozen log-history scaling diagnostic on its exact grid."""

    positive_gammas = [float(gamma) for gamma in gammas if float(gamma) > 0.0]
    expected_cells = set(product(positive_gammas, [int(window) for window in windows]))
    selected = history[history["teacher_gamma"] > 0.0]
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for root_id, teacher_n in product(roots, teacher_ns):
        rows = selected[
            (selected["root_id"].astype(str) == str(root_id))
            & (selected["teacher_n_qubits"] == int(teacher_n))
        ].copy()
        observed_cells = [
            (float(row.teacher_gamma), int(row.window))
            for row in rows.itertuples(index=False)
        ]
        result: dict[str, Any] = {
            "slope": None,
            "intercept": None,
            "r_squared": None,
            "n_cells": int(len(rows)),
            "passed": False,
        }
        if (
            len(observed_cells) != len(expected_cells)
            or set(observed_cells) != expected_cells
        ):
            output[(str(root_id), int(teacher_n))] = result
            continue
        error = rows["history_nrmse"].to_numpy(dtype=float)
        if not np.all(np.isfinite(error)) or np.any(error <= 0.0):
            output[(str(root_id), int(teacher_n))] = result
            continue
        x = (
            rows["teacher_gamma"].to_numpy(dtype=float)
            * float(teacher_dt)
            * rows["window"].to_numpy(dtype=float)
        )
        y = np.log(error)
        design = np.column_stack([np.ones(len(x), dtype=float), x])
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if int(rank) != 2:
            output[(str(root_id), int(teacher_n))] = result
            continue
        fitted = design @ coefficients
        total = float(np.sum((y - np.mean(y)) ** 2))
        residual = float(np.sum((y - fitted) ** 2))
        if not np.isfinite(total) or total <= 0.0 or not np.isfinite(residual):
            output[(str(root_id), int(teacher_n))] = result
            continue
        slope = float(coefficients[1])
        intercept = float(coefficients[0])
        r_squared = float(1.0 - residual / total)
        result.update(
            {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "passed": bool(slope < 0.0 and r_squared >= float(min_r_squared)),
            }
        )
        output[(str(root_id), int(teacher_n))] = result
    return output


def _gain(numerator: float, denominator: float) -> float | None:
    if not (
        np.isfinite(float(numerator))
        and np.isfinite(float(denominator))
        and float(denominator) > 0.0
    ):
        return None
    return float(1.0 - float(numerator) / float(denominator))


def _root_g4_gains(
    stream: pd.DataFrame,
    roots: list[str],
    teacher_ns: list[int],
    max_budget: int,
    windows: list[int],
    fixed_gammas: list[float],
) -> dict[str, float | None]:
    min_window, max_window = min(windows), max(windows)
    selected = stream[
        (stream["model"] == "ptd_parallel")
        & (stream["snapshot_budget"] == int(max_budget))
        & (stream["teacher_n_qubits"].isin(teacher_ns))
        & (stream["teacher_gamma"].isin(fixed_gammas))
        & (stream["window"].isin([min_window, max_window]))
    ]
    output: dict[str, float | None] = {}
    for root_id in roots:
        rows = selected[selected["root_id"].astype(str) == str(root_id)]
        gains: list[float] = []
        valid = True
        for teacher_n, gamma in product(teacher_ns, fixed_gammas):
            cell = rows[
                (rows["teacher_n_qubits"] == int(teacher_n))
                & (rows["teacher_gamma"] == float(gamma))
            ]
            small = cell.loc[cell["window"] == min_window, "nrmse"]
            large = cell.loc[cell["window"] == max_window, "nrmse"]
            if len(small) != 1 or len(large) != 1:
                valid = False
                break
            value = _gain(float(large.iloc[0]), float(small.iloc[0]))
            if value is None:
                valid = False
                break
            gains.append(value)
        output[str(root_id)] = float(np.median(gains)) if valid and gains else None
    return output


def _g4_history_transitions(
    suffix: pd.DataFrame,
    roots: list[str],
    teacher_ns: list[int],
    windows: list[int],
    fixed_gammas: list[float],
    *,
    tolerance: float,
    min_variance: float,
) -> dict[tuple[str, int, float], bool]:
    """Verify that every registered G4 contrast crosses the history boundary."""

    min_window, max_window = min(windows), max(windows)
    selected = suffix[
        (suffix["model"] == "rr_same_identity")
        & (suffix["teacher_gamma"].isin(fixed_gammas))
        & (suffix["window"].isin([min_window, max_window]))
    ]
    output: dict[tuple[str, int, float], bool] = {}
    for root_id, teacher_n, gamma in product(roots, teacher_ns, fixed_gammas):
        rows = selected[
            (selected["root_id"].astype(str) == str(root_id))
            & (selected["teacher_n_qubits"] == int(teacher_n))
            & (selected["teacher_gamma"] == float(gamma))
        ]
        passed = False
        if len(rows) == 2 and set(rows["window"].astype(int)) == {
            min_window,
            max_window,
        }:
            indexed = rows.set_index(rows["window"].astype(int))
            short = indexed.loc[min_window]
            long = indexed.loc[max_window]
            values = np.asarray(
                [
                    short["history_nrmse"],
                    long["history_nrmse"],
                    short["target_variance"],
                    long["target_variance"],
                ],
                dtype=float,
            )
            passed = bool(
                np.all(np.isfinite(values))
                and float(short["target_variance"]) >= float(min_variance)
                and float(long["target_variance"]) >= float(min_variance)
                and float(short["history_nrmse"]) > float(tolerance)
                and float(long["history_nrmse"]) <= float(tolerance)
            )
        output[(str(root_id), int(teacher_n), float(gamma))] = passed
    return output


def _g5_gains_by_root_n(
    suffix: pd.DataFrame,
    roots: list[str],
    teacher_ns: list[int],
    min_budget: int,
    max_budget: int,
    max_window: int,
    fixed_gamma: float,
) -> dict[tuple[str, int], float | None]:
    selected = suffix[
        (suffix["model"] == "ptd_parallel")
        & (suffix["teacher_gamma"] == float(fixed_gamma))
        & (suffix["window"] == int(max_window))
        & (suffix["snapshot_budget"].isin([int(min_budget), int(max_budget)]))
    ]
    output: dict[tuple[str, int], float | None] = {}
    for root_id, teacher_n in product(roots, teacher_ns):
        cell = selected[
            (selected["root_id"].astype(str) == str(root_id))
            & (selected["teacher_n_qubits"] == int(teacher_n))
        ]
        small = cell.loc[
            cell["snapshot_budget"] == int(min_budget), "representation_nrmse"
        ]
        large = cell.loc[
            cell["snapshot_budget"] == int(max_budget), "representation_nrmse"
        ]
        output[(str(root_id), int(teacher_n))] = (
            _gain(float(large.iloc[0]), float(small.iloc[0]))
            if len(small) == 1 and len(large) == 1
            else None
        )
    return output


def _fixed_history_feasibility_by_root_n(
    suffix: pd.DataFrame,
    roots: list[str],
    teacher_ns: list[int],
    *,
    max_window: int,
    fixed_gamma: float,
    tolerance: float,
    min_variance: float,
) -> dict[tuple[str, int], bool]:
    """Check that the registered G5 cell is actually past the history bottleneck."""

    selected = suffix[
        (suffix["model"] == "rr_same_identity")
        & (suffix["teacher_gamma"] == float(fixed_gamma))
        & (suffix["window"] == int(max_window))
    ]
    output: dict[tuple[str, int], bool] = {}
    for root_id, teacher_n in product(roots, teacher_ns):
        rows = selected[
            (selected["root_id"].astype(str) == str(root_id))
            & (selected["teacher_n_qubits"] == int(teacher_n))
        ]
        passed = False
        if len(rows) == 1:
            history = float(rows.iloc[0]["history_nrmse"])
            variance = float(rows.iloc[0]["target_variance"])
            passed = bool(
                np.isfinite(history)
                and np.isfinite(variance)
                and variance >= float(min_variance)
                and history <= float(tolerance)
            )
        output[(str(root_id), int(teacher_n))] = passed
    return output


def _mask_gains_by_feasibility(
    gains: Mapping[tuple[str, int], float | None],
    feasibility: Mapping[tuple[str, int], bool],
) -> dict[tuple[str, int], float | None]:
    return {
        key: value if feasibility.get(key) is True else None
        for key, value in gains.items()
    }


def _secondary_gains_by_root_n(
    stream: pd.DataFrame,
    roots: list[str],
    teacher_ns: list[int],
    max_budget: int,
    max_window: int,
    fixed_gamma: float,
) -> dict[tuple[str, int], float | None]:
    selected = stream[
        (stream["teacher_gamma"] == float(fixed_gamma))
        & (stream["window"] == int(max_window))
        & (stream["snapshot_budget"] == int(max_budget))
        & (stream["model"].isin(["ptd_parallel", "classical_tanh"]))
    ]
    output: dict[tuple[str, int], float | None] = {}
    for root_id, teacher_n in product(roots, teacher_ns):
        cell = selected[
            (selected["root_id"].astype(str) == str(root_id))
            & (selected["teacher_n_qubits"] == int(teacher_n))
        ]
        qelm = cell.loc[cell["model"] == "ptd_parallel", "nrmse"]
        classical = cell.loc[cell["model"] == "classical_tanh", "nrmse"]
        output[(str(root_id), int(teacher_n))] = (
            _gain(float(qelm.iloc[0]), float(classical.iloc[0]))
            if len(qelm) == 1 and len(classical) == 1
            else None
        )
    return output


def _best_classical_gains_by_root_n(
    stream: pd.DataFrame,
    roots: list[str],
    teacher_ns: list[int],
    max_budget: int,
    max_window: int,
    fixed_gamma: float,
) -> tuple[
    dict[tuple[str, int], float | None],
    dict[tuple[str, int], str | None],
]:
    """Contrast QELM with the lowest-NRMSE registered classical control."""

    selected = stream[
        (stream["teacher_gamma"] == float(fixed_gamma))
        & (stream["window"] == int(max_window))
        & (stream["model"].isin(
            ["ptd_parallel", "linear_delay", "poly2_delay", "classical_tanh"]
        ))
    ]
    gains: dict[tuple[str, int], float | None] = {}
    winners: dict[tuple[str, int], str | None] = {}
    for root_id, teacher_n in product(roots, teacher_ns):
        cell = selected[
            (selected["root_id"].astype(str) == str(root_id))
            & (selected["teacher_n_qubits"] == int(teacher_n))
        ]
        qelm = cell.loc[
            (cell["model"] == "ptd_parallel")
            & (cell["snapshot_budget"] == int(max_budget)),
            "nrmse",
        ]
        controls: dict[str, float] = {}
        for model in ("linear_delay", "poly2_delay"):
            values = cell.loc[
                (cell["model"] == model) & cell["snapshot_budget"].isna(),
                "nrmse",
            ]
            if len(values) == 1 and np.isfinite(float(values.iloc[0])):
                controls[model] = float(values.iloc[0])
        tanh = cell.loc[
            (cell["model"] == "classical_tanh")
            & (cell["snapshot_budget"] == int(max_budget)),
            "nrmse",
        ]
        if len(tanh) == 1 and np.isfinite(float(tanh.iloc[0])):
            controls["classical_tanh"] = float(tanh.iloc[0])

        key = (str(root_id), int(teacher_n))
        if len(qelm) != 1 or len(controls) != 3:
            gains[key] = None
            winners[key] = None
            continue
        winner = min(controls, key=controls.get)
        gains[key] = _gain(float(qelm.iloc[0]), controls[winner])
        winners[key] = winner if gains[key] is not None else None
    return gains, winners


def _aggregate_root_gains(
    by_root_n: Mapping[tuple[str, int], float | None],
    roots: list[str],
    teacher_ns: list[int],
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for root_id in roots:
        values = [by_root_n.get((str(root_id), int(n))) for n in teacher_ns]
        output[str(root_id)] = (
            float(np.median([float(value) for value in values]))
            if values and all(value is not None for value in values)
            else None
        )
    return output


def _gain_gate(
    root_gains: Mapping[str, float | None],
    *,
    min_gain: float,
    min_root_successes: int,
    min_defined_roots: int | None = None,
) -> dict[str, Any]:
    defined = [float(value) for value in root_gains.values() if value is not None]
    required_defined = (
        len(root_gains) if min_defined_roots is None else int(min_defined_roots)
    )
    defined_enough = bool(len(defined) >= required_defined and defined)
    median = float(np.median(defined)) if defined_enough else None
    positive = int(sum(value is not None and float(value) > 0.0 for value in root_gains.values()))
    passed = bool(
        median is not None
        and median >= float(min_gain)
        and positive >= int(min_root_successes)
    )
    all_defined = len(defined) == len(root_gains) and bool(defined)
    sign_tail = None
    if all_defined and all(value != 0.0 for value in defined):
        n_roots = len(defined)
        sign_tail = float(
            sum(comb(n_roots, k) for k in range(positive, n_roots + 1))
            / (2**n_roots)
        )
    return {
        "passed": passed,
        "median_gain": median,
        "minimum_median_gain": float(min_gain),
        "positive_roots": positive,
        "minimum_positive_roots": int(min_root_successes),
        "defined_roots": int(len(defined)),
        "minimum_defined_roots": int(required_defined),
        "median_gain_bootstrap_95_ci": _bootstrap_median_ci(
            {
                str(index): value
                for index, value in enumerate(defined)
            }
            if defined_enough
            else {},
            seed=BOOTSTRAP_SEED,
        ),
        "positive_root_exact_sign_p_one_sided": sign_tail,
        "root_gains": dict(root_gains),
    }


def evaluate_canonical_gates(
    stream: pd.DataFrame,
    suffix: pd.DataFrame,
    systems: pd.DataFrame,
    profile: Mapping[str, Any],
    *,
    g0_evidence: Mapping[str, bool | None],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate G0--G6 and the best-registered-classical secondary gate.

    Missing rows, undefined correlations, absent G0 evidence, and non-finite
    contrasts all fail closed.  The function never opens or mutates a profile.
    """

    _require_columns(
        stream,
        set(STREAM_REQUIRED_COLUMNS),
        "stream_metrics",
    )
    _require_columns(
        suffix,
        set(SUFFIX_REQUIRED_COLUMNS),
        "suffix_decomposition",
    )
    _require_columns(
        systems,
        set(SYSTEM_REQUIRED_COLUMNS),
        "system_diagnostics",
    )

    roots = [str(root["root_id"]) for root in profile["roots"]]
    teacher_ns = [int(value) for value in profile["teacher_n_qubits"]]
    windows = [int(value) for value in profile["windows"]]
    budgets = [int(value) for value in profile["snapshot_budgets"]]
    decision = _gate_decision(profile)
    tolerance = float(decision["tolerance_nrmse"])
    min_variance = float(decision["min_target_variance"])
    min_root_successes = int(decision["min_root_successes"])
    expected_roots = int(decision["roots_total"])
    min_spearman = float(decision["min_spearman"])
    min_g2_live_systems = int(decision["min_g2_live_systems_per_family"])
    min_window_gain = float(decision["min_window_gain"])
    min_feature_gain = float(decision["min_feature_gain"])
    min_best_classical_gain = float(decision["min_best_classical_gain"])
    min_scaling_r_squared = float(decision["min_scaling_r_squared"])
    g4_gammas = [float(value) for value in decision["g4_gammas"]]
    g5_gamma = float(decision["g5_gamma"])
    coverage = _coverage_report(stream, suffix, systems, profile)

    required_g0 = (
        "tests_passed",
        "strict_json",
        "manifest_hashes_valid",
        "freeze_identity_matches",
        "seed_namespaces_disjoint",
        "duplicate_smoke_deterministic",
    )
    evidence = {key: g0_evidence.get(key) for key in required_g0}
    g0_passed = bool(
        coverage["passed"]
        and len(roots) == int(expected_roots)
        and all(evidence[key] is True for key in required_g0)
    )
    g0 = {
        "passed": g0_passed,
        "coverage": coverage,
        "expected_roots": int(expected_roots),
        "observed_roots": len(roots),
        "evidence": evidence,
    }

    agreement = _bool_values_fail_closed(
        suffix["predictions_agree"], "predictions_agree"
    )
    rms_ok = _bool_values_fail_closed(
        suffix["rms_half_bound_satisfied"], "rms_half_bound_satisfied"
    )
    diameter_ok = _bool_values_fail_closed(
        suffix["diameter_half_bound_satisfied"],
        "diameter_half_bound_satisfied",
    )
    all_bounds = _bool_values_fail_closed(
        suffix["suffix_bounds_satisfied"], "suffix_bounds_satisfied"
    )
    identity = np.abs(suffix["identity_error"].to_numpy(dtype=float))
    recomputed_identity_signed = (
        suffix["total_mse"].to_numpy(dtype=float)
        - suffix["history_mse"].to_numpy(dtype=float)
        - suffix["representation_mse"].to_numpy(dtype=float)
    )
    recomputed_identity = np.abs(recomputed_identity_signed)
    recorded_identity_disagreement = np.abs(
        recomputed_identity_signed - suffix["identity_error"].to_numpy(dtype=float)
    )
    disagreement = suffix["prediction_disagreement"].to_numpy(dtype=float)
    rms_slack = suffix["rms_bound_slack"].to_numpy(dtype=float)
    diameter_slack = suffix["diameter_bound_slack"].to_numpy(dtype=float)
    max_diameter = suffix["max_group_diameter"].to_numpy(dtype=float)
    half_diameter = suffix["half_max_group_diameter"].to_numpy(dtype=float)
    half_diameter_identity = np.abs(2.0 * half_diameter - max_diameter)
    g1_passed = bool(
        np.all(np.isfinite(identity))
        and np.max(identity) <= IDENTITY_ATOL
        and np.all(np.isfinite(recomputed_identity))
        and np.max(recomputed_identity) <= IDENTITY_ATOL
        and np.all(np.isfinite(recorded_identity_disagreement))
        and np.max(recorded_identity_disagreement) <= IDENTITY_ATOL
        and np.all(np.isfinite(disagreement))
        and np.max(disagreement) <= IDENTITY_ATOL
        and np.all(np.isfinite(rms_slack))
        and np.min(rms_slack) >= -IDENTITY_ATOL
        and np.all(np.isfinite(diameter_slack))
        and np.min(diameter_slack) >= -IDENTITY_ATOL
        and np.all(np.isfinite(half_diameter_identity))
        and np.max(half_diameter_identity) <= IDENTITY_ATOL
        and agreement.all()
        and rms_ok.all()
        and diameter_ok.all()
        and all_bounds.all()
    )
    g1 = {
        "passed": g1_passed,
        "identity_tolerance": IDENTITY_ATOL,
        "max_identity_error": float(np.max(identity)),
        "max_recomputed_identity_error": float(np.max(recomputed_identity)),
        "max_recorded_identity_disagreement": float(
            np.max(recorded_identity_disagreement)
        ),
        "max_prediction_disagreement": float(np.max(disagreement)),
        "min_rms_bound_slack": float(np.min(rms_slack)),
        "min_diameter_bound_slack": float(np.min(diameter_slack)),
        "max_half_diameter_identity_error": float(
            np.max(half_diameter_identity)
        ),
    }

    history = _history_frame(suffix)
    stream_rr = stream.loc[
        stream["model"] == "rr_same_identity",
        ["root_id", "system_id", "window", "nrmse"],
    ].rename(columns={"nrmse": "stream_rr_nrmse"})
    if stream_rr.duplicated(["root_id", "system_id", "window"]).any():
        raise ValueError("stream reset-replay rows must be unique per system/window")
    history = history.merge(
        stream_rr,
        on=["root_id", "system_id", "window"],
        how="left",
        validate="one_to_one",
    )
    # G2 and P6 are suffix-panel mechanisms.  Their liveness filter must use
    # the same suffix-panel variance that defines history NRMSE, rather than a
    # separate streaming-test variance from system_diagnostics.csv.
    live_history = history[
        history["target_variance"].to_numpy(dtype=float) >= min_variance
    ]
    g2_values, g2_eligible_counts = _spearman_by_root(
        history,
        windows,
        roots,
        min_variance=min_variance,
        min_live_systems_per_family=min_g2_live_systems,
    )
    g2_successes = _success_count(g2_values, min_spearman)
    g2_defined = {
        str(root_id): float(value)
        for root_id, value in g2_values.items()
        if value is not None
    }
    g2_defined_enough = len(g2_defined) >= min_root_successes
    g2 = {
        "passed": bool(g2_successes >= int(min_root_successes)),
        "comparison": "suffix_history_nrmse_vs_independent_stream_rr_nrmse",
        "minimum_spearman": min_spearman,
        "minimum_live_systems_per_family": min_g2_live_systems,
        "successful_roots": g2_successes,
        "minimum_successful_roots": int(min_root_successes),
        "defined_roots": int(len(g2_defined)),
        "minimum_defined_roots": int(min_root_successes),
        "median_root_median_spearman": (
            float(np.median(list(g2_defined.values())))
            if g2_defined_enough
            else None
        ),
        "median_root_median_spearman_bootstrap_95_ci": _bootstrap_median_ci(
            g2_defined if g2_defined_enough else {},
            seed=BOOTSTRAP_SEED + 1,
        ),
        "root_median_spearman": g2_values,
        "eligible_system_counts_by_root_family": g2_eligible_counts,
    }

    family_success: dict[int, dict[str, bool]] = {n: {} for n in teacher_ns}
    for root_id, teacher_n in product(roots, teacher_ns):
        rows = history[
            (history["root_id"].astype(str) == str(root_id))
            & (history["teacher_n_qubits"] == int(teacher_n))
        ]
        live = rows["target_variance"].to_numpy(dtype=float) >= min_variance
        history_error = rows["history_nrmse"].to_numpy(dtype=float)
        finite = np.isfinite(history_error) & np.isfinite(
            rows["target_variance"].to_numpy(dtype=float)
        )
        feasible = bool(np.any(finite & live & (history_error <= tolerance)))
        impossible = bool(np.any(finite & live & (history_error > tolerance)))
        family_success[int(teacher_n)][str(root_id)] = bool(feasible and impossible)
    g3_by_family = {
        str(teacher_n): {
            "successful_roots": int(sum(values.values())),
            "minimum_successful_roots": int(min_root_successes),
            "passed": bool(sum(values.values()) >= int(min_root_successes)),
            "root_success": values,
        }
        for teacher_n, values in family_success.items()
    }
    g3 = {
        "passed": bool(all(item["passed"] for item in g3_by_family.values())),
        "history_tolerance_nrmse": tolerance,
        "minimum_target_variance": min_variance,
        "families": g3_by_family,
    }

    raw_g4_values = _root_g4_gains(
        stream,
        roots,
        teacher_ns,
        max(budgets),
        windows,
        g4_gammas,
    )
    g4_transition_cells = _g4_history_transitions(
        suffix,
        roots,
        teacher_ns,
        windows,
        g4_gammas,
        tolerance=tolerance,
        min_variance=min_variance,
    )
    g4_root_transition = {
        str(root_id): bool(
            all(
                g4_transition_cells.get(
                    (str(root_id), int(teacher_n), float(gamma)),
                    False,
                )
                for teacher_n, gamma in product(teacher_ns, g4_gammas)
            )
        )
        for root_id in roots
    }
    g4_values = {
        str(root_id): (
            raw_g4_values.get(str(root_id))
            if g4_root_transition[str(root_id)]
            else None
        )
        for root_id in roots
    }
    g4 = _gain_gate(
        g4_values,
        min_gain=min_window_gain,
        min_root_successes=min_root_successes,
        min_defined_roots=min_root_successes,
    )
    g4["fixed_teacher_gammas"] = g4_gammas
    g4["window_contrast"] = [min(windows), max(windows)]
    g4["snapshot_budget"] = max(budgets)
    g4["history_transition_root_success"] = g4_root_transition
    g4["history_transition_cells"] = {
        f"{root_id}|N={teacher_n}|gamma={gamma:g}": bool(value)
        for (root_id, teacher_n, gamma), value in g4_transition_cells.items()
    }

    fixed_feasibility = _fixed_history_feasibility_by_root_n(
        suffix,
        roots,
        teacher_ns,
        max_window=max(windows),
        fixed_gamma=g5_gamma,
        tolerance=tolerance,
        min_variance=min_variance,
    )
    raw_g5_by_root_n = _g5_gains_by_root_n(
        suffix,
        roots,
        teacher_ns,
        min(budgets),
        max(budgets),
        max(windows),
        g5_gamma,
    )
    g5_by_root_n = _mask_gains_by_feasibility(
        raw_g5_by_root_n,
        fixed_feasibility,
    )
    g5_values = _aggregate_root_gains(g5_by_root_n, roots, teacher_ns)
    g5 = _gain_gate(
        g5_values,
        min_gain=min_feature_gain,
        min_root_successes=min_root_successes,
        min_defined_roots=min_root_successes,
    )
    g5["fixed_teacher_gamma"] = g5_gamma
    g5["fixed_window"] = max(windows)
    g5["snapshot_budget_contrast"] = [min(budgets), max(budgets)]
    g5["history_feasible_root_family_cells"] = int(sum(fixed_feasibility.values()))
    g5["expected_root_family_cells"] = int(len(fixed_feasibility))
    g5["history_feasibility_by_root_family"] = {
        f"{root_id}|N={teacher_n}": bool(value)
        for (root_id, teacher_n), value in fixed_feasibility.items()
    }

    g6_families: dict[str, Any] = {}
    for teacher_n in teacher_ns:
        n_spearman, n_eligible_counts = _spearman_by_root(
            history,
            windows,
            roots,
            teacher_n=teacher_n,
            min_variance=min_variance,
            min_live_systems_per_family=min_g2_live_systems,
        )
        n_g2_successes = _success_count(n_spearman, min_spearman)
        n_defined_spearman = {
            str(root_id): float(value)
            for root_id, value in n_spearman.items()
            if value is not None
        }
        n_defined_enough = len(n_defined_spearman) >= min_root_successes
        n_gains = {
            root_id: g5_by_root_n.get((str(root_id), int(teacher_n)))
            for root_id in roots
        }
        n_g5 = _gain_gate(
            n_gains,
            min_gain=min_feature_gain,
            min_root_successes=min_root_successes,
            min_defined_roots=min_root_successes,
        )
        g6_families[str(teacher_n)] = {
            "passed": bool(
                n_g2_successes >= int(min_root_successes) and n_g5["passed"]
            ),
            "g2_successful_roots": n_g2_successes,
            "g2_minimum_spearman": min_spearman,
            "g2_minimum_successful_roots": min_root_successes,
            "g2_defined_roots": int(len(n_defined_spearman)),
            "g2_minimum_defined_roots": int(min_root_successes),
            "g2_median_root_median_spearman": (
                float(np.median(list(n_defined_spearman.values())))
                if n_defined_enough
                else None
            ),
            "g2_root_median_spearman": n_spearman,
            "g2_eligible_system_counts_by_root_family": n_eligible_counts,
            "g2_median_root_median_spearman_bootstrap_95_ci": _bootstrap_median_ci(
                n_defined_spearman if n_defined_enough else {},
                seed=BOOTSTRAP_SEED + 10 + int(teacher_n),
            ),
            "g5": n_g5,
        }
    g6 = {
        "passed": bool(all(item["passed"] for item in g6_families.values())),
        "families": g6_families,
    }

    tanh_by_root_n = _secondary_gains_by_root_n(
        stream,
        roots,
        teacher_ns,
        max(budgets),
        max(windows),
        g5_gamma,
    )
    tanh_by_root_n = _mask_gains_by_feasibility(
        tanh_by_root_n,
        fixed_feasibility,
    )
    tanh_values = _aggregate_root_gains(
        tanh_by_root_n,
        roots,
        teacher_ns,
    )
    tanh_descriptive = _gain_gate(
        tanh_values,
        min_gain=min_best_classical_gain,
        min_root_successes=min_root_successes,
        min_defined_roots=min_root_successes,
    )
    tanh_descriptive["decision_bearing"] = False
    tanh_descriptive["fixed_teacher_gamma"] = g5_gamma
    tanh_descriptive["fixed_window"] = max(windows)
    tanh_descriptive["snapshot_budget"] = max(budgets)

    best_classical_by_root_n, best_classical_winner = (
        _best_classical_gains_by_root_n(
            stream,
            roots,
            teacher_ns,
            max(budgets),
            max(windows),
            g5_gamma,
        )
    )
    best_classical_by_root_n = _mask_gains_by_feasibility(
        best_classical_by_root_n,
        fixed_feasibility,
    )
    best_classical_winner = {
        key: value if fixed_feasibility.get(key) is True else None
        for key, value in best_classical_winner.items()
    }
    best_classical_values = _aggregate_root_gains(
        best_classical_by_root_n,
        roots,
        teacher_ns,
    )
    secondary = _gain_gate(
        best_classical_values,
        min_gain=min_best_classical_gain,
        min_root_successes=min_root_successes,
        min_defined_roots=min_root_successes,
    )
    secondary["decision_bearing"] = True
    secondary["registered_classical_controls"] = [
        "linear_delay",
        "poly2_delay",
        "classical_tanh",
    ]
    secondary["fixed_teacher_gamma"] = g5_gamma
    secondary["fixed_window"] = max(windows)
    secondary["snapshot_budget"] = max(budgets)

    p6_by_root_n = _p6_by_root_n(
        live_history,
        roots,
        teacher_ns,
        [float(value) for value in profile["teacher_gammas"]],
        windows,
        float(profile["teacher"]["dt"]),
        min_scaling_r_squared,
    )
    p6_families: dict[str, Any] = {}
    for teacher_n in teacher_ns:
        root_fits = {
            root_id: p6_by_root_n[(str(root_id), int(teacher_n))]
            for root_id in roots
        }
        successful = int(sum(bool(item["passed"]) for item in root_fits.values()))
        defined_slopes = [
            float(item["slope"])
            for item in root_fits.values()
            if item["slope"] is not None
        ]
        negative_slopes = int(sum(value < 0.0 for value in defined_slopes))
        negative_sign_p = None
        if len(defined_slopes) == len(root_fits) and all(
            value != 0.0 for value in defined_slopes
        ):
            negative_sign_p = float(
                sum(
                    comb(len(root_fits), count)
                    for count in range(negative_slopes, len(root_fits) + 1)
                )
                / (2 ** len(root_fits))
            )
        slopes = {
            root_id: item["slope"]
            for root_id, item in root_fits.items()
        }
        defined_slope_map = {
            root_id: float(value)
            for root_id, value in slopes.items()
            if value is not None
        }
        enough_defined_slopes = len(defined_slope_map) >= min_root_successes
        p6_families[str(teacher_n)] = {
            "passed": bool(successful >= min_root_successes),
            "successful_roots": successful,
            "minimum_successful_roots": min_root_successes,
            "defined_roots": len(defined_slope_map),
            "minimum_defined_roots": min_root_successes,
            "negative_slope_roots": negative_slopes,
            "negative_slope_exact_sign_p_one_sided": negative_sign_p,
            "median_slope": (
                float(np.median(list(defined_slope_map.values())))
                if enough_defined_slopes
                else None
            ),
            "median_slope_bootstrap_95_ci": _bootstrap_median_ci(
                defined_slope_map if enough_defined_slopes else {},
                seed=BOOTSTRAP_SEED + 20 + int(teacher_n),
            ),
            "root_fits": root_fits,
        }
    p6 = {
        "decision_bearing": False,
        "passed": bool(all(item["passed"] for item in p6_families.values())),
        "replicated_in_both_teacher_families": bool(
            all(item["passed"] for item in p6_families.values())
        ),
        "minimum_r_squared": min_scaling_r_squared,
        "slope_direction": "negative",
        "predictor": "teacher_gamma * teacher_dt * window",
        "teacher_dt": float(profile["teacher"]["dt"]),
        "gamma_filter": "teacher_gamma > 0",
        "families": p6_families,
    }

    primary = {"G0": g0, "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6}
    if not bool(profile.get("confirmatory", False)):
        decision_label = "CALIBRATION_ONLY"
    elif all(primary[name]["passed"] for name in primary):
        decision_label = "CONFIRMS_FINITE_RESOURCE_FRONTIER"
        if secondary["passed"]:
            decision_label += "_QELM_SPECIFIC"
    elif not all(primary[name]["passed"] for name in ("G0", "G1", "G2", "G3")):
        decision_label = "NO_GO"
    elif primary["G4"]["passed"] and primary["G5"]["passed"] and not primary["G6"]["passed"]:
        decision_label = "UNRESOLVED_FAMILY_REPLICATION"
    else:
        decision_label = "SUPPORTS_HISTORY_ONLY"

    contrasts: list[dict[str, Any]] = []
    for root_id, value in g2_values.items():
        contrasts.append(
            {
                "root_id": root_id,
                "gate_id": "G2",
                "teacher_n_qubits": None,
                "contrast": "median_suffix_history_vs_stream_rr_spearman",
                "value": value,
                "threshold": min_spearman,
                "passed": bool(value is not None and value >= min_spearman),
            }
        )
    for root_id, value in g4_values.items():
        contrasts.append(
            {
                "root_id": root_id,
                "gate_id": "G4",
                "teacher_n_qubits": None,
                "contrast": "Bmax_d1_to_dmax_nrmse_gain",
                "value": value,
                "threshold": 0.0,
                "passed": bool(value is not None and value > 0.0),
            }
        )
    for (root_id, teacher_n), value in g5_by_root_n.items():
        contrasts.append(
            {
                "root_id": root_id,
                "gate_id": "G5",
                "teacher_n_qubits": teacher_n,
                "contrast": "Bmin_to_Bmax_representation_nrmse_gain",
                "value": value,
                "threshold": 0.0,
                "passed": bool(value is not None and value > 0.0),
            }
        )
    for (root_id, teacher_n), value in tanh_by_root_n.items():
        contrasts.append(
            {
                "root_id": root_id,
                "gate_id": "SECONDARY_TANH_DESCRIPTIVE",
                "teacher_n_qubits": teacher_n,
                "contrast": "qelm_gain_over_classical_tanh",
                "value": value,
                "threshold": 0.0,
                "passed": bool(value is not None and value > 0.0),
            }
        )
    for (root_id, teacher_n), value in best_classical_by_root_n.items():
        contrasts.append(
            {
                "root_id": root_id,
                "gate_id": "SECONDARY_BEST_CLASSICAL",
                "teacher_n_qubits": teacher_n,
                "contrast": "qelm_gain_over_best_registered_classical",
                "value": value,
                "threshold": 0.0,
                "passed": bool(value is not None and value > 0.0),
                "best_classical_model": best_classical_winner[
                    (str(root_id), int(teacher_n))
                ],
            }
        )
    for (root_id, teacher_n), fit in p6_by_root_n.items():
        contrasts.append(
            {
                "root_id": root_id,
                "gate_id": "P6",
                "teacher_n_qubits": teacher_n,
                "contrast": "log_history_nrmse_vs_gamma_dt_window",
                "value": fit["slope"],
                "threshold": 0.0,
                "r_squared": fit["r_squared"],
                "minimum_r_squared": min_scaling_r_squared,
                "passed": bool(fit["passed"]),
            }
        )

    report = {
        "schema_version": 1,
        "study": str(profile["study"]),
        "profile": str(profile["profile"]),
        "confirmatory": bool(profile.get("confirmatory", False)),
        "statistical_unit": "root_block",
        "expected_roots": int(expected_roots),
        "minimum_successful_roots": int(min_root_successes),
        "decision": decision_label,
        "frozen_decision_config": decision,
        "primary_gates": primary,
        "secondary_qelm_vs_best_classical": secondary,
        "descriptive_qelm_vs_tanh": tanh_descriptive,
        "secondary_p6_forgetting_scaling": p6,
    }
    return report, pd.DataFrame(contrasts)


__all__ = [
    "evaluate_canonical_gates",
    "load_canonical_artifacts",
    "load_strict_json",
    "verify_canonical_manifest",
    "verify_duplicate_smoke_runs",
    "verify_pytest_junit",
]
