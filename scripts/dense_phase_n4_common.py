"""Frozen-material, configuration, and prior-seed audits for the N4 phase map."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "canonical_emulation_dense_n4.yaml"
FREEZE_PATH = ROOT / "configs" / "canonical_emulation_dense_n4_freeze.json"
PROTOCOL_PATH = ROOT / "DENSE_PHASE_N4_PROTOCOL.md"
RUNNER_PATH = ROOT / "scripts" / "run_dense_phase_n4.py"
ANALYSIS_PATH = ROOT / "scripts" / "analyze_dense_phase_n4.py"
COMMON_PATH = Path(__file__).resolve()
PROFILE = "dense_n4"
STUDY = "canonical_qrc_to_parallel_qelm_dense_n4"
EXPECTED_GAMMAS = (0.0, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
EXPECTED_WINDOWS = (1, 2, 3, 4, 5, 6)
EXPECTED_BUDGETS = (18, 36, 72)
EXPECTED_ROOTS = tuple(f"dense_n4_{index:02d}" for index in range(1, 13))
SEED_SUFFIXES = {".yaml", ".yml", ".json", ".csv"}
N4_SEED_BAND = range(94_000_000, 95_000_000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def dense_n4_roots(config_path: Path = CONFIG_PATH) -> list[dict[str, Any]]:
    raw = load_yaml(config_path)
    roots = raw.get("profiles", {}).get(PROFILE, {}).get("roots")
    if not isinstance(roots, list) or not all(isinstance(root, dict) for root in roots):
        raise ValueError("dense_n4 profile roots must be a list of mappings")
    return roots


def root_set_sha256(config_path: Path = CONFIG_PATH) -> str:
    return sha256_json(dense_n4_roots(config_path))


def dense_n4_seed_map(config_path: Path = CONFIG_PATH) -> dict[int, str]:
    used: dict[int, str] = {}
    scalar_fields = (
        "student_coupling_seed",
        "train_input_seed",
        "val_input_seed",
        "test_input_seed",
        "suffix_input_seed",
        "prefix_input_seed",
        "classical_feature_seed",
    )
    for root_index, root in enumerate(dense_n4_roots(config_path), start=1):
        root_id = str(root.get("root_id"))
        teacher = root.get("teacher_coupling_seeds")
        if not isinstance(teacher, dict):
            raise ValueError(f"{root_id}: invalid teacher seed map")
        items = [(f"{root_id}.teacher[{key}]", value) for key, value in teacher.items()]
        items.extend((f"{root_id}.{field}", root.get(field)) for field in scalar_fields)
        for label, value in items:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label}: seed must be a non-negative integer")
            if value in used:
                raise ValueError(f"N4 seed collision: {label} and {used[value]}")
            used[value] = label
        exact = {
            int(teacher["4"]): 94_010_000 + root_index,
            int(root["student_coupling_seed"]): 94_020_000 + root_index,
            int(root["train_input_seed"]): 94_030_000 + root_index,
            int(root["val_input_seed"]): 94_040_000 + root_index,
            int(root["test_input_seed"]): 94_050_000 + root_index,
            int(root["suffix_input_seed"]): 94_060_000 + root_index,
            int(root["prefix_input_seed"]): 94_070_000 + root_index,
            int(root["classical_feature_seed"]): 94_080_000 + root_index,
        }
        if any(observed != expected for observed, expected in exact.items()):
            raise ValueError(f"{root_id}: seed-role namespace has drifted")
    if len(used) != 96 or any(seed not in N4_SEED_BAND for seed in used):
        raise ValueError("N4 seeds must be 96 unique values in the reserved 94m band")
    return used


def _is_seed_key(key: str) -> bool:
    normalized = str(key).strip().lower()
    return normalized in {
        "seed",
        "seeds",
        "input_seeds_kept",
        "coupling_seeds_kept",
    } or normalized.endswith(("_seed", "_seeds"))


def _collect_structured_seeds(
    value: Any,
    source: str,
    path: tuple[str, ...] = (),
    inside_seed_field: bool = False,
) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            found.extend(
                _collect_structured_seeds(
                    item,
                    source,
                    (*path, key_text),
                    inside_seed_field or _is_seed_key(key_text),
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(
                _collect_structured_seeds(
                    item,
                    source,
                    (*path, str(index)),
                    inside_seed_field,
                )
            )
    elif (
        inside_seed_field
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        found.append((value, f"{source}:{'.'.join(path)}"))
    return found


def _csv_structured_seeds(path: Path, source: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        seed_columns = [name for name in (reader.fieldnames or []) if _is_seed_key(name)]
        if not seed_columns:
            return found
        for row_number, row in enumerate(reader, start=2):
            for column in seed_columns:
                raw = str(row.get(column, "")).strip()
                if not raw:
                    continue
                for index, token in enumerate(raw.split("|")):
                    try:
                        numeric = float(token.strip())
                    except ValueError as error:
                        raise ValueError(
                            f"structured seed is not numeric: "
                            f"{source}:{row_number}:{column}[{index}]"
                        ) from error
                    if not numeric.is_integer() or numeric < 0:
                        raise ValueError(
                            f"structured seed is not a non-negative integer: "
                            f"{source}:{row_number}:{column}[{index}]"
                        )
                    found.append(
                        (int(numeric), f"{source}:{row_number}:{column}[{index}]")
                    )
    return found


def _automatic_prior_seed_files(config_path: Path) -> list[Path]:
    excluded = {Path(config_path).resolve(), FREEZE_PATH.resolve()}
    candidates: set[Path] = set()
    for folder in (ROOT / "configs", ROOT / "data" / "frozen", ROOT / "data" / "reference", ROOT / "checkpoints"):
        if folder.is_dir():
            candidates.update(
                path.resolve()
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in SEED_SUFFIXES
            )
    candidates.update(
        path.resolve()
        for path in (ROOT / "pra_extension").glob("config*.yaml")
        if path.is_file()
    )
    for folder_name in (
        "results_canonical_confirmatory_v1",
        "results_dense_phase_secondary_v1",
    ):
        folder = ROOT / folder_name
        if folder.is_dir():
            candidates.update(
                path.resolve()
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in SEED_SUFFIXES
            )
    release_path = ROOT / "pra_extension" / "PRA_EXTENSION_RELEASE.json"
    if release_path.is_file():
        release = json.loads(release_path.read_text(encoding="utf-8"))
        artifacts = release.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError("release manifest artifacts must be a mapping")
        for relative in artifacts:
            path = (ROOT / str(relative)).resolve()
            if path.is_file() and path.suffix.lower() in SEED_SUFFIXES:
                candidates.add(path)
    return sorted(path for path in candidates if path not in excluded)


def prior_seed_registry(
    config_path: Path = CONFIG_PATH,
    reference_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    if reference_files is None:
        paths = _automatic_prior_seed_files(config_path)
    else:
        paths = [(ROOT / relative).resolve() for relative in reference_files]
    entries: dict[str, list[str]] = {}
    source_files: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"prior seed-registry source is missing: {path}")
        relative = path.relative_to(ROOT).as_posix()
        try:
            if path.suffix.lower() == ".csv":
                found = _csv_structured_seeds(path, relative)
            else:
                if path.suffix.lower() in {".yaml", ".yml"}:
                    value = yaml.safe_load(path.read_text(encoding="utf-8"))
                else:
                    value = json.loads(path.read_text(encoding="utf-8"))
                found = _collect_structured_seeds(value, relative)
        except (UnicodeDecodeError, csv.Error, json.JSONDecodeError, yaml.YAMLError) as error:
            raise ValueError(f"cannot parse structured seed source: {relative}") from error
        if not found:
            continue
        source_files[relative] = sha256_file(path)
        for seed, label in found:
            entries.setdefault(str(seed), []).append(label)
    normalized = {key: sorted(set(labels)) for key, labels in sorted(entries.items())}
    identity = sha256_json({"entries": normalized, "source_files": source_files})
    return {
        "entries": normalized,
        "source_files": source_files,
        "identity_sha256": identity,
        "seed_count": len(normalized),
    }


def assert_seed_disjointness(
    config_path: Path = CONFIG_PATH,
    reference_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    dense = dense_n4_seed_map(config_path)
    registry = prior_seed_registry(config_path, reference_files)
    conflicts = [
        {
            "seed": seed,
            "n4_label": dense[seed],
            "prior_locations": registry["entries"][str(seed)],
        }
        for seed in sorted(dense)
        if str(seed) in registry["entries"]
    ]
    if conflicts:
        raise ValueError(f"N4 seeds collide with prior structured seed fields: {conflicts}")
    reserved_prior = sorted(
        int(seed) for seed in registry["entries"] if int(seed) in N4_SEED_BAND
    )
    if reserved_prior:
        raise ValueError(
            f"prior structured seeds occupy the reserved N4 seed band: {reserved_prior}"
        )
    return {
        "passed": True,
        "n4_seed_count": len(dense),
        "prior_seed_count": registry["seed_count"],
        "prior_seed_registry_sha256": registry["identity_sha256"],
        "prior_seed_source_files": registry["source_files"],
        "root_set_sha256": root_set_sha256(config_path),
    }


def dense_n4_material_files(config_path: Path = CONFIG_PATH) -> list[Path]:
    paths = [
        Path(config_path).resolve(),
        PROTOCOL_PATH.resolve(),
        RUNNER_PATH.resolve(),
        ANALYSIS_PATH.resolve(),
        COMMON_PATH.resolve(),
        (ROOT / "pyproject.toml").resolve(),
    ]
    paths.extend(sorted((ROOT / "src" / "qrc_vs_qelm").rglob("*.py")))
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"N4 material files are missing: {missing}")
    return paths


def dense_n4_material_manifest(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    files = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in dense_n4_material_files(config_path)
    }
    return {"files": files, "identity_sha256": sha256_json(files)}


def validate_dense_n4_configuration(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    raw = load_yaml(config_path)
    common = raw.get("common", {})
    teacher = common.get("teacher", {})
    student = common.get("student", {})
    decision = common.get("decision", {})
    profile = raw.get("profiles", {}).get(PROFILE, {})
    if raw.get("schema_version") != 1 or raw.get("study") != STUDY:
        raise ValueError("unexpected N4 study identity")
    if raw.get("status") != "PROSPECTIVE_SIZE_EXTRAPOLATION":
        raise ValueError("N4 study status is not prospective size extrapolation")
    if raw.get("freeze_record") != FREEZE_PATH.name:
        raise ValueError("N4 freeze-record pointer has drifted")
    if profile.get("confirmatory") is not False:
        raise ValueError("N4 study must remain separate from the original confirmation")
    gammas = tuple(float(value) for value in teacher.get("gammas", []))
    windows = tuple(int(value) for value in student.get("windows", []))
    snapshots = tuple(int(value) for value in student.get("snapshot_budgets", []))
    roots = tuple(str(root.get("root_id")) for root in dense_n4_roots(config_path))
    if tuple(int(value) for value in teacher.get("n_qubits", [])) != (4,):
        raise ValueError("N4 teacher-size scope has drifted")
    if tuple(teacher.get("dissipators", [])) != ("local_loss",):
        raise ValueError("N4 dissipator scope has drifted")
    if gammas != EXPECTED_GAMMAS or windows != EXPECTED_WINDOWS:
        raise ValueError("N4 gamma/window grid has drifted")
    if (
        student.get("architecture") != "ptd_parallel"
        or student.get("encoding") != "parallel"
        or int(student.get("n_qubits", 0)) != 6
        or snapshots != (1, 2, 4)
    ):
        raise ValueError("N4 QELM architecture or budgets have drifted")
    if roots != EXPECTED_ROOTS:
        raise ValueError("N4 root ordering has drifted")
    exact_profile = {
        "train_rows": 480,
        "val_rows": 160,
        "test_rows": 320,
        "suffix_groups": 32,
        "prefix_branches": 4,
        "prefix_burnin": 128,
    }
    if any(int(profile.get(key, -1)) != value for key, value in exact_profile.items()):
        raise ValueError("N4 data-panel sizes have drifted")
    if (
        float(decision.get("tolerance_nrmse", -1.0)) != 0.10
        or int(decision.get("min_root_successes", 0)) != 10
        or int(decision.get("roots_total", 0)) != 12
        or float(decision.get("min_scaling_r_squared", -1.0)) != 0.85
    ):
        raise ValueError("N4 decision thresholds have drifted")
    return {
        "passed": True,
        "systems": 132,
        "target_live_systems": 132,
        "stream_rows": 7_920,
        "suffix_rows": 7_920,
        "aggregate_cells": 66,
        "history_root_cells": 792,
        "ptd_root_budget_cells": 2_376,
        "scaling_root_fits": 12,
        "positive_loss_cells_per_fit": 60,
        "root_set_sha256": root_set_sha256(config_path),
    }


def verify_dense_n4_freeze(
    config_path: Path = CONFIG_PATH,
    freeze_path: Path = FREEZE_PATH,
) -> dict[str, Any]:
    record = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    material = dense_n4_material_manifest(config_path)
    registry_files = record.get("prior_seed_source_files", {})
    if not isinstance(registry_files, dict):
        registry_files = {}
    registry = prior_seed_registry(config_path, registry_files.keys())
    try:
        timestamp = datetime.fromisoformat(str(record.get("frozen_at_utc")).replace("Z", "+00:00"))
        utc_timestamp = (
            timestamp.tzinfo is not None
            and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp)
        )
    except (TypeError, ValueError):
        utc_timestamp = False
    checks = {
        "schema_version": record.get("schema_version") == 1,
        "status": record.get("status") == "FROZEN_PROSPECTIVE_N4",
        "study": record.get("study") == STUDY,
        "profile": record.get("profile") == PROFILE,
        "config_sha256": record.get("config_sha256") == sha256_file(config_path),
        "material_identity_sha256": record.get("material_identity_sha256") == material["identity_sha256"],
        "material_files": record.get("material_files") == material["files"],
        "root_ids": tuple(record.get("root_ids", [])) == EXPECTED_ROOTS,
        "root_set_sha256": record.get("root_set_sha256") == root_set_sha256(config_path),
        "prior_seed_registry_sha256": record.get("prior_seed_registry_sha256") == registry["identity_sha256"],
        "prior_seed_source_files": registry_files == registry["source_files"],
        "design_informed_by_open_n2_n3_dense_results": record.get(
            "design_informed_by_open_n2_n3_dense_results"
        )
        is True,
        "n4_replication_results_previously_opened": record.get(
            "n4_replication_results_previously_opened"
        )
        is False,
        "frozen_at_utc": utc_timestamp,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "record_sha256": sha256_file(freeze_path),
        "material_identity_sha256": material["identity_sha256"],
        "prior_seed_registry_sha256": registry["identity_sha256"],
        "root_set_sha256": root_set_sha256(config_path),
    }
