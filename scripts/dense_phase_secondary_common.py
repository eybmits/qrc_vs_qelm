"""Frozen-material and seed audits for the dense secondary phase map."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "canonical_emulation_dense_secondary.yaml"
FREEZE_PATH = ROOT / "configs" / "canonical_emulation_dense_secondary_freeze.json"
PROTOCOL_PATH = ROOT / "DENSE_PHASE_SECONDARY_PROTOCOL.md"
RUNNER_PATH = ROOT / "scripts" / "run_dense_phase_secondary.py"
ANALYSIS_PATH = ROOT / "scripts" / "analyze_dense_phase_secondary.py"
COMMON_PATH = Path(__file__).resolve()
PROFILE = "dense_secondary"
STUDY = "canonical_qrc_to_parallel_qelm_dense_secondary"
EXPECTED_GAMMAS = (0.0, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
EXPECTED_WINDOWS = (1, 2, 3, 4, 5, 6)
EXPECTED_BUDGETS = (18, 36, 72)
EXPECTED_ROOTS = tuple(f"dense{index:02d}" for index in range(1, 13))


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


def dense_roots(config_path: Path = CONFIG_PATH) -> list[dict[str, Any]]:
    raw = load_yaml(config_path)
    roots = raw.get("profiles", {}).get(PROFILE, {}).get("roots")
    if not isinstance(roots, list) or not all(isinstance(root, dict) for root in roots):
        raise ValueError("dense profile roots must be a list of mappings")
    return roots


def root_set_sha256(config_path: Path = CONFIG_PATH) -> str:
    return sha256_json(dense_roots(config_path))


def dense_seed_map(config_path: Path = CONFIG_PATH) -> dict[int, str]:
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
    for root in dense_roots(config_path):
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
                raise ValueError(f"dense seed collision: {label} and {used[value]}")
            used[value] = label
    return used


def _large_integers(value: Any) -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for item in value.values():
            found.update(_large_integers(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_large_integers(item))
    elif isinstance(value, int) and not isinstance(value, bool) and value >= 100_000:
        found.add(value)
    return found


def assert_seed_disjointness(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    dense = dense_seed_map(config_path)
    checked: list[str] = []
    conflicts: list[dict[str, Any]] = []
    for path in sorted(ROOT.rglob("*.yaml")):
        if path.resolve() == Path(config_path).resolve() or ".git" in path.parts:
            continue
        try:
            values = _large_integers(load_yaml(path))
        except (OSError, ValueError, yaml.YAMLError):
            continue
        checked.append(path.relative_to(ROOT).as_posix())
        for seed in sorted(set(dense).intersection(values)):
            conflicts.append(
                {"seed": seed, "dense_label": dense[seed], "reference": checked[-1]}
            )
    if conflicts:
        raise ValueError(f"dense seeds collide with existing configuration seeds: {conflicts}")
    return {
        "passed": True,
        "dense_seed_count": len(dense),
        "reference_files": checked,
        "root_set_sha256": root_set_sha256(config_path),
    }


def dense_material_files(config_path: Path = CONFIG_PATH) -> list[Path]:
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
        raise FileNotFoundError(f"dense material files are missing: {missing}")
    return paths


def dense_material_manifest(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    files = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in dense_material_files(config_path)
    }
    return {"files": files, "identity_sha256": sha256_json(files)}


def validate_dense_configuration(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    raw = load_yaml(config_path)
    common = raw.get("common", {})
    profile = raw.get("profiles", {}).get(PROFILE, {})
    if raw.get("schema_version") != 1 or raw.get("study") != STUDY:
        raise ValueError("unexpected dense study identity")
    if raw.get("status") != "PROSPECTIVE_SECONDARY_REPLICATION":
        raise ValueError("dense study status is not prospective secondary replication")
    if profile.get("confirmatory") is not False:
        raise ValueError("dense study must remain separate from the original confirmation")
    gammas = tuple(float(value) for value in common.get("teacher", {}).get("gammas", []))
    windows = tuple(int(value) for value in common.get("student", {}).get("windows", []))
    snapshots = tuple(int(value) for value in common.get("student", {}).get("snapshot_budgets", []))
    roots = tuple(str(root.get("root_id")) for root in dense_roots(config_path))
    if gammas != EXPECTED_GAMMAS or windows != EXPECTED_WINDOWS:
        raise ValueError("dense gamma/window grid has drifted")
    if snapshots != (1, 2, 4) or int(common.get("student", {}).get("n_qubits", 0)) != 6:
        raise ValueError("dense QELM architecture or budgets have drifted")
    if roots != EXPECTED_ROOTS:
        raise ValueError("dense root ordering has drifted")
    if int(profile.get("train_rows", 0)) != 480 or int(profile.get("val_rows", 0)) != 160:
        raise ValueError("dense train/validation sizes have drifted")
    if int(profile.get("test_rows", 0)) != 320 or int(profile.get("suffix_groups", 0)) != 32:
        raise ValueError("dense test/suffix sizes have drifted")
    return {
        "passed": True,
        "history_root_cells": 12 * 2 * len(gammas) * len(windows),
        "ptd_root_budget_cells": 12 * 2 * len(gammas) * len(windows) * 3,
        "expected_science_rows_per_table": 12 * 2 * len(gammas) * len(windows) * 10,
        "root_set_sha256": root_set_sha256(config_path),
    }


def verify_dense_freeze(
    config_path: Path = CONFIG_PATH,
    freeze_path: Path = FREEZE_PATH,
) -> dict[str, Any]:
    record = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    material = dense_material_manifest(config_path)
    try:
        timestamp = datetime.fromisoformat(str(record.get("frozen_at_utc")).replace("Z", "+00:00"))
        utc_timestamp = timestamp.tzinfo is not None and timestamp.utcoffset() == timezone.utc.utcoffset(timestamp)
    except (TypeError, ValueError):
        utc_timestamp = False
    checks = {
        "schema_version": record.get("schema_version") == 1,
        "status": record.get("status") == "FROZEN_PROSPECTIVE_SECONDARY",
        "study": record.get("study") == STUDY,
        "profile": record.get("profile") == PROFILE,
        "config_sha256": record.get("config_sha256") == sha256_file(config_path),
        "material_identity_sha256": record.get("material_identity_sha256") == material["identity_sha256"],
        "material_files": record.get("material_files") == material["files"],
        "root_ids": tuple(record.get("root_ids", [])) == EXPECTED_ROOTS,
        "root_set_sha256": record.get("root_set_sha256") == root_set_sha256(config_path),
        "secondary_roots_previously_opened": record.get("secondary_roots_previously_opened") is False,
        "frozen_at_utc": utc_timestamp,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "record_sha256": sha256_file(freeze_path),
        "material_identity_sha256": material["identity_sha256"],
        "root_set_sha256": root_set_sha256(config_path),
    }
