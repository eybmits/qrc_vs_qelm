import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess

import pytest

from qrc_vs_qelm.canonical_study import run_canonical_study
from qrc_vs_qelm.provenance import (
    environment_manifest,
    git_worktree_manifest,
    material_file_manifest,
    verify_canonical_freeze_record,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "canonical_emulation.yaml"
RUNNER = ROOT / "scripts" / "run_canonical_emulation.py"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def test_smoke_run_embeds_verifiable_reproducibility_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "canonical_smoke"
    summary = run_canonical_study(
        config_path=CONFIG,
        profile_name="smoke",
        output_dir=output,
        runner_path=RUNNER,
    )
    manifest = json.loads(
        (output / "canonical_manifest.json").read_text(encoding="utf-8")
    )
    reproducibility = manifest["reproducibility"]

    assert summary["profile"] == "smoke"
    assert reproducibility["capture_phase"] == "before_simulation"
    assert SHA256.fullmatch(reproducibility["identity_sha256"])
    assert manifest["config_sha256"] == reproducibility["material"]["files"][
        "configuration"
    ]["sha256"]
    assert manifest["module_sha256"] == reproducibility["material"]["files"][
        "source:src/qrc_vs_qelm/canonical_study.py"
    ]["sha256"]
    assert manifest["runner_sha256"] == reproducibility["material"]["files"][
        "runner"
    ]["sha256"]
    assert manifest["git"] == reproducibility["git"]
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected


def test_material_manifest_hashes_all_package_sources_and_study_inputs() -> None:
    manifest = material_file_manifest(
        repo_root=ROOT,
        config_path=CONFIG,
        runner_path=RUNNER,
    )
    files = manifest["files"]

    expected_sources = {
        f"source:{path.relative_to(ROOT).as_posix()}"
        for path in (ROOT / "src" / "qrc_vs_qelm").rglob("*.py")
    }
    expected_tests = {
        f"test:{path.relative_to(ROOT).as_posix()}"
        for path in (ROOT / "tests").rglob("*.py")
    }
    assert expected_sources <= set(files)
    assert expected_tests <= set(files)
    assert {
        "configuration",
        "project_metadata",
        "runner",
        "protocol:CANONICAL_EMULATION_LITERATURE_POSITIONING.md",
        "protocol:CANONICAL_EMULATION_PREREGISTRATION.md",
        "protocol:CANONICAL_EMULATION_THEORY.md",
        "protocol:EMULATION_MATCHED_PROTOCOL.md",
        "protocol:PREREGISTRATION.md",
        "protocol:PROTOCOL.md",
        "script:scripts/evaluate_canonical_gates.py",
        "script:scripts/render_canonical_emulation_figures.py",
    } <= set(files)
    assert files["configuration"]["path"] == "configs/canonical_emulation.yaml"
    assert files["runner"]["path"] == "scripts/run_canonical_emulation.py"
    assert files["source:src/qrc_vs_qelm/provenance.py"]["sha256"] == hashlib.sha256(
        (ROOT / "src" / "qrc_vs_qelm" / "provenance.py").read_bytes()
    ).hexdigest()
    assert manifest["file_count"] == len(files)
    assert SHA256.fullmatch(manifest["identity_sha256"])
    for entry in files.values():
        assert SHA256.fullmatch(entry["sha256"])
        assert entry["size_bytes"] >= 0


def test_environment_manifest_is_bounded_and_versioned() -> None:
    manifest = environment_manifest()

    assert manifest["python"]["version"] == platform.python_version()
    assert set(manifest["packages"]) == {
        "matplotlib",
        "numpy",
        "pandas",
        "pyyaml",
        "qrc_vs_qelm",
        "scikit_learn",
        "scipy",
    }
    assert manifest["privacy"] == {
        "environment_variables_captured": False,
        "hostname_captured": False,
    }


def test_confirmatory_freeze_record_binds_config_and_material(tmp_path: Path) -> None:
    record = {
        "schema_version": 1,
        "status": "FROZEN_CONFIRMATORY",
        "study": "canonical_qrc_to_parallel_qelm_emulation",
        "profile": "confirmatory",
        "config_sha256": "a" * 64,
        "material_identity_sha256": "b" * 64,
        "confirmatory_roots_previously_opened": False,
        "frozen_at_utc": "2026-08-28T12:00:00Z",
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    audit = verify_canonical_freeze_record(
        path,
        study=record["study"],
        profile=record["profile"],
        config_sha256="a" * 64,
        material_identity_sha256="b" * 64,
    )
    assert audit["passed"]

    mismatch = verify_canonical_freeze_record(
        path,
        study=record["study"],
        profile=record["profile"],
        config_sha256="c" * 64,
        material_identity_sha256="b" * 64,
    )
    assert not mismatch["passed"]
    assert not mismatch["checks"]["config_sha256"]

    record["frozen_at_utc"] = "2026-08-28T12:00:00"
    path.write_text(json.dumps(record), encoding="utf-8")
    naive_timestamp = verify_canonical_freeze_record(
        path,
        study=record["study"],
        profile=record["profile"],
        config_sha256="a" * 64,
        material_identity_sha256="b" * 64,
    )
    assert not naive_timestamp["passed"]
    assert not naive_timestamp["checks"]["frozen_at_utc"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_git_manifest_identifies_tracked_and_untracked_dirty_content(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test User")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "baseline")

    clean = git_worktree_manifest(tmp_path)
    assert clean["available"] is True
    assert clean["dirty"] is False
    assert clean["tracked_diff_bytes"] == 0
    assert clean["untracked_file_count"] == 0
    assert SHA256.fullmatch(clean["worktree_identity_sha256"])

    tracked.write_text("sensitive tracked change\n", encoding="utf-8")
    untracked = tmp_path / "untracked.txt"
    untracked.write_text("first private content\n", encoding="utf-8")
    dirty = git_worktree_manifest(tmp_path)
    assert dirty["dirty"] is True
    assert dirty["tracked_diff_bytes"] > 0
    assert dirty["untracked_file_count"] == 1
    assert dirty["worktree_identity_sha256"] != clean["worktree_identity_sha256"]
    assert "sensitive tracked change" not in json.dumps(dirty)
    assert "first private content" not in json.dumps(dirty)

    untracked.write_text("other private content\n", encoding="utf-8")
    changed_untracked = git_worktree_manifest(tmp_path)
    assert changed_untracked["status_sha256"] == dirty["status_sha256"]
    assert changed_untracked["untracked_tree_sha256"] != dirty["untracked_tree_sha256"]
    assert (
        changed_untracked["worktree_identity_sha256"]
        != dirty["worktree_identity_sha256"]
    )


def test_git_manifest_degrades_safely_outside_a_repository(tmp_path: Path) -> None:
    manifest = git_worktree_manifest(tmp_path)

    assert manifest == {
        "available": False,
        "commit": None,
        "branch": None,
        "tag_or_none": None,
        "dirty": None,
        "worktree_identity_sha256": None,
    }


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )
