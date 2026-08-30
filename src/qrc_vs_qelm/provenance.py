"""Privacy-conscious provenance capture for reproducible study runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Any


_RUNTIME_DISTRIBUTIONS = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyyaml": "PyYAML",
    "qrc_vs_qelm": "qrc-vs-qelm",
    "scikit_learn": "scikit-learn",
    "scipy": "scipy",
}
_CANONICAL_PROTOCOL_FILES = (
    "CANONICAL_EMULATION_LITERATURE_POSITIONING.md",
    "CANONICAL_EMULATION_PREREGISTRATION.md",
    "CANONICAL_EMULATION_THEORY.md",
    "EMULATION_MATCHED_PROTOCOL.md",
    "PREREGISTRATION.md",
    "PROTOCOL.md",
)
_CANONICAL_SCRIPTS = (
    "scripts/evaluate_canonical_gates.py",
    "scripts/render_canonical_emulation_figures.py",
)


def sha256_file(path: Path) -> str:
    """Hash a file without loading large inputs into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strict_json_object(path: Path) -> dict[str, Any]:
    """Load a standards-compliant JSON object and reject duplicate keys."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError("freeze record must be a JSON object")
    return value


def verify_canonical_freeze_record(
    record_path: Path,
    *,
    study: str,
    profile: str,
    config_sha256: str,
    material_identity_sha256: str,
) -> dict[str, Any]:
    """Verify the independent approval record for a confirmatory material set."""

    path = Path(record_path).resolve()
    try:
        record = load_strict_json_object(path)
    except (OSError, ValueError) as exc:
        return {
            "passed": False,
            "path": str(path),
            "reason": str(exc),
        }
    frozen_at = record.get("frozen_at_utc")
    try:
        parsed_frozen_at = datetime.fromisoformat(str(frozen_at).replace("Z", "+00:00"))
        frozen_at_is_utc = parsed_frozen_at.tzinfo is not None and (
            parsed_frozen_at.utcoffset() == timezone.utc.utcoffset(parsed_frozen_at)
        )
    except (TypeError, ValueError):
        frozen_at_is_utc = False
    checks = {
        "schema_version": record.get("schema_version") == 1,
        "status": record.get("status") == "FROZEN_CONFIRMATORY",
        "study": str(record.get("study")) == str(study),
        "profile": str(record.get("profile")) == str(profile),
        "config_sha256": str(record.get("config_sha256")) == str(config_sha256),
        "material_identity_sha256": str(record.get("material_identity_sha256"))
        == str(material_identity_sha256),
        "confirmatory_roots_previously_opened": record.get(
            "confirmatory_roots_previously_opened"
        )
        is False,
        "frozen_at_utc": bool(frozen_at_is_utc),
    }
    return {
        "passed": bool(all(checks.values())),
        "path": str(path),
        "sha256": sha256_file(path),
        "checks": checks,
        "record": record,
    }


def canonical_material_files(
    *,
    repo_root: Path,
    config_path: Path,
    runner_path: Path | None,
) -> dict[str, Path]:
    """Return every code, configuration, and protocol file material to a run."""

    root = Path(repo_root).resolve()
    config = Path(config_path).resolve()
    runner = (
        Path(runner_path).resolve()
        if runner_path is not None
        else root / "scripts" / "run_canonical_emulation.py"
    )
    files: dict[str, Path] = {"configuration": config}
    fixed_files = {
        "project_metadata": root / "pyproject.toml",
        "runner": runner,
    }
    for label, path in fixed_files.items():
        if path.is_file():
            files[label] = path
    for filename in _CANONICAL_PROTOCOL_FILES:
        path = root / filename
        if path.is_file():
            files[f"protocol:{filename}"] = path
    for filename in _CANONICAL_SCRIPTS:
        path = root / filename
        if path.is_file():
            files[f"script:{filename}"] = path
    package_root = root / "src" / "qrc_vs_qelm"
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        files[f"source:{relative}"] = path
    tests_root = root / "tests"
    for path in sorted(tests_root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        files[f"test:{relative}"] = path
    return dict(sorted(files.items()))


def material_file_manifest(
    *,
    repo_root: Path,
    config_path: Path,
    runner_path: Path | None,
) -> dict[str, Any]:
    """Build a deterministic, content-addressed material-file inventory."""

    root = Path(repo_root).resolve()
    entries: dict[str, dict[str, Any]] = {}
    for label, path in canonical_material_files(
        repo_root=root,
        config_path=config_path,
        runner_path=runner_path,
    ).items():
        if not path.is_file():
            raise FileNotFoundError(f"material study file does not exist: {path}")
        entries[label] = {
            "path": _display_path(path, root),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    identity_payload = {
        label: entry["sha256"] for label, entry in entries.items()
    }
    return {
        "files": entries,
        "file_count": len(entries),
        "identity_sha256": _sha256_json(identity_payload),
    }


def environment_manifest() -> dict[str, Any]:
    """Record reproducibility-relevant versions without copying environment data."""

    versions: dict[str, str | None] = {}
    for label, distribution in _RUNTIME_DISTRIBUTIONS.items():
        try:
            versions[label] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[label] = None
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable_name": Path(sys.executable).name,
            "cache_tag": getattr(sys.implementation, "cache_tag", None),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "packages": versions,
        "privacy": {
            "environment_variables_captured": False,
            "hostname_captured": False,
        },
    }


def git_worktree_manifest(repo_root: Path) -> dict[str, Any]:
    """Identify a git worktree, including dirty content, without storing its diff."""

    root = Path(repo_root).resolve()
    commit_result = _git(root, "rev-parse", "HEAD")
    if commit_result.returncode != 0:
        return {
            "available": False,
            "commit": None,
            "branch": None,
            "tag_or_none": None,
            "dirty": None,
            "worktree_identity_sha256": None,
        }

    commit = _decode_line(commit_result.stdout)
    branch_result = _git(root, "branch", "--show-current")
    tag_result = _git(root, "describe", "--tags", "--exact-match")
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    combined = _git(
        root, "diff", "--binary", "--full-index", "--no-ext-diff", "HEAD", "--"
    )
    staged = _git(
        root, "diff", "--binary", "--full-index", "--no-ext-diff", "--cached", "--"
    )
    unstaged = _git(root, "diff", "--binary", "--full-index", "--no-ext-diff", "--")
    untracked = _untracked_tree_identity(root)

    status_bytes = status.stdout if status.returncode == 0 else b""
    tracked_diff = combined.stdout if combined.returncode == 0 else b""
    staged_diff = staged.stdout if staged.returncode == 0 else b""
    unstaged_diff = unstaged.stdout if unstaged.returncode == 0 else b""
    identity_payload = {
        "commit": commit,
        "status_sha256": _sha256_bytes(status_bytes),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "staged_diff_sha256": _sha256_bytes(staged_diff),
        "unstaged_diff_sha256": _sha256_bytes(unstaged_diff),
        "untracked_tree_sha256": untracked["sha256"],
    }
    return {
        "available": True,
        "commit": commit,
        "branch": _decode_line(branch_result.stdout) or None,
        "tag_or_none": (
            _decode_line(tag_result.stdout) if tag_result.returncode == 0 else None
        ),
        "dirty": bool(status_bytes),
        "status_sha256": identity_payload["status_sha256"],
        "tracked_diff_sha256": identity_payload["tracked_diff_sha256"],
        "tracked_diff_bytes": len(tracked_diff),
        "staged_diff_sha256": identity_payload["staged_diff_sha256"],
        "staged_diff_bytes": len(staged_diff),
        "unstaged_diff_sha256": identity_payload["unstaged_diff_sha256"],
        "unstaged_diff_bytes": len(unstaged_diff),
        "untracked_tree_sha256": untracked["sha256"],
        "untracked_file_count": untracked["file_count"],
        "untracked_bytes": untracked["size_bytes"],
        "worktree_identity_sha256": _sha256_json(identity_payload),
        "diff_content_captured": False,
        "untracked_paths_captured": False,
    }


def canonical_reproducibility_snapshot(
    *,
    repo_root: Path,
    config_path: Path,
    runner_path: Path | None,
) -> dict[str, Any]:
    """Capture the complete pre-simulation provenance envelope."""

    snapshot = {
        "schema_version": 1,
        "capture_phase": "before_simulation",
        "material": material_file_manifest(
            repo_root=repo_root,
            config_path=config_path,
            runner_path=runner_path,
        ),
        "environment": environment_manifest(),
        "git": git_worktree_manifest(repo_root),
    }
    snapshot["identity_sha256"] = _sha256_json(snapshot)
    return snapshot


def _untracked_tree_identity(repo_root: Path) -> dict[str, Any]:
    result = _git(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if result.returncode != 0:
        return {"sha256": None, "file_count": 0, "size_bytes": 0}
    digest = hashlib.sha256()
    count = 0
    total_size = 0
    for raw_relative in sorted(item for item in result.stdout.split(b"\0") if item):
        relative = raw_relative.decode("utf-8", errors="surrogateescape")
        path = repo_root / relative
        try:
            file_stat = path.lstat()
        except FileNotFoundError:
            kind = b"missing"
            content_digest = hashlib.sha256(b"").digest()
            size = 0
            mode = 0
        else:
            mode = stat.S_IMODE(file_stat.st_mode)
            if stat.S_ISLNK(file_stat.st_mode):
                kind = b"symlink"
                target = os.readlink(path).encode("utf-8", errors="surrogateescape")
                content_digest = hashlib.sha256(target).digest()
                size = len(target)
            elif stat.S_ISREG(file_stat.st_mode):
                kind = b"file"
                content_digest = bytes.fromhex(sha256_file(path))
                size = int(file_stat.st_size)
            else:
                kind = b"special"
                metadata_bytes = f"{file_stat.st_mode}:{file_stat.st_size}".encode()
                content_digest = hashlib.sha256(metadata_bytes).digest()
                size = int(file_stat.st_size)
        digest.update(len(raw_relative).to_bytes(8, byteorder="big"))
        digest.update(raw_relative)
        digest.update(kind)
        digest.update(mode.to_bytes(4, byteorder="big", signed=False))
        digest.update(size.to_bytes(16, byteorder="big", signed=False))
        digest.update(content_digest)
        count += 1
        total_size += size
    return {
        "sha256": digest.hexdigest(),
        "file_count": count,
        "size_bytes": total_size,
    }


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
    )


def _display_path(path: Path, repo_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)


def _decode_line(value: bytes) -> str:
    return value.decode("utf-8", errors="replace").strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


__all__ = [
    "canonical_material_files",
    "canonical_reproducibility_snapshot",
    "environment_manifest",
    "git_worktree_manifest",
    "load_strict_json_object",
    "material_file_manifest",
    "sha256_file",
    "verify_canonical_freeze_record",
]
