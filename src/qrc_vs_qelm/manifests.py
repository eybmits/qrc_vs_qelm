"""Manifest and hashing helpers for frozen stage runners."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd


_SHA256_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")


def git_state(repo_root: str | Path) -> dict[str, Any]:
    """Return the current commit, exact tag if any, and dirty-tree status."""

    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD").stdout.strip()
    tag_result = _git(root, "describe", "--tags", "--exact-match", check=False)
    tag = tag_result.stdout.strip() if tag_result.returncode == 0 else None
    status = _git(root, "status", "--porcelain").stdout
    return {"commit": commit, "tag_or_none": tag, "dirty": bool(status.strip())}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's raw bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    """Hash canonical CSV bytes for a DataFrame."""

    columns = sorted(df.columns, key=lambda column: str(column))
    csv = df.loc[:, columns].to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()


def write_manifest(out_dir: str | Path, name: str, payload: dict[str, Any]) -> Path:
    """Write a manifest with deterministic core plus separate generated_at."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    repo_root = Path(payload.get("repo_root", Path.cwd()))
    normalized_payload = _normalize(payload)
    core: dict[str, Any] = {
        "name": name,
        "payload": normalized_payload,
        "git_state": git_state(repo_root),
        "package_versions": _package_versions(),
    }

    manifest = {
        "core": core,
        "core_sha256": hashlib.sha256(_canonical_json(core)).hexdigest(),
        "generated_at": _utc_now(),
    }
    manifest_path = out_path / f"{name}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def require_clean_tree(allow_dirty: bool) -> dict[str, bool]:
    """Enforce the preregistered clean-tree rule for stage runners."""

    state = git_state(Path.cwd())
    violated = bool(state["dirty"])
    if violated and not allow_dirty:
        raise SystemExit(
            "Preregistration requires a clean git tree; commit current changes first "
            "or rerun with --allow-dirty to record the violation."
        )
    return {"preregistration_violated": violated}


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"git {' '.join(args)} failed in {root}: {stderr}")
    return result


def _package_versions() -> dict[str, str | None]:
    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "pyyaml": "pyyaml",
        "qrc_vs_qelm": "qrc-vs-qelm",
    }
    versions: dict[str, str | None] = {}
    for key, distribution in distributions.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _expected_sha256(expected_sha256_path_or_value: str | Path) -> str:
    candidate = Path(expected_sha256_path_or_value)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
    else:
        text = str(expected_sha256_path_or_value)
    match = _SHA256_RE.search(text)
    if match is None:
        raise ValueError(
            "expected sha256 must be a 64-character hex digest or a file containing one"
        )
    return match.group(0).lower()


__all__ = [
    "git_state",
    "require_clean_tree",
    "sha256_file",
    "sha256_frame",
    "write_manifest",
]
