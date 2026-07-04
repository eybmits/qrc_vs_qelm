from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from qrc_vs_qelm.manifests import require_clean_tree, sha256_frame, write_manifest


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_dirty_tree_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Tests")
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")

    tracked.write_text("dirty\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit, match="commit current changes first"):
        require_clean_tree(allow_dirty=False)

    assert require_clean_tree(allow_dirty=True) == {"preregistration_violated": True}


def test_manifest_determinism(tmp_path: Path) -> None:
    payload = {
        "repo_root": Path.cwd(),
        "stage": 1,
        "rows": [3, 1, 2],
    }

    first_path = write_manifest(tmp_path, "stage1", payload)
    first_bytes = first_path.read_text(encoding="utf-8")
    second_path = write_manifest(tmp_path, "stage1", payload)
    second_bytes = second_path.read_text(encoding="utf-8")

    first = json.loads(first_bytes)
    second = json.loads(second_bytes)
    first.pop("generated_at")
    second.pop("generated_at")

    assert first == second
    assert first["core"]["payload"]["repo_root"] == str(Path.cwd())
    assert "numpy" in first["core"]["package_versions"]


def test_sha256_frame_uses_canonical_column_order() -> None:
    left = pd.DataFrame({"b": [1.25], "a": [2.5]})
    right = pd.DataFrame({"a": [2.5], "b": [1.25]})

    assert sha256_frame(left) == sha256_frame(right)
