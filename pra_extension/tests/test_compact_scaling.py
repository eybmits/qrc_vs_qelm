from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "pra_extension" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from render_compact_scaling import render  # noqa: E402


def test_compact_scaling_is_frozen_complete_and_traceable(tmp_path: Path) -> None:
    source = (
        REPO_ROOT
        / "paper"
        / "figures"
        / "canonical_emulation"
        / "fig07_dissipation_window_scaling_source.csv"
    )
    manifest = render(source, tmp_path)
    assert set(manifest) == {
        "fig03_dissipation_window_scaling_compact.pdf",
        "fig03_dissipation_window_scaling_compact.png",
        "fig03_dissipation_window_scaling_compact_source.csv",
    }
    assert (tmp_path / "fig03_dissipation_window_scaling_compact.pdf").stat().st_size > 10_000
    assert (tmp_path / "fig03_dissipation_window_scaling_compact.png").stat().st_size > 50_000
    copied = pd.read_csv(tmp_path / "fig03_dissipation_window_scaling_compact_source.csv")
    original = pd.read_csv(source).sort_values(
        ["teacher_n_qubits", "root_id", "teacher_gamma", "window"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(copied, original)
    assert copied.shape[0] == 480
    assert copied["root_fit_passed"].all()
    ledger = json.loads(
        (tmp_path / "dark_sector_figure_sha256.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest.items():
        actual = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert actual == expected == ledger[name]
