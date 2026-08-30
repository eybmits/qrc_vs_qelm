from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "pra_extension" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from render_compact_phase_map import (  # noqa: E402
    CHART_QUESTION,
    CHART_TAKEAWAY,
    COLORS,
    _validated_source,
    render,
)


def test_compact_phase_map_is_legible_and_traceable(tmp_path: Path) -> None:
    source = (
        REPO_ROOT
        / "paper"
        / "figures"
        / "canonical_emulation"
        / "fig04_regime_map_source.csv"
    )
    manifest = render(source, tmp_path)
    assert set(manifest) == {
        "fig02_phase_map_compact.pdf",
        "fig02_phase_map_compact.png",
        "fig02_phase_map_compact_source.csv",
    }
    assert (tmp_path / "fig02_phase_map_compact.pdf").stat().st_size > 10_000
    assert (tmp_path / "fig02_phase_map_compact.png").stat().st_size > 50_000
    copied = pd.read_csv(tmp_path / "fig02_phase_map_compact_source.csv")
    original = pd.read_csv(source).sort_values(
        ["teacher_n_qubits", "teacher_gamma", "window", "feature_budget"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(copied, original)
    assert (
        tmp_path / "fig02_phase_map_compact_source.csv"
    ).read_bytes() == source.read_bytes()
    assert original["modal_regime"].value_counts().to_dict() == {
        "HISTORY_IMPOSSIBLE": 108,
        "EMULATED": 17,
        "UNRESOLVED": 10,
        "REPRESENTATION_LIMITED": 7,
        "MIXED": 2,
    }
    assert original["agreement_n"].value_counts().sort_index().to_dict() == {
        5: 1,
        6: 4,
        7: 12,
        8: 7,
        10: 1,
        11: 1,
        12: 118,
    }
    assert int(original["n_roots"].sum()) == 144 * 12
    assert int(original["agreement_n"].lt(original["n_roots"]).sum()) == 26
    ledger = json.loads(
        (tmp_path / "dark_sector_figure_sha256.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest.items():
        actual = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert actual == expected == ledger[name]
    assert ledger["fig02_phase_map_compact_source.csv"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    assert "finite-window emulation" in CHART_QUESTION
    assert "out of H" in CHART_TAKEAWAY
    assert COLORS == {
        "HISTORY_IMPOSSIBLE": "#7A828C",
        "EMULATED": "#2F5D8C",
        "REPRESENTATION_LIMITED": "#D88C2D",
        "UNRESOLVED": "#2A7F78",
        "DEGENERATE_TARGET": "#1F2933",
        "MIXED": "#F2F3F5",
    }


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("n_roots", 11, "exactly 12 roots"),
        ("agreement_n", 11, "does not match regimes_present"),
        ("provenance_config_sha256", "0" * 63, "invalid provenance_config_sha256"),
    ),
)
def test_compact_phase_map_rejects_tampered_frozen_source(
    tmp_path: Path, column: str, value: object, message: str
) -> None:
    source = (
        REPO_ROOT
        / "paper"
        / "figures"
        / "canonical_emulation"
        / "fig04_regime_map_source.csv"
    )
    frame = pd.read_csv(source)
    frame.loc[0, column] = value
    tampered = tmp_path / "tampered.csv"
    frame.to_csv(tampered, index=False)
    with pytest.raises(ValueError, match=message):
        _validated_source(tampered)
