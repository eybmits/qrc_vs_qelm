from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "pra_extension" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from render_dark_sector_figure import render  # noqa: E402


def test_dark_sector_figure_renders_with_traceable_source(tmp_path: Path) -> None:
    source = (
        REPO_ROOT
        / "pra_extension"
        / "outputs"
        / "dark_sector_validation"
        / "dark_sector_trajectories.csv"
    )
    manifest = render(source, tmp_path)
    assert set(manifest) == {
        "fig01_protected_excited_visible.pdf",
        "fig01_protected_excited_visible.png",
        "fig01_protected_excited_visible_source.csv",
        "fig01_protected_excited_visible_leakage_source.csv",
    }
    assert (tmp_path / "fig01_protected_excited_visible.pdf").stat().st_size > 10_000
    assert (tmp_path / "fig01_protected_excited_visible.png").stat().st_size > 50_000
    copied = pd.read_csv(tmp_path / "fig01_protected_excited_visible_source.csv")
    original = pd.read_csv(source)
    pd.testing.assert_frame_equal(copied, original)
    leakage = pd.read_csv(
        tmp_path / "fig01_protected_excited_visible_leakage_source.csv"
    )
    registered = leakage[leakage["source_kind"].eq("registered_frozen_points")]
    frozen = pd.read_csv(
        REPO_ROOT
        / "pra_extension"
        / "outputs"
        / "dark_sector_validation"
        / "dark_sector_leakage.csv"
    )
    pd.testing.assert_frame_equal(
        registered[frozen.columns].reset_index(drop=True), frozen
    )
    assert leakage["source_kind"].eq("analytic_from_frozen_config").sum() == 151
    robust = leakage[leakage["source_kind"].eq("robust_generator_certificate")]
    assert len(robust) == 4 * 81
    assert sorted(robust["asymmetry"].unique().tolist()) == [0.02, 0.05, 0.1, 0.2]
    assert (robust["protection_defect_upper_bound"] > 0).all()
    assert (robust["certified_gap_lower_bound"] >= 0).all()
    horizons = robust.groupby("asymmetry")["certified_horizon"].first()
    assert horizons.loc[0.02] > horizons.loc[0.05] > horizons.loc[0.1]
