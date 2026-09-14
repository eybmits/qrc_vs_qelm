"""Keep Cobalt plot roots and the restored yellow observable accent consistent."""
from pathlib import Path
import re
import sys

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import paper_palette as palette  # noqa: E402
import active_paper_palette as active_palette  # noqa: E402
from render_information_interface_figures import PALETTE  # noqa: E402
from render_long_context_n5 import SERIES, FEATURE_SERIES  # noqa: E402

ROOT = CODE.parent.parent


def test_selected_cobalt_roots_and_mapping():
    assert [palette.COBALT, palette.CORAL, palette.VIOLET] == [
        "#3868D9", "#D66A56", "#7C5A9F"]
    assert palette.QRC_SIZES == {2: palette.COBALT, 3: palette.VIOLET, 4: palette.CORAL}
    assert palette.FEATURE_BUDGETS == {18: palette.COBALT, 36: palette.CORAL, 72: palette.VIOLET}
    assert len(set(palette.QRC_SIZES.values()) | set(palette.FEATURE_BUDGETS.values())) == 3


def test_native_tikz_exactly_matches_shared_palette():
    source = ROOT / "paper/figures/information_interface/fig01_information_interface.tex"
    definitions = dict(re.findall(
        r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-F]{6})\}", source.read_text()))
    assert definitions == {key: color.removeprefix("#") for key, color in active_palette.TIKZ.items()}


def test_observable_cards_restore_the_original_yellow_only():
    assert palette.FEATURE_FILL == "#FFC943"
    assert palette.FEATURE_BACKGROUND == "#FFDD92"
    assert palette.DIAGRAM["measured_quantities_and_readouts"] == "#FFC943"
    assert palette.DIAGRAM["classical_input_injection"] == palette.VIOLET


def test_qelm_color_repeats_across_model_and_feature_panels():
    assert PALETTE["onset"] == palette.DIAGRAM["ideal_window_output"] == palette.CORAL
    assert {row[0]: row[2] for row in SERIES} == {
        "history": palette.INK, "qelm": palette.CORAL, "quadratic": palette.VIOLET}
    assert {row[3] for row in FEATURE_SERIES} == {palette.CORAL, "white"}


def test_icaart_color_words_and_link_ink_match():
    base = ROOT / "paper/icaart2027"
    intro = " ".join((base / "sections/introduction.tex").read_text().split())
    models = (base / "sections/models.tex").read_text()
    assert "Orange denotes the QRC; blue denotes an ideal window prediction" in intro
    # Background styling remains in the artwork, not in the shorter caption.
    figure = (ROOT / "paper/figures/information_interface/fig01_information_interface.tex").read_text()
    assert r"\fill[green!24] (0,0) rectangle ({4.15*43/106},3.20);" in figure
    assert r"\fill[blue!16] ({4.15*43/106},0) rectangle (4.15,3.20);" in figure
    assert "The blue prediction" in models
    assert r"\definecolor{linkblue}{HTML}{3868D9}" in (base / "main.tex").read_text()


def test_active_model_semantics_preserve_the_legacy_palette():
    assert active_palette.TARGET == active_palette.INPUT == palette.INK
    assert active_palette.QRC == "#D97632"
    assert active_palette.QELM == palette.COBALT
    assert active_palette.DIAGRAM["target_output"] == active_palette.QRC
    assert active_palette.DIAGRAM["ideal_window_output"] == active_palette.QELM
    assert active_palette.DIAGRAM["classical_input_injection"] == palette.INK
    assert active_palette.FEATURE_FILL == palette.FEATURE_FILL
    assert active_palette.QRC_SIZES == palette.QRC_SIZES
    assert active_palette.FEATURE_BUDGETS == palette.FEATURE_BUDGETS
