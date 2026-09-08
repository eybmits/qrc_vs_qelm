"""Keep Cobalt plot roots and the restored yellow observable accent consistent."""
from pathlib import Path
import re
import sys

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import paper_palette as palette  # noqa: E402
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
    selected = ROOT / "paper/figures/figure1_variants/variant_a_architectures.tex"
    assert source.read_bytes() == selected.read_bytes()
    definitions = dict(re.findall(
        r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-F]{6})\}", source.read_text()))
    assert definitions == {key: color.removeprefix("#") for key, color in palette.TIKZ.items()}


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
    intro = (base / "sections/introduction.tex").read_text()
    models = (base / "sections/models.tex").read_text()
    assert "Charcoal shows the target QRC; coral" in intro
    assert "backgrounds change from coral to blue" in intro
    assert "The coral prediction" in models
    assert r"\definecolor{linkblue}{HTML}{3868D9}" in (base / "main.tex").read_text()
