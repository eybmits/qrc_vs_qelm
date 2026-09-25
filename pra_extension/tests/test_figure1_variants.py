from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "pra_extension/code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from figure1_timeseries import (  # noqa: E402
    BEGIN,
    END,
    INPUT_PARAMETERS,
    N_SAMPLES,
    OUTPUT_BEGIN,
    OUTPUT_END,
    OUTPUT_FORGETTING_HORIZON,
    OUTPUT_MEMORY_AMPLITUDE,
    OUTPUT_N_SAMPLES,
    OUTPUT_REVEAL_DURATION,
    OUTPUT_REVEAL_START,
    OUTPUT_SEED,
    OUTPUT_TEMPLATE_TIMESCALE,
    OUTPUT_WINDOW,
    OUTPUT_WRITE_STEP,
    PARAMETERS,
    SEED,
    SELECTED_STEP,
    WINDOW,
    WINDOW_START,
    output_baseline,
    output_samples,
    samples,
    tikz_output_sample_block,
    tikz_sample_block,
)
from qrc_vs_qelm.core.tasks import mackey_glass_series  # noqa: E402
from render_active_information_interface import (  # noqa: E402
    active_figure1_provenance as provenance,
    active_figure1_output_provenance as output_provenance,
)

DESIGNS = ROOT / "paper/figures/figure1_variants"
CANONICAL = ROOT / "paper/figures/information_interface/fig01_information_interface.tex"
COMPACT_ASPECT = 17.9197 / 9.6
INPUT_SHA256 = "4a7feed74ea28313ececc2b922b9d1a385046e3805fd6546960a8766cc80ba8e"
LEGACY_INPUT_SHA256 = "279ffd9bff72e717ab0a77396efdd92c4495e57d417f65f2bdddcc05680e25ba"
OUTPUT_SHA256 = "fc6853ac40b7445701d938ee595b7c19428cfab410dfba23c6f5544f23de7409"
FADING_TARGET_SHA256 = "1cc930442abbe9a3a69a8e2824ab0ee0ea120116cb8ebbdba6a16376c1dc4e5d"
NAMES = (
    "variant_a_architectures",
    "variant_b_same_window",
    "variant_c_relevant_memory",
)


@pytest.mark.parametrize("name", NAMES)
def test_figure1_design_dimensions_and_source_addresses(name: str) -> None:
    manifest = json.loads((DESIGNS / "design_manifest.json").read_text())
    assert manifest["schema"] == "figure1_design_candidates_v2"
    assert manifest["status"] == "variant_a_selected_for_main_paper"
    assert manifest["selected_variant"] == "variant_a_architectures"
    assert set(manifest["variants"]) == set(NAMES)
    assert manifest["native_width_inches"] == 7.055
    assert "canvas_aspect_ratio" not in manifest
    expected_aspect = COMPACT_ASPECT if name == "variant_a_architectures" else 1
    variant = manifest["variants"][name]
    assert hashlib.sha256((DESIGNS / "DESIGN_NOTES.md").read_bytes()).hexdigest() == (
        manifest["design_notes_sha256"]
    )
    for filename, digest in manifest["variants"][name]["artifacts"].items():
        assert hashlib.sha256((DESIGNS / filename).read_bytes()).hexdigest() == digest
    info = subprocess.run(
        ["pdfinfo", str(DESIGNS / f"{name}.pdf")],
        check=True, capture_output=True, text=True,
    ).stdout
    assert re.search(r"Pages:\s+1\b", info)
    match = re.search(r"Page size:\s*([\d.]+) x ([\d.]+) pts", info)
    assert match
    width, height = map(float, match.groups())
    assert width / height == pytest.approx(expected_aspect, abs=0.001)
    assert width / 72 == pytest.approx(7.055, abs=0.002)
    assert variant["pdf_size_points"] == pytest.approx([width, height])
    assert variant["canvas_aspect_ratio"] == pytest.approx(expected_aspect, abs=0.001)
    with Image.open(DESIGNS / f"{name}.png") as preview:
        assert preview.width == 1600
        assert preview.height == pytest.approx(1600 / expected_aspect, abs=1)
    source = (DESIGNS / f"{name}.tex").read_text()
    assert "schematic" in source.lower()
    assert r"\resizebox" not in source
    assert r"\scalebox" not in source
    assert r"\title{" not in source
    core = source.split("% Case labels sit outside the data frames", 1)[0]
    font_sizes = [float(size) for size in re.findall(r"\\fontsize\{([\d.]+)\}", core)]
    assert font_sizes
    assert min(font_sizes) >= 9.5
    assert max(font_sizes) <= 12
    main = (ROOT / "paper/canonical_emulation/main.tex").read_text()
    introduction = (ROOT / "paper/canonical_emulation/sections/introduction.tex").read_text()
    # The manuscript retains its canonical artifact path; archived alternatives
    # do not introduce duplicate figure inclusions after the user's selection.
    assert name not in main + introduction
    assert "fig01_information_interface.pdf" in introduction


def test_past_state_tint_stops_at_the_explicit_window() -> None:
    source = CANONICAL.read_text()
    fill = r"\fill[blue!16] (1.045,6.725) rectangle ({1.045+4.15*146.5/212},10.875);"
    assert source.count(fill) == 1
    assert source.index(fill) < source.index(r"\fill[green!24]")
    assert source.index(fill) < source.index(r"\foreach \val in {0.4,0.8,1.2}")
    assert source.index(fill) < source.index(r"in \MGsamples{")
    cue = provenance(CANONICAL)["past_state_cue"]
    assert cue["fill"] == "blue!16" and cue["root_color"] == "#D97632"
    assert cue["window_fill"] == "green!24"
    assert cue["window_start_boundary"]["index"] == WINDOW_START - .5
    boundary = (
        r"\draw[green!65!ink,line width=.4pt] "
        r"({1.045+4.15*146.5/212},6.725)--({1.045+4.15*146.5/212},10.875);"
    )
    assert source.count(boundary) == 1
    assert source.index(fill) < source.index(boundary) < source.index(r"in \MGsamples{")
    assert r"\node[text=green,fill=green!24,inner sep=.5pt]" in source
    assert cue["index_extent"] == [-6, WINDOW_START - .5]
    assert cue["data_indices"] == [0, WINDOW_START - 1]
    assert cue["label_center_cm"] == [2.55, 11.72]
    assert "no lossless-storage claim" in cue["scope"]
    assert r"\node[text=blue,align=center] at (2.55,11.72)" in source


def test_every_current_figure1_plot_has_a_quiet_two_direction_grid() -> None:
    source = CANONICAL.read_text()
    assert r"plotgrid/.style={draw=rule,line width=.35pt,dash pattern=on 2pt off 2pt}" in source
    assert r"\definecolor{rule}{HTML}{DEE0E3}" in source
    input_plot = source.split("% The old-input brace", 1)[0]
    output_plot = source.split(r"\newcommand{\memoryplot}[3]{%", 1)[1].split(
        r"\memoryplot{", 1,
    )[0]
    for panel in (input_plot, output_plot):
        assert panel.count(r"\draw[plotgrid]") == 2
    assert r"\foreach \kk in {0,50,100,150,200}" in input_plot
    assert r"\draw[plotgrid] (\xx,6.725)--(\xx,10.875);" in input_plot
    assert r"\draw[plotgrid] (1.045,\yy)--(5.195,\yy);" in input_plot
    assert input_plot.index(r"\fill[green!24]") < input_plot.index(r"\draw[plotgrid]")
    assert input_plot.rindex(r"\draw[plotgrid]") < input_plot.index(r"in \MGsamples{")
    assert r"\foreach \kk in {0,20,40,60,80,100}" in output_plot
    assert r"\draw[plotgrid] (\xx,0)--(\xx,3.20);" in output_plot
    assert r"\draw[plotgrid] (0,\yy)--(4.15,\yy);" in output_plot
    assert output_plot.index(r"\fill[blue!16]") < output_plot.index(r"\draw[plotgrid]")
    assert output_plot.rindex(r"\draw[plotgrid]") < output_plot.index(r"in #2{")
    assert len(re.findall(r"\\memoryplot\{[\d.]+\}\{\\OutputSamples(?:Fading|Persistent)\}", source)) == 2
    grid = provenance(CANONICAL)["plot_grid"]
    assert grid["directions"] == ["horizontal", "vertical"]
    assert grid["color"] == "#DEE0E3"
    assert grid["line_width_pt"] == 0.35
    assert grid["dash_pattern_pt"] == [2.0, 2.0]
    assert grid["input_vertical_steps"] == [0, 50, 100, 150, 200]
    assert grid["output_vertical_steps"] == [0, 20, 40, 60, 80, 100]
    assert grid["output_horizontal_values"] == [0.6, 0.8, 1.0, 1.2]
    assert "history_grid_local_y" not in grid

def test_selected_architecture_source_is_the_canonical_overview() -> None:
    manifest = json.loads((DESIGNS / "design_manifest.json").read_text())
    selected = DESIGNS / f"{manifest['selected_variant']}.tex"
    def without_colors(text):
        text = re.sub(r"%[^\n]*", "", text)
        text = re.sub(r"\\definecolor\{\w+\}\{HTML\}\{[0-9A-F]{6}\}", "", text)
        text = text.replace(r"\def\tracecolor{green}", r"\def\tracecolor{ink}")
        text = re.sub(
            r"(?<![\w])(?:draw=|fill=|text=)?(?:ink|blue|green|orange|purple|gold|muted|neutral|palegold|rule|white)(?:![\w.]+)*(?:,|(?=\]))",
            "", text,
        )
        return " ".join(text.replace(",]", "]").replace("[]", "").split())
    assert without_colors(CANONICAL.read_text()) == without_colors(selected.read_text())
    assert ROOT / manifest["canonical_source"] == CANONICAL
    digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    assert manifest["canonical_source_sha256"] == digest
    assert manifest["variants"]["variant_a_architectures"]["artifacts"][
        "variant_a_architectures.tex"
    ] == digest


def test_canonical_overview_omits_memory_headings_and_panel_letters() -> None:
    source = CANONICAL.read_text()
    panel_nodes = re.findall(
        r"\\node\[([^\]]*)\]\s+at\s+\([^;]*?\)\s+\{([abc])\};",
        source,
    )
    assert panel_nodes == []
    assert r"{Fading\\memory}" not in source
    assert r"{Persistent\\memory}" not in source
    assert not re.search(r"\\(?:sub)?title\{", source)


def test_selected_overview_keeps_larger_native_labels_and_measurement_symbols() -> None:
    source = CANONICAL.read_text()
    assert r"font=\fontsize{11}{12.5}\selectfont" in source
    assert r"font=\bfseries\fontsize{12}{13}\selectfont" in source
    core = source.split("% Case labels sit outside the data frames", 1)[0]
    font_sizes = [float(size) for size in re.findall(r"\\fontsize\{([\d.]+)\}", core)]
    assert min(font_sizes) >= 10
    meter = source.split(r"\newcommand{\meter}[3]{%", maxsplit=1)[1].split(
        BEGIN, maxsplit=1,
    )[0]
    box = re.search(
        r"\((-?[\d.]+),(-?[\d.]+)\) rectangle \(([\d.]+),([\d.]+)\)", meter,
    )
    assert box
    left, bottom, right, top = map(float, box.groups())
    assert right - left >= 0.48
    assert top - bottom >= 0.44
    observable_width = re.search(r"feature/\.style=.*?minimum width=([\d.]+)cm", source, re.S)
    assert observable_width
    # The new O_i labels omit repeated state subscripts so the compact banks
    # can leave a full 3.60-cm comparison column without shrinking the text.
    assert float(observable_width[1]) >= 1.50
    assert r"minimum height=.48cm" in source


@pytest.mark.parametrize("paper", ("canonical_emulation", "icaart2027"))
def test_selected_overview_is_introduced_once_near_the_start(paper: str) -> None:
    sections = ROOT / "paper" / paper / "sections"
    introduction = (sections / "introduction.tex").read_text()
    models = (sections / "models.tex").read_text()
    filename = "fig01_information_interface.pdf"
    assert introduction.count(filename) == 1
    assert filename not in models
    # Queue the overview as the first figure within the short opening setup,
    # independently of source line wrapping. The active edition first states
    # the question and then explains state versus window; the archived edition
    # uses one opening paragraph. Final-size placement still needs visual review.
    first_figure = re.search(
        r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", introduction, re.S,
    )
    assert first_figure is not None
    assert first_figure.group().startswith(r"\begin{figure*}")
    assert filename in first_figure.group()
    before_figure = introduction[:first_figure.start()]
    before_figure = re.sub(r"(?m)^[ \t]*%[^\n]*\n?", "", before_figure)
    before_figure = re.sub(r"\\(?:section|label)\{[^{}]*\}", "", before_figure)
    assert not re.search(r"\\(?:input|include|subsection)\b", before_figure)
    paragraphs = re.split(r"\n\s*\n", before_figure.strip())
    assert all(paragraph.strip() for paragraph in paragraphs)
    if paper == "icaart2027":
        assert 1 <= len(paragraphs) <= 2
        assert len(before_figure.split()) <= 200
        opening = " ".join(before_figure.split())
        assert re.search(r"replace.*?depends on.*?access to the past", opening)
        assert "A QRC carries its quantum state forward as new inputs arrive" in opening
        assert re.search(r"reset-window QELMs.*?start afresh for each prediction.*?"
                         r"explicit window of recent inputs", opening)
    else:
        assert len(paragraphs) == 1


def test_compact_canvas_keeps_native_typography_and_unscaled_architecture() -> None:
    source = CANONICAL.read_text()
    assert "x=1cm,y=1cm" in source
    bounds = re.search(
        r"\\useasboundingbox \(([\d.-]+),([\d.-]+)\) rectangle \(([\d.-]+),([\d.-]+)\);",
        source,
    )
    assert bounds
    assert tuple(map(float, bounds.groups())) == pytest.approx((0, 3.6, 17.9197, 13.2))
    assert not re.search(r"(?:^|[\[,\s])(?:xscale|yscale|scale)\s*=", source)
    assert "transform shape" not in source
    geometry = provenance(CANONICAL)["plot_geometry"]
    assert geometry["canvas_cm"] == pytest.approx([17.9197, 9.6])
    assert geometry["canvas_bounds_cm"] == pytest.approx([0, 3.6, 17.9197, 13.2])

def test_retired_history_error_panel_is_absent_from_the_current_artwork() -> None:
    source = CANONICAL.read_text()
    artwork = re.sub(r"%[^\n]*", "", source)
    for retired_label in (
        "best history error", "Window QELM", "Two reservoir-memory cases",
        "desired error tolerance", r"D_d/2", r"\varepsilon", r"\epsilon",
    ):
        assert retired_label not in artwork
    metadata = provenance(CANONICAL)
    assert "history_error_curve" not in metadata
    assert "history_error_axes_bounds_cm" not in metadata["plot_geometry"]
    # The toy gaps belong to chosen trajectories and must not be labelled
    # as the supremal ambiguity over all admissible histories.
    assert not re.search(r"\\node[^;]*D_d", artwork)

def test_scalar_midpoint_inset_is_removed() -> None:
    source = CANONICAL.read_text()
    assert "Same window, different pasts" not in source
    assert r"one shared prediction\\at the midpoint" not in source
    assert "midpoint_inset" not in provenance(CANONICAL)

def test_native_pdf_has_outside_case_labels_and_one_shared_model_legend() -> None:
    bbox = subprocess.run(
        ["pdftotext", "-bbox", str(CANONICAL.with_suffix(".pdf")), "-"],
        capture_output=True, text=True, check=True,
    ).stdout
    words = [node for node in ET.fromstring(bbox).iter() if node.tag.endswith("}word")]
    right_column = [word for word in words if float(word.attrib["xMin"]) > 13.5 * 72 / 2.54]
    text = [word.text for word in right_column]
    assert text.count("Fading") == text.count("Persistent") == 1
    assert text.count("memory") == text.count("Window") == 2
    assert text.count("sufficient") == text.count("insufficient") == 1
    assert text.count("QRC") == text.count("QELM") == 1
    assert not {"a", "b", "epsilon", "tolerance"} & set(text)

def test_output_toy_is_reproducible_and_obeys_the_balanced_bit_information_cut(monkeypatch) -> None:
    assert OUTPUT_N_SAMPLES == 101 and OUTPUT_SEED == 2
    assert OUTPUT_WINDOW == 40 and OUTPUT_WRITE_STEP == 0
    assert OUTPUT_MEMORY_AMPLITUDE == 0.25
    assert OUTPUT_REVEAL_START == 32 and OUTPUT_REVEAL_DURATION == 64
    k = np.arange(101, dtype=float)
    raw = mackey_glass_series(101, rng=np.random.default_rng(2), **PARAMETERS)
    template = 0.4 + 0.65 * (raw - raw.min()) / (raw.max() - raw.min())
    fraction = np.clip((k - 32) / 64, 0, 1)
    coefficient = 0.25 * fraction * fraction * (3 - 2 * fraction)
    baseline = template + 0.25 - coefficient
    target = template + 0.25
    other_history = baseline - coefficient
    prediction = np.where(k < 40, target, baseline)
    expected = np.column_stack((target, prediction))
    np.testing.assert_array_equal(output_baseline("persistent"), baseline)
    np.testing.assert_array_equal(output_samples(), expected)
    np.testing.assert_array_equal(output_samples(case="persistent"), expected)
    assert hashlib.sha256(expected.astype("<f8").tobytes()).hexdigest() == OUTPUT_SHA256
    # Keep every target coordinate byte-identical to the previous drawing;
    # only the ideal lower prediction and its declared balanced pair change.
    assert hashlib.sha256(target.astype("<f8").tobytes()).hexdigest() == (
        "d1c8824b249900beeba99ac5668a73bf88827b9b381aede88004722b22482d1c"
    )
    assert template.min() == pytest.approx(0.4)
    assert template.max() == pytest.approx(1.05)
    np.testing.assert_array_equal(prediction[:40], target[:40])
    np.testing.assert_allclose(prediction[40:], (target[40:] + other_history[40:]) / 2)
    gap = target[40:] - prediction[40:]
    np.testing.assert_allclose(gap, coefficient[40:])
    np.testing.assert_allclose(prediction[40:] - other_history[40:], coefficient[40:])
    assert gap[0] == pytest.approx(0.0107421875)
    assert np.all(np.diff(gap) >= -1e-15)
    np.testing.assert_allclose(gap[96 - 40:], 0.25)
    # These windows include the bit exactly through time39, not at time40.
    for time in range(101):
        available_steps = range(time - OUTPUT_WINDOW + 1, time + 1)
        assert (OUTPUT_WRITE_STEP in available_steps) == (time < 40)
    # The balanced scalar pair makes the midpoint both Bayes-optimal and
    # minimax; its lower bound concerns both histories, not one chosen trace.
    pair_mse = 0.5 * ((target[40:] - prediction[40:]) ** 2 + (other_history[40:] - prediction[40:]) ** 2)
    np.testing.assert_allclose(pair_mse, coefficient[40:] ** 2)
    assert np.min(expected) > 0.3 and np.max(expected) < 1.5

    def forbid_input_dependency():
        raise AssertionError("The auxiliary toy must not depend on the displayed input trace")

    monkeypatch.setattr("figure1_timeseries.samples", forbid_input_dependency)
    np.testing.assert_array_equal(output_samples(), expected)
    metadata = output_provenance()
    assert "not simulated, fitted, or measured QRC/QELM performance" in metadata["status"]
    assert "separate auxiliary toy timeline" in metadata["relationship_to_input"]
    assert "not predictions computed from the left input trace" in metadata["relationship_to_input"]
    assert metadata["template_stream"]["seed"] == 2
    assert metadata["template_stream"]["generated_count"] == 101
    assert metadata["template_stream"]["indices"] == [0, 100]
    assert "predetermined common baseline" in metadata["template_stream"]["scope"]
    assert "not fitted preprocessing" in metadata["template_stream"]["scope"]
    np.testing.assert_array_equal(metadata["baseline_values"], template)
    persistent = metadata["cases"]["persistent"]
    np.testing.assert_array_equal(persistent["baseline_values"], baseline)
    np.testing.assert_array_equal(persistent["memory_coefficients"], coefficient)
    np.testing.assert_array_equal(persistent["other_history_values"], other_history)
    access = metadata["memory_access"]
    assert access["bit_values"] == [-1, 1] and access["bit_probabilities"] == [0.5, 0.5]
    assert access["displayed_bit"] == 1
    assert access["write_step"] == 0 and access["first_excluded_step"] == 40
    assert access["window_length"] == 40 and access["maximum_memory_amplitude"] == 0.25
    np.testing.assert_array_equal(persistent["post_expiry_pair_gap_values"], 2 * coefficient[40:])
    np.testing.assert_array_equal(persistent["post_expiry_pair_midpoint_error_values"], coefficient[40:])
    assert "identical later inputs" in access["shared_future"]
    assert "independent of time and later inputs" in access["shared_future"]
    assert "cannot fit both histories exactly" in metadata["optimality_scope"]
    assert "not a fitted finite-feature QELM" in metadata["emulator_scope"]
    assert metadata["units"] == "arbitrary time and output units"


def test_persistent_reveal_is_a_fixed_output_schedule_not_window_dependent(monkeypatch) -> None:
    import figure1_timeseries as toy

    coefficients = np.asarray(output_provenance()["cases"]["persistent"]["memory_coefficients"])
    target = output_samples("persistent")[:, 0].copy()
    baseline = output_baseline("persistent").copy()
    np.testing.assert_array_equal(coefficients[:OUTPUT_REVEAL_START + 1], 0)
    assert coefficients[OUTPUT_REVEAL_START + 1] > 0
    np.testing.assert_array_equal(coefficients[96:], 0.25)
    assert np.all(np.diff(coefficients) >= 0)
    # Changing what the window can see changes the conditional prediction,
    # never the carried bit's fixed reveal schedule or the target itself.
    monkeypatch.setattr(toy, "OUTPUT_WINDOW", 60)
    changed = output_samples("persistent")
    np.testing.assert_array_equal(changed[:, 0], target)
    np.testing.assert_array_equal(output_baseline("persistent"), baseline)
    np.testing.assert_array_equal(changed[:60, 1], target[:60])
    np.testing.assert_array_equal(changed[60:, 1], baseline[60:])
    np.testing.assert_array_equal(
        output_provenance()["cases"]["persistent"]["memory_coefficients"], coefficients,
    )


def test_fading_output_toy_forgets_the_pair_exactly_without_changing_the_charcoal_target(
    monkeypatch,
) -> None:
    assert OUTPUT_FORGETTING_HORIZON == 40 and OUTPUT_TEMPLATE_TIMESCALE == 30
    k = np.arange(OUTPUT_N_SAMPLES, dtype=float)
    raw = mackey_glass_series(101, rng=np.random.default_rng(2), **PARAMETERS)
    template = 0.4 + 0.65 * (raw - raw.min()) / (raw.max() - raw.min())
    # The exponential is only the legacy charcoal drawing template. The
    # old bit itself follows a separate finite-horizon physical coefficient.
    target = template + 0.25 * np.exp(-k / 30)
    coefficient = 0.25 * (1 - np.clip(k / 40, 0, 1)) ** 2
    baseline = target - coefficient
    other_history = baseline - coefficient
    prediction = np.where(k < OUTPUT_WINDOW, target, baseline)
    expected = np.column_stack((target, prediction))
    np.testing.assert_array_equal(output_baseline("fading"), baseline)
    np.testing.assert_array_equal(output_samples(case="fading"), expected)
    assert hashlib.sha256(target.astype("<f8").tobytes()).hexdigest() == FADING_TARGET_SHA256
    np.testing.assert_array_equal(expected[:40, 0], expected[:40, 1])
    np.testing.assert_array_equal(coefficient[40:], 0)
    np.testing.assert_array_equal(expected[40:, 0], expected[40:, 1])
    np.testing.assert_array_equal(target[40:], other_history[40:])
    np.testing.assert_array_equal(prediction[40:], other_history[40:])
    # Both bit histories are predicted exactly, not just the displayed +1
    # trajectory. Before expiry their differing windows still reveal the bit.
    other_prediction = np.where(k < OUTPUT_WINDOW, other_history, baseline)
    np.testing.assert_array_equal(other_prediction, other_history)
    assert np.all(coefficient[:40] > 0)
    assert np.all(np.diff(coefficient[:41]) < 0)
    assert target[0] - baseline[0] == pytest.approx(0.25)
    assert target[39] - baseline[39] == pytest.approx(0.25 / 40**2)
    np.testing.assert_allclose(target - other_history, 2 * coefficient)
    np.testing.assert_allclose(
        0.5 * ((target[40:] - prediction[40:]) ** 2
               + (other_history[40:] - prediction[40:]) ** 2),
        coefficient[40:] ** 2,
    )

    def forbid_input_dependency():
        raise AssertionError("The auxiliary toy must not use the displayed input trace")

    monkeypatch.setattr("figure1_timeseries.samples", forbid_input_dependency)
    np.testing.assert_array_equal(output_samples(case="fading"), expected)
    assert hashlib.sha256(output_samples().astype("<f8").tobytes()).hexdigest() == OUTPUT_SHA256
    metadata = output_provenance()
    fading = metadata["cases"]["fading"]
    assert fading["forgetting_horizon_steps"] == 40
    assert fading["formulas"]["memory_coefficient"] == "c[t]=0.25*(1-s[t])^2; s[t]=clip(t/40,0,1)"
    np.testing.assert_array_equal(fading["memory_coefficients"], coefficient)
    np.testing.assert_array_equal(fading["baseline_values"], baseline)
    np.testing.assert_array_equal(fading["other_history_values"], other_history)
    np.testing.assert_array_equal(fading["post_expiry_pair_midpoint_error_values"], coefficient[40:])
    np.testing.assert_array_equal(fading["post_expiry_pair_gap_values"], 2 * coefficient[40:])
    assert "not the uniform D_d limit" in fading["decay_scope"]
    assert "special illustrative case, not a generic property of fading memory" in fading["decay_scope"]
    assert "not the bit's memory" in fading["preservation_scope"]
    assert "not labeled or identified as the generic supremal D_d/2" in metadata["curve_gap_scope"]


def test_fading_forgetting_horizon_is_physical_and_independent_of_window_access(monkeypatch) -> None:
    import figure1_timeseries as toy

    original = output_samples("fading").copy()
    baseline = output_baseline("fading").copy()
    fading = output_provenance()["cases"]["fading"]
    coefficients = np.asarray(fading["memory_coefficients"])
    other_history = np.asarray(fading["other_history_values"])
    # A shorter window loses the bit before the same physical horizon.
    # This must expose a real residual gap, not shift or snap the target.
    monkeypatch.setattr(toy, "OUTPUT_WINDOW", 20)
    changed = output_samples("fading")
    np.testing.assert_array_equal(changed[:, 0], original[:, 0])
    np.testing.assert_array_equal(output_baseline("fading"), baseline)
    np.testing.assert_array_equal(changed[:20, 1], original[:20, 0])
    np.testing.assert_array_equal(changed[20:, 1], baseline[20:])
    assert np.all(changed[20:40, 0] > changed[20:40, 1])
    assert np.all(changed[20:40, 1] > other_history[20:40])
    np.testing.assert_allclose(changed[20:40, 0] - changed[20:40, 1], coefficients[20:40])
    np.testing.assert_allclose(changed[20:40, 1] - other_history[20:40], coefficients[20:40])
    np.testing.assert_array_equal(changed[40:, 0], changed[40:, 1])
    np.testing.assert_array_equal(changed[40:, 1], other_history[40:])
    updated = output_provenance()
    assert updated["memory_access"]["first_excluded_step"] == 20
    assert updated["cases"]["fading"]["forgetting_horizon_steps"] == 40
    np.testing.assert_array_equal(updated["cases"]["fading"]["memory_coefficients"], coefficients)
    np.testing.assert_array_equal(updated["cases"]["fading"]["other_history_values"], other_history)


def test_output_toy_rejects_unknown_memory_case() -> None:
    with pytest.raises(ValueError):
        output_samples(case="unknown")


def test_tikz_contains_exact_two_case_output_toy_block() -> None:
    source = CANONICAL.read_text()
    literal = source[source.index(OUTPUT_BEGIN):source.index(OUTPUT_END) + len(OUTPUT_END)]
    assert literal == tikz_output_sample_block()
    for case in ("persistent", "fading"):
        block = re.search(
            rf"\\def\\OutputSamples{case.capitalize()}\{{%\n(.*?)\n\}}",
            literal, re.S,
        )
        assert block
        rows = re.findall(r"^(\d+)/([\d.eE+-]+)/([\d.eE+-]+),?$", block[1], re.M)
        assert [int(row[0]) for row in rows] == list(range(101))
        recorded = np.array([row[1:] for row in rows], dtype=float)
        np.testing.assert_array_equal(recorded, output_samples(case=case))
    metadata = provenance(CANONICAL)["output_comparisons"]
    assert metadata == output_provenance()
    assert metadata["column_order"] == ["target", "prediction"]
    assert metadata["cases"]["persistent"]["float64_le_sha256"] == OUTPUT_SHA256
    assert metadata["sample_indices"] == list(range(101))
    for case in ("persistent", "fading"):
        values = output_samples(case=case)
        np.testing.assert_array_equal(metadata["cases"][case]["values"], values)
        assert metadata["cases"][case]["float64_le_sha256"] == hashlib.sha256(
            values.astype("<f8").tobytes(),
        ).hexdigest()


@pytest.mark.parametrize("change", ("altered_persistent", "altered_fading", "duplicated", "missing"))
def test_output_provenance_rejects_corrupted_blocks(tmp_path: Path, change: str) -> None:
    source = CANONICAL.read_text()
    block = tikz_output_sample_block()
    if change.startswith("altered_"):
        case = change.removeprefix("altered_")
        start = f"\\def\\OutputSamples{case.capitalize()}{{%\n0/"
        value = f"{output_samples(case=case)[0, 0]:.17g}"
        replacement = block.replace(start + value, start + "0.1", 1)
        assert replacement != block
    else:
        replacement = {"duplicated": block + "\n" + block, "missing": ""}[change]
    changed = tmp_path / "changed.tex"
    changed.write_text(source.replace(block, replacement, 1))
    with pytest.raises(ValueError, match="output"):
        provenance(changed)


def test_overview_adds_whitespace_without_shrinking_its_contents() -> None:
    source = CANONICAL.read_text()
    architecture = source.split("% Deterministic illustrative output overlays", maxsplit=1)[0]
    # Architecture coordinates are absolute; only data-coordinate transforms
    # remain scoped. Whole-lane translations would move the specified axes.
    assert not re.search(r"\\begin\{scope\}\[shift=\{\(0,[\d.]+\)\}\]", architecture)
    rows = []
    for upper, lower in (("qo1", "qom"), ("qf1", "qfb")):
        centers = []
        for name in (upper, lower):
            match = re.search(rf"\({name}\) at \(([\d.]+),([\d.]+)\)", source)
            assert match
            centers.append(float(match.group(2)))
        assert centers[0] - centers[1] >= 1.22
        rows.append(centers)
    assert rows[0][1] - rows[1][0] > 1.0
    wires = re.findall(r"\\foreach \\yy in \{([\d.,]+)\}", architecture)
    assert len(wires) == 2
    assert all(len(row.split(",")) == 3 for row in wires)
    assert r"font=\fontsize{11}{12.5}\selectfont" in source


def test_current_input_and_carried_state_do_not_imply_batch_encoding() -> None:
    source = CANONICAL.read_text()
    qrc_source = source.split("% Fresh reset,", maxsplit=1)[0]
    scalar_leader = re.search(
        r"\\draw\[purple,[^\]]+\]\s*\(currentinput\)([^;]+);", qrc_source,
    )
    assert scalar_leader and "(inputnode.south)" in scalar_leader[1]
    assert "0 |- currentinput" in scalar_leader[1]
    assert re.search(r"\\draw\[flow,purple,[^\]]+\] \(inputnode.east\)", qrc_source)
    scalar_node = re.search(r"\(inputnode\) at \(([\d.]+),([\d.]+)\)", qrc_source)
    assert scalar_node and float(scalar_node[1]) > 5.195 and float(scalar_node[2]) > 10.875
    assert r"\ifnum\kk=180 \coordinate (currentinput)" in qrc_source
    selected_x = 1.045 + 4.15 * (SELECTED_STEP + 6) / 212
    assert 1.045 < selected_x < 5.195
    guide = re.search(r"\\draw\[muted,densely dashed,[^\]]+\] ([^;]+);", qrc_source)
    assert guide
    guide_points = re.findall(
        r"\(\{([\d.]+)\+([\d.]+)\*([\d.]+)/([\d.]+)\},([\d.]+)\)", guide[1],
    )
    assert len(guide_points) == 2
    actual_guide = [(a + b * c / d, y) for a, b, c, d, y in (
        map(float, point) for point in guide_points
    )]
    np.testing.assert_allclose(actual_guide, [(selected_x, 6.725), (selected_x, 10.875)])
    assert re.search(r"\\node\[draw=ink,fill=purple,circle,[^\]]+\] at \(currentinput\)", qrc_source)
    assert r"{$a_t$}" in qrc_source and r"$\Phi_{a_t}$" in qrc_source
    assert r"carried state $\rho_t$" in qrc_source
    assert "Only the current scalar enters each general QRC channel update" in qrc_source
    # A blue brace associates older inputs with carried state, rather than
    # feeding the whole historical trace to the current channel update.
    past_brace = re.search(r"\\draw\[blue,decorate,[^\n]+\]\s*([^;]+);", qrc_source)
    assert past_brace
    past_points = re.findall(
        r"\(\{([\d.]+)\+([\d.]+)\*([\d.]+)/([\d.]+)\},([\d.]+)\)", past_brace[1],
    )
    assert len(past_points) == 2
    actual_past = [(a + b * c / d, y) for a, b, c, d, y in (
        map(float, point) for point in past_points
    )]
    np.testing.assert_allclose(actual_past, [
        (1.045 + 4.15 * 6 / 212, 11.14),
        (1.045 + 4.15 * 146.5 / 212, 11.14),
    ])
    assert r"past carried in\\QRC state $\rho_{t-1}$" in qrc_source
    assert "incomingstate" not in qrc_source
    channel = re.search(
        r"\\draw\[ink,fill=blue!12,[^\]]+\]\s*"
        r"\(([\d.]+),([\d.]+)\) rectangle \(([\d.]+),([\d.]+)\)", qrc_source,
    )
    assert channel
    channel_left, channel_bottom, channel_right, _ = map(float, channel.groups())
    control_route = re.search(
        r"\\draw\[flow,purple,[^\]]+\] \(inputnode.east\)\s*"
        r"--\(([\d.]+),([\d.]+)\)--\(([\d.]+),([\d.]+)\);",
        qrc_source,
    )
    assert control_route  # horizontal first, then a vertical downward entry
    bend_x, bend_y, tip_x, tip_y = map(float, control_route.groups())
    channel_top = float(channel[4])
    assert float(scalar_node[1]) < bend_x == tip_x
    assert bend_y == float(scalar_node[2])
    assert tip_x == pytest.approx((channel_left + channel_right) / 2)
    assert tip_y == channel_top
    assert bend_y - tip_y >= 0.30
    assert "controls" not in control_route[0]
    assert "rounded corners" not in scalar_leader[0]
    heading = re.search(r"\\node\[text=blue\] at \(([\d.]+),([\d.]+)\) \{QRC\};", qrc_source)
    assert heading and float(heading[1]) == tip_x
    assert float(heading[2]) - bend_y >= 0.35
    loop = re.search(r"\\draw\[flow,blue\] ([^;]+);", qrc_source)
    assert loop
    points = np.array(re.findall(r"\(([\d.]+),([\d.]+)\)", loop[1]), dtype=float)
    assert points.shape == (4, 2)
    # The state-feedback loop leaves after the channel, before measurement,
    # and returns before the channel: it is not a measurement-feedback loop.
    assert points[0, 0] > channel_right
    assert points[-1, 0] < channel_left
    assert points[1, 1] < channel_bottom and points[2, 1] < channel_bottom
    bank = re.search(r"\(qo1\) at \(([\d.]+),([\d.]+)\)", qrc_source)
    assert bank and points[0, 0] < float(bank[1])
    assert "The whole window, rather than its endpoint" in source
    assert r"$E(S_d)$" in source
    assert r"$U(S_d)$" not in source
    assert "fixed gates" in source
    for paper in ("canonical_emulation", "icaart2027"):
        text = (ROOT / "paper" / paper / "sections/introduction.tex").read_text()
        caption = " ".join(text.split(r"\caption{", 1)[1].split(r"\label{", 1)[0].split())
        # Brace geometry is checked above. The active caption now explains the
        # same state-versus-window distinction in a plain sentence; the archived
        # edition retains its original wording.
        if paper == "icaart2027":
            assert "QRC carries its quantum state forward" in caption
            assert "reset-QELM receives a window of recent inputs" in caption
        else:
            assert "Carried QRC state" in caption
            assert "reset-QELM's explicit input window" in caption
        models = " ".join((ROOT / "paper" / paper / "sections/models.tex").read_text().split())
        # Causal sample membership is guarded numerically above and by the
        # source record; the rewritten Section 2 explains the model itself.
        assert not set(provenance(CANONICAL)["future_context_indices"]) & set(
            provenance(CANONICAL)["window_indices"],
        )
        assert "A trained linear readout can" in models
        if paper == "icaart2027":
            assert re.search(r"fixed map.*?\\mathcal C.*?finite vector of observable expectations", models)
            assert re.search(r"A trained linear readout can.*?weights fixed during emulation", models)
            assert r"\rho_t=\Phi_{a_t}(\rho_{t-1})" in models
            assert "Emulation targets the reservoir's output after the same current input" in models
            assert "not the next value of an external time series" in models
            # Empirical readout details now live with the protocols/results,
            # not in the architecture definition. Keep the same distinction:
            # measured QRC observables for emulation, externally fitted task
            # targets for both models in forecasting.
            methods = " ".join((ROOT / "paper" / paper / "sections/methodology.tex").read_text().split())
            evaluation = " ".join((ROOT / "paper" / paper / "sections/evaluation.tex").read_text().split())
            assert re.search(r"Pauli measurements give 9, 18, and 30 outputs", methods)
            assert "QELM predicts the QRC's final $Z_0$ expectation, not the next laser value" in methods
            assert re.search(r"NARMA10 tests prediction rather than emulation: both models learn "
                             r"the next target value from the input sequence", evaluation)
            assert "readouts predict unquantized $y_{t+1}$" in methods
            assert "never past targets" in methods
        else:
            assert "fixed weights into the observables" in models
            assert "target readout is the identity" in models


def test_input_gutter_increases_without_resizing_the_dynamics() -> None:
    source = CANONICAL.read_text()
    geometry = provenance(CANONICAL)["architecture_geometry"]
    assert geometry["dynamics_shift_right_cm"] == 0.35
    assert geometry["observable_banks_shift_right_cm"] == 0.15
    for fill, key, old_left, width in (
        ("blue!12", "qrc_channel_bounds_cm", 6.25, 1.35),
        ("green!13", "reset_encoding_bounds_cm", 6.12, 1.06),
    ):
        bounds = re.search(
            rf"\\draw\[ink,fill={fill},[^\]]+\]\s*"
            r"\(([\d.]+),([\d.]+)\) rectangle \(([\d.]+),([\d.]+)\);",
            source,
        )
        assert bounds
        left, bottom, right, top = map(float, bounds.groups())
        assert [left, bottom, right, top] == geometry[key]
        assert left - old_left == pytest.approx(0.35)
        assert right - left == pytest.approx(width)
        assert top - bottom == pytest.approx(1.90)
    ket = re.search(r"\\node\[anchor=east\] at \(([\d.]+),\\yy\) \{\$\|0\\rangle\$\};", source)
    assert ket
    assert float(ket[1]) == geometry["reset_state_ket_anchor_x_cm"]
    assert float(ket[1]) - 6.00 == pytest.approx(0.35)
    scalar = re.search(r"\(inputnode\) at \(([\d.]+),([\d.]+)\)", source)
    assert scalar and list(map(float, scalar.groups())) == geometry["current_input_node_cm"]
    for x, y in geometry["current_input_control_vertices_cm"]:
        assert f"--({x:.3f},{y:.2f})" in source
    assert "final vertical arrow down" in geometry["current_input_route"]


def test_observable_banks_feed_small_readouts_and_the_shared_comparison_column() -> None:
    architecture = CANONICAL.read_text().split("% Deterministic illustrative output overlays", maxsplit=1)[0]
    coordinate = r"\(([\d.]+),([\d.]+)\)"
    triangles = re.findall(
        rf"\\draw\[ink,fill=gold,[^\]]+\] {coordinate}--{coordinate}--{coordinate}--cycle;",
        architecture,
    )
    assert len(triangles) == 2
    triangles = [tuple(map(float, values)) for values in triangles]
    output_positions = []
    for names, label in ((("qo1", "qom"), "$y_t$"), (("qf1", "qfb"), r"$\widehat y_t$")):
        ends = []
        bank_y = []
        for name in names:
            node = re.search(rf"\({name}\) at {coordinate}", architecture)
            assert node
            bank_x, y = map(float, node.groups())
            branch = re.search(
                rf"\\draw\[ink,rounded corners=2pt\] \({name}\.east\)--{coordinate}--{coordinate};",
                architecture,
            )
            assert branch
            bend_x, bend_y, end_x, end_y = map(float, branch.groups())
            assert bank_x < bend_x < end_x
            assert bend_y == y
            bank_y.append(y)
            ends.append((end_x, end_y))
        assert ends[0] == ends[1]
        assert ends[0][1] == pytest.approx(np.mean(bank_y))
        triangle = [t for t in triangles if (t[0], (t[1] + t[3]) / 2) == ends[0]]
        assert len(triangle) == 1
        left, lower, upper_x, upper, tip_x, tip_y = triangle[0]
        assert left == upper_x
        assert tip_y == pytest.approx((lower + upper) / 2)
        assert tip_x - left == pytest.approx(0.38)
        assert upper - lower == pytest.approx(0.50)
        arrows = [tuple(map(float, coords)) for coords in re.findall(
            rf"\\draw\[flow,(?:blue|green)\] {coordinate}--{coordinate};", architecture,
        )]
        arrows = [arrow for arrow in arrows if arrow[:2] == (tip_x, tip_y)]
        assert len(arrows) == 1
        start_x, start_y, tip_x, tip_y = arrows[0]
        assert 0 < tip_x - start_x < 1.0
        assert tip_y == start_y
        output = re.search(
            rf"\\node\[anchor=west,text=(?:blue|green)\] at {coordinate} \{{{re.escape(label)}\}};",
            architecture,
        )
        assert output
        output_x, output_y = map(float, output.groups())
        assert tip_x < output_x < 12.40
        assert output_y == tip_y
        output_positions.append((output_x, output_y))
    brace = re.search(r"\\draw\[muted\] ([^;]+);", architecture)
    assert brace
    brace_points = np.asarray(re.findall(coordinate, brace[1]), dtype=float)
    x1, top = brace_points[0]
    x2, bottom = brace_points[-1]
    assert x1 == x2 == 12.40
    assert top > output_positions[0][1] > output_positions[1][1] > bottom
    connector = re.findall(
        rf"\\draw\[flow,muted\] {coordinate}--{coordinate};", architecture,
    )
    assert len(connector) == 1
    x0, y0, x1, y1 = map(float, connector[0])
    assert 12.40 < x0 < x1 < 13.525
    assert y0 == y1
    assert [x0, y0] == pytest.approx([12.61, 8.45])
    assert any(np.allclose(point, [x0, y0]) for point in brace_points)
    # Join the grouping brace at the open gap between the two case frames.
    assert 8.19 < y0 < 8.58
    assert "classical outputs; it is not a dynamical connection or quantum-state mixing" in architecture


def test_rounded_readouts_show_omitted_observables_only_after_measurement() -> None:
    source = CANONICAL.read_text()
    coordinate = r"\(([\d.]+),([\d.]+)\)"
    lane_style = r"\[muted!70,line width=\.55pt,rounded corners=1pt\]"
    lanes = re.findall(
        rf"\\draw{lane_style} {coordinate}--{coordinate}--{coordinate};", source,
    )
    assert len(lanes) == 6
    lanes = [tuple(map(float, lane)) for lane in lanes]
    for center, endpoint in ((11.10, "qo1"), (6.50, "qf1")):
        bank_lanes = [lane for lane in lanes if abs(lane[1] - center) < 0.31]
        assert len(bank_lanes) == 3
        for lane, offset in zip(bank_lanes, (-0.30, 0, 0.30)):
            start_x, start_y, bend_x, bend_y, end_x, end_y = lane
            assert (start_x, bend_x, end_x) == (10.80, 10.93, 11.04)
            assert start_y == pytest.approx(center + offset)
            assert bend_y == start_y
            assert end_y == pytest.approx(center + 0.4 * offset)
        first_lane = f"(10.80,{center - 0.30:.2f})"
        black_branch = rf"\draw[ink,rounded corners=2pt] ({endpoint}.east)"
        assert source.index(first_lane) < source.index(black_branch)
    assert source.count(
        r"\draw[ink,fill=gold,line width=.65pt,rounded corners=1.8pt]"
    ) == 2
    assert source.count(r"\draw[flow,ink,rounded corners=2pt]") == 4
    assert "illustrative count specifies neither a feature budget nor quantum wires" in source
    style = provenance(CANONICAL)["observable_readout_style"]
    assert style["readout_corner_radius_pt"] == 1.8
    assert style["readout_vertex_span_cm"] == [0.38, 0.50]
    assert style["omitted_observable_lanes_per_bank"] == 3
    assert style["omitted_lane_span_cm"] == [10.80, 11.04]


def test_two_output_timelines_share_axes_and_render_the_information_cut_at_expiry() -> None:
    source = CANONICAL.read_text()
    comparison = source.split("% Deterministic illustrative output overlays", 1)[1]
    plot = comparison.split(r"\newcommand{\memoryplot}[3]{%", 1)[1].split(
        r"\memoryplot{", 1,
    )[0]
    calls = re.findall(r"\\memoryplot\{([\d.]+)\}\{\\OutputSamples(Fading|Persistent)\}\{([01])\}", comparison)
    assert calls == [("8.58", "Fading", "0"), ("4.99", "Persistent", "1")]
    assert r"\begin{scope}[shift={(13.525,#1)}]" in plot
    assert plot.count(r"\draw[ink,line width=.6pt] (0,0) rectangle (4.15,3.20);") == 1
    metadata = output_provenance()
    assert metadata["axis_count"] == 2
    assert metadata["axes_size_cm"] == [4.15, 3.2]
    assert metadata["case_order"] == ["fading", "persistent"]
    assert metadata["axes_bounds_cm"] == {
        "fading": [13.525, 8.58, 17.675, 11.78],
        "persistent": [13.525, 4.99, 17.675, 8.19],
    }
    for bottom, case_title, ticks in calls:
        case = case_title.lower()
        bounds = metadata["cases"][case]["axes_bounds_cm"]
        np.testing.assert_allclose(bounds, [13.525, float(bottom), 17.675, float(bottom) + 3.2])
        assert metadata["cases"][case]["time_tick_labels_visible"] == bool(int(ticks))
    assert r"\foreach \val in {0.6,0.8,1.0,1.2}" in plot
    assert r"3.20*(\val-.45)/.9" in plot
    assert metadata["amplitude_limits"] == [0.45, 1.35]
    assert metadata["amplitude_ticks"] == [0.6, 0.8, 1.0, 1.2]
    assert metadata["time_limits"] == [-3, 103]
    assert metadata["time_ticks"] == [0, 40, 80]
    assert metadata["time_tick_labels"] == ["0", "d", "2d"]
    assert r"\ifnum#3=1" in plot
    tick_scope = plot.split(r"\ifnum#3=1", 1)[1].split(r"\fi", 1)[0]
    assert r"\foreach \kk/\lab in {0/0,40/d,80/2d}" in tick_scope
    assert r"\node[anchor=north]" in tick_scope
    assert r"{time $t$}" not in comparison
    assert r"{input step $k$}" not in source
    assert metadata["shared_time_label_anchor_cm"] is None
    transform = re.search(r"\\begin\{scope\}\[([^\]]*xshift=[^\]]+)\]", plot)
    assert transform
    scales = {key: float(value) for key, value in re.findall(
        r"(x|y|xshift|yshift)=(-?[\d.]+)cm", transform[1],
    )}
    assert set(scales) == {"x", "y", "xshift", "yshift"}
    x_shift, x_scale, y_scale, y_shift = (scales[key] for key in ("xshift", "x", "y", "yshift"))
    assert x_shift - 3 * x_scale == pytest.approx(0)
    assert x_shift + 103 * x_scale == pytest.approx(4.15)
    assert 0 < x_shift < x_shift + 100 * x_scale < 4.15
    assert y_shift + y_scale * 0.45 == pytest.approx(0)
    assert y_shift + y_scale * 1.35 == pytest.approx(3.2)
    for case in ("fading", "persistent"):
        visible_y = y_shift + y_scale * output_samples(case=case)
        # Native marker radii fit inside the same frame in both cases.
        assert np.all((visible_y > 0.06) & (visible_y < 3.2 - 0.06))
    assert r"\foreach \kk/\target/\prediction" in plot and r"in #2{" in plot
    assert r"\ifnum\kk=40" in plot
    assert r"(\pk,\pprediction)--(\kk,\target)--(\kk,\prediction);" in plot
    assert r"(\pk,\pprediction)--(\kk,\prediction);" in plot
    assert r"(\pk,\ptarget)--(\kk,\target);" in plot
    assert re.search(r"\\node\[fill=blue,circle,[^\]]+\] at \(\\kk,\\target\)", plot)
    assert re.search(r"\\node\[draw=green,fill=white,circle,[^\]]+\] at \(\\kk,\\prediction\)", plot)
    assert r"mod(\kk,2)" in plot
    assert r"inner sep=1.45pt] at (\kk,\prediction)" in plot
    assert plot.index(r"at (\kk,\prediction) {};") < plot.index(r"at (\kk,\target) {};")
    assert metadata["prediction_marker"]["every_nth_sample"] == 2
    assert metadata["prediction_marker"]["inner_sep_pt"] == 1.45
    artwork = re.sub(r"%[^\n]*", "", comparison)
    assert ".. controls" not in artwork and "smooth" not in artwork
    assert r"D_d/2" not in artwork
    cutoff = re.search(
        r"\\draw\[muted,densely dashed,[^\]]+\]\s*"
        r"\(\{([\d.]+)\*([\d.]+)/([\d.]+)\},0\)--"
        r"\(\{([\d.]+)\*([\d.]+)/([\d.]+)\},([\d.]+)\)", plot,
    )
    assert cutoff
    a, b, c, aa, bb, cc, height = map(float, cutoff.groups())
    assert a * b / c == pytest.approx(x_shift + 40 * x_scale)
    assert aa * bb / cc == pytest.approx(x_shift + 40 * x_scale)
    assert height == 3.2
    for fill in (
        r"\fill[green!24] (0,0) rectangle ({4.15*43/106},3.20);",
        r"\fill[blue!16] ({4.15*43/106},0) rectangle (4.15,3.20);",
    ):
        assert plot.count(fill) == 1
        assert plot.index(fill) < plot.index(r"\draw[plotgrid]") < plot.index(r"in #2{")
    cue = metadata["background_access_cue"]
    input_cue = provenance(CANONICAL)["past_state_cue"]
    assert cue["before_expiry"]["fill"] == input_cue["window_fill"] == "green!24"
    assert cue["before_expiry"]["time_extent"] == [-3, 40]
    assert cue["after_expiry"]["fill"] == input_cue["fill"] == "blue!16"
    assert cue["after_expiry"]["time_extent"] == [40, 103]
    assert cue["split_time"] == OUTPUT_WINDOW
    assert cue["split_x_local_cm"] == pytest.approx(x_shift + 40 * x_scale)
    assert "not the onset of QRC memory" in cue["scope"]
    assert "dashed guide and tick d" in cue["non_color_cue"]
    assert r"{old input\\leaves window}" not in comparison
    assert "{QRC retains its influence}" not in comparison
    assert r"\draw[flow,muted]" not in comparison
    assert metadata["legend_labels"] == ["QRC", "QELM"]
    assert "not an independent third forecasting target" in metadata["target_definition"]
    assert metadata["column_order"] == ["target", "prediction"]


def test_shared_legend_straddles_output_frames_without_internal_case_titles() -> None:
    source = CANONICAL.read_text()
    legend = source.split("% One shared reference-style model key", 1)[1]
    assert legend.count("{QRC}") == legend.count("{QELM}") == 1
    assert "(ideal)" not in legend
    assert r"\fill[blue]" in legend and r"\draw[green,fill=white" in legend
    assert r"circle (.070)" in legend
    assert r"\fill[ink!15,rounded corners=1.3pt]" in legend
    shift = re.search(r"\\begin\{scope\}\[shift=\{\(([\d.]+),([\d.]+)\)\}\]", legend)
    box = re.search(
        r"\\draw\[ink,fill=white,[^\]]+\] \(([\d.]+),([\d.]+)\) rectangle \(([\d.]+),([\d.]+)\)",
        legend,
    )
    assert shift and box
    dx, dy = map(float, shift.groups())
    left, bottom, right, top = map(float, box.groups())
    absolute_bounds = [left + dx, bottom + dy, right + dx, top + dy]
    metadata = output_provenance()
    np.testing.assert_allclose(absolute_bounds, metadata["legend_geometry"]["bounds_cm"])
    assert absolute_bounds == pytest.approx([14.0, 8.11, 17.27, 8.64])
    assert 13.525 < absolute_bounds[0] < absolute_bounds[2] < 17.675
    upper_bottom = metadata["axes_bounds_cm"]["fading"][1]
    lower_top = metadata["axes_bounds_cm"]["persistent"][3]
    assert absolute_bounds[1] < lower_top < upper_bottom < absolute_bounds[3]
    # Deliberate small border overlap signals that the one key serves both
    # panels; it does not introduce a second legend or cover their interiors.
    assert lower_top - absolute_bounds[1] == pytest.approx(0.08)
    assert absolute_bounds[3] - upper_bottom == pytest.approx(0.06)
    key_rows = [float(value) + dy for value in re.findall(
        r"at \([\d.]+,([\d.]+)\) \{(?:QRC|QELM)\}", legend,
    )]
    assert len(key_rows) == 2
    assert key_rows[0] == key_rows[1]
    assert key_rows[0] == pytest.approx(metadata["legend_geometry"]["row_y_cm"])
    assert all(absolute_bounds[1] < row < absolute_bounds[3] for row in key_rows)
    for case in ("fading", "persistent"):
        title = metadata["cases"][case]["title"].replace(" ", r"\\")
        assert metadata["cases"][case]["title_anchor_cm"] is None
        assert f"{{{title}}}" not in source


def test_case_labels_are_equal_sized_outside_frames_and_do_not_obscure_data() -> None:
    source = CANONICAL.read_text()
    label_marker = "% Case labels sit outside the data frames"
    assert source.index(label_marker) > source.index("% One shared reference-style model key")
    labels = source.split(label_marker, 1)[1]
    # Rectangles and text are separate: long wording must not enlarge a box
    # or place either label over the plotted trajectories.
    assert r"\foreach \yy/\case/\outcome in" in labels
    assert r"\begin{scope}[shift={(13.70,\yy)}]" in labels
    assert r"\draw[ink,fill=white,rounded corners=1.3pt,line width=.65pt] (0,0) rectangle (3.80,.68);" in labels
    assert r"\node[font=\fontsize{9}{10.5}\selectfont] at (1.90,.46) {\case};" in labels
    assert r"\node[text=muted,font=\fontsize{9}{10.5}\selectfont] at (1.90,.20) {\outcome};" in labels
    metadata = output_provenance()
    cases = metadata["case_labels"]
    expected = {
        "fading": (["Fading memory", "Window sufficient"], [13.70, 11.98, 17.50, 12.66]),
        "persistent": (["Persistent memory", "Window insufficient"], [13.70, 3.68, 17.50, 4.36]),
    }

    def separated(first, second):
        return (first[2] < second[0] or second[2] < first[0]
                or first[3] < second[1] or second[3] < first[1])

    for case, (lines, bounds) in expected.items():
        assert cases[case] == {"lines": lines, "bounds_cm": bounds}
        assert f"{bounds[1]:.2f}/{lines[0]}/{lines[1]}" in labels
        assert bounds[2] - bounds[0] == pytest.approx(3.80)
        assert bounds[3] - bounds[1] == pytest.approx(0.68)
        assert 0 < bounds[0] < bounds[2] < 17.9197
        assert 3.60 < bounds[1] < bounds[3] < 13.2
        assert all(separated(bounds, frame) for frame in metadata["axes_bounds_cm"].values())
        assert separated(bounds, metadata["legend_geometry"]["bounds_cm"])
    assert cases["fading"]["bounds_cm"][1] > metadata["axes_bounds_cm"]["fading"][3]
    assert cases["persistent"]["bounds_cm"][3] < metadata["axes_bounds_cm"]["persistent"][1]
    upper_gap = cases["fading"]["bounds_cm"][1] - metadata["axes_bounds_cm"]["fading"][3]
    lower_gap = metadata["axes_bounds_cm"]["persistent"][1] - cases["persistent"]["bounds_cm"][3]
    # Compact upper label; leave the lower box clear of the shared time ticks.
    assert upper_gap == pytest.approx(0.20)
    assert lower_gap == pytest.approx(0.63)
    assert "not a uniform claim for all histories" in cases["scope"]


def test_simplified_overview_retains_interface_distinctions() -> None:
    source = CANONICAL.read_text()
    # One shared input trace supplies a scalar to the QRC and its highlighted
    # causal window to the QELM; no duplicated lower-axis traversal remains.
    assert source.count(r"in \MGsamples{") == 1
    assert r"\windowtrace{" not in source
    assert r"\ifnum\kk=180 \coordinate (currentinput)" in source
    assert re.search(r"\\draw\[flow,purple,[^\]]+\] \(inputnode.east\)", source)
    assert "% The whole window, rather than its endpoint" in source
    assert r"$\Phi_{a_t}$" in source
    assert r"$E(S_d)$" in source
    labels = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", source).replace(r"\\", " ")
    assert "carried state" in labels and "{fresh state}" not in labels
    assert r"\node[anchor=east] at (6.35,\yy) {$|0\rangle$};" in source
    assert "{fixed gates}" not in labels
    # Removing an annotation must not remove or move either gate block.
    assert r"\draw[ink,fill=white,rounded corners=2pt,line width=.65pt] (7.77,6.21) rectangle (8.11,7.43);" in source
    assert r"\draw[ink,fill=white,rounded corners=2pt,line width=.65pt] (8.24,5.57) rectangle (8.58,6.79);" in source
    assert labels.count("{observables}") == 2
    assert "measured features" not in labels
    architecture = re.sub(
        r"%[^\n]*", "", source.split("% Deterministic illustrative output overlays", maxsplit=1)[0],
    )
    assert not re.search(
        r"\\node[^;]*\{[^{}]*(?:trained|readout)[^{}]*\};", architecture, re.I,
    )
    assert "declared output" not in architecture and "emulated output" not in architecture
    assert r"{$y_t$}" in architecture and r"{$\widehat y_t$}" in architecture
    assert r"\mathcal C" not in architecture
    assert source.count(r"{$\langle O_1\rangle$}") == 1
    assert source.count(r"{$\langle O_m\rangle$}") == 1
    assert source.count(r"{$\langle P_1\rangle$}") == 1
    assert source.count(r"{$\langle P_B\rangle$}") == 1
    assert "not equal-budget operator identities" in source
    # Both cases use the same causal bit-access construction.
    assert r"{$\Delta\rho$}" not in source
    assert "state 1" not in labels and "state 2" not in labels
    assert "can differ" not in labels and "identical" not in labels
    assert r"{old input leaves window}" not in labels
    assert r"{QRC retains its influence}" not in labels
    assert "{output}" not in labels
    assert r"\equaloutputs" not in source and r"\sharedstart" not in source
    assert "% c:" not in source
    core_labels = labels.split("% Case labels sit outside the data frames", 1)[0]
    assert "Fading memory" not in core_labels and "Persistent memory" not in core_labels
    assert labels.count("Fading memory") == labels.count("Persistent memory") == 1
    assert "best history error" not in labels and r"$D_d/2$" not in source
    assert "scalar optimum" not in labels
    assert "Window QELM" not in labels
    assert "desired error tolerance" not in labels
    assert "Same window, different pasts" not in labels and "at the midpoint" not in labels
    assert "{persistent floor}" not in labels and "{past influence fades}" not in labels
    assert "within tolerance" not in labels and r"$\varepsilon$" not in source
    # Preserve the scientific scope in the machine-readable source record.
    # Manuscript notation and theorem statements have their own focused tests;
    # panel-b wording is no longer part of the current drawing's contract.
    trace = provenance(CANONICAL)
    assert trace["sample_count"] == 201
    assert trace["selected_step"] == 180
    assert trace["window_indices"] == list(range(141, 181))
    assert trace["future_context_indices"] == list(range(181, 201))
    assert "not evaluation data" in trace["kind"]
    assert "target readout is fixed" in trace["architecture_scope"]
    assert "only the emulator readout is trained" in trace["architecture_scope"]
    assert "No specific topology or gate count" in trace["architecture_scope"]
    output = trace["output_comparisons"]
    assert output["case_order"] == ["fading", "persistent"]
    assert "same fixed d=40" in output["timeline_scope"]
    assert "not simulated, fitted, or measured QRC/QELM performance" in output["status"]
    assert "not an independent third forecasting target" in output["target_definition"]
    assert "not a fitted finite-feature QELM" in output["emulator_scope"]
    assert "cannot fit both histories exactly" in output["optimality_scope"]
    assert "specific balanced pair" in output["curve_gap_scope"]
    assert "not the uniform D_d limit" in output["cases"]["fading"]["decay_scope"]

def test_illustrative_input_is_reproducible_raw_mackey_glass() -> None:
    assert N_SAMPLES == 201
    assert WINDOW == 40
    assert SELECTED_STEP == 180 and WINDOW_START == 141
    assert SEED == 1
    assert PARAMETERS == {
        "tau": 17, "beta": 0.2, "gamma": 0.1, "n_exp": 10,
        "dt": 1.0, "sample_every": 1, "x0": 1.2, "discard": 1000,
    }
    assert INPUT_PARAMETERS == {**PARAMETERS, "dt": 0.5, "discard": 2000}
    expected = mackey_glass_series(
        201, rng=np.random.default_rng(1), **INPUT_PARAMETERS,
    )
    np.testing.assert_array_equal(samples(), expected)
    np.testing.assert_array_equal(samples(), samples())
    assert hashlib.sha256(expected.astype("<f8").tobytes()).hexdigest() == INPUT_SHA256
    legacy = mackey_glass_series(201, rng=np.random.default_rng(1), **PARAMETERS)
    assert hashlib.sha256(legacy[:120].astype("<f8").tobytes()).hexdigest() == LEGACY_INPUT_SHA256
    # In particular, the displayed values are not min-max normalized to [0, 1].
    assert expected.min() == pytest.approx(0.44887161450448193)
    assert expected.max() == pytest.approx(1.2803715455860114)
    assert expected[180] == pytest.approx(0.8069982550981397)
    window = expected[141:181]
    assert len(window) == 40
    assert window.min() == pytest.approx(0.44887161450448193)
    assert window.max() == pytest.approx(0.8876420562457414)
    assert len(expected[181:]) == 20


def test_quieter_input_sampling_leaves_the_separate_output_toys_unchanged(monkeypatch) -> None:
    legacy = mackey_glass_series(201, rng=np.random.default_rng(1), **PARAMETERS)
    quieter = samples()

    def peak_count(values):
        return np.count_nonzero((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))

    assert peak_count(quieter) == 4 < peak_count(legacy) == 8
    assert INPUT_PARAMETERS["dt"] * INPUT_PARAMETERS["discard"] == (
        PARAMETERS["dt"] * PARAMETERS["discard"]
    )
    baseline = output_baseline().copy()
    outputs = {case: output_samples(case).copy() for case in ("fading", "persistent")}
    assert hashlib.sha256(outputs["fading"][:, 0].astype("<f8").tobytes()).hexdigest() == FADING_TARGET_SHA256
    assert hashlib.sha256(outputs["persistent"].astype("<f8").tobytes()).hexdigest() == OUTPUT_SHA256
    monkeypatch.setitem(INPUT_PARAMETERS, "dt", 0.25)
    monkeypatch.setitem(INPUT_PARAMETERS, "discard", 4000)
    assert not np.array_equal(samples(), quieter)
    np.testing.assert_array_equal(output_baseline(), baseline)
    for case, expected in outputs.items():
        np.testing.assert_array_equal(output_samples(case), expected)


def test_tikz_contains_exact_samples_and_only_the_causal_forty_sample_window() -> None:
    source = CANONICAL.read_text()
    literal = source[source.index(BEGIN):source.index(END) + len(END)]
    assert literal == tikz_sample_block()
    pairs = re.findall(r"^(\d+)/([\d.eE+-]+),?$", literal, flags=re.MULTILINE)
    assert [int(index) for index, _ in pairs] == list(range(201))
    recorded = np.array([float(value) for _, value in pairs])
    np.testing.assert_array_equal(recorded, samples())
    assert r"\def\tracecolor{ink}" in source
    assert r"\ifnum\kk>140 \def\tracecolor{ink}\fi" in source
    assert r"\ifnum\kk>180 \def\tracecolor{muted!60}\fi" in source
    assert source.index(r"\ifnum\kk>140 \def\tracecolor{ink}") < source.index(
        r"\ifnum\kk>180 \def\tracecolor{muted!60}",
    )
    assert source.count(r"in \MGsamples{") == 1
    assert "% Lower square plot:" not in source
    brace = re.search(r"\\draw\[green,decorate,[^\n]+\]\s*([^;]+);", source)
    assert brace and "mirror" in brace[0]
    assert brace[1].strip() == "(windowleft)--(windowright)"
    brace_points = []
    for name in ("windowleft", "windowright"):
        point = re.search(
            rf"\\coordinate \({name}\) at "
            r"\(\{([\d.]+)\+([\d.]+)\*([\d.]+)/([\d.]+)\},([\d.]+)\)", source,
        )
        assert point
        brace_points.append(point.groups())
    assert len(brace_points) == 2
    actual_brace = [(a + b * c / d, y) for a, b, c, d, y in (
        map(float, point) for point in brace_points
    )]
    brace_indices = [(x - 1.045) * 212 / 4.15 - 6 for x, _ in actual_brace]
    # A half-step left edge encloses sample141; the right edge stops exactly
    # at the selected step180. Grey future samples never enter this bracket.
    assert brace_indices == pytest.approx([WINDOW_START - 0.5, SELECTED_STEP])
    assert actual_brace[0][1] == actual_brace[1][1] < 6.725
    assert r"(windowhook) at ($(windowleft)!.5!(windowright)+(0,-.11)$)" in source
    window_feed = re.search(r"\\draw\[flow,green,[^\]]+\] \(windowhook\)([^;]+);", source)
    assert window_feed
    assert "currentinput" not in window_feed[1] and "inputnode" not in window_feed[1]
    # The bracket, not a scalar sample, enters the reset encoding from below.
    encoding = re.search(
        r"\\draw\[ink,fill=green!13,[^\]]+\] "
        r"\(([\d.]+),([\d.]+)\) rectangle \(([\d.]+),([\d.]+)\)", source,
    )
    assert encoding
    encoding_left, encoding_bottom, encoding_right, _ = map(float, encoding.groups())
    feed_points = np.array(re.findall(r"\(([\d.]+),([\d.]+)\)", window_feed[1]), dtype=float)
    assert feed_points.shape == (2, 2)
    assert feed_points[-1, 0] == pytest.approx((encoding_left + encoding_right) / 2)
    assert feed_points[-1, 1] == encoding_bottom
    assert feed_points[-2, 1] < encoding_bottom
    selected = [k for k in range(201) if k > 140 and k < 181]
    assert selected == list(range(WINDOW_START, SELECTED_STEP + 1))
    assert len(selected) == 40
    assert not set(selected) & set(range(181, 201))
    metadata = provenance(CANONICAL)
    assert metadata["sample_count"] == 201
    assert metadata["selected_step"] == 180
    assert metadata["selected_value"] == recorded[180]
    assert metadata["window_start"] == 141
    assert metadata["window_end"] == 180
    assert metadata["window_length"] == 40
    assert metadata["window_indices"] == selected
    assert metadata["future_context_indices"] == list(range(181, 201))
    assert metadata["legacy_first_120_float64_le_sha256"] == LEGACY_INPUT_SHA256
    assert metadata["float64_le_sha256"] == INPUT_SHA256
    np.testing.assert_array_equal(metadata["values"], recorded)
    np.testing.assert_array_equal(metadata["window_values"], recorded[141:181])
    np.testing.assert_array_equal(metadata["future_context_values"], recorded[181:])
    assert "future display context only" in metadata["causal_scope"]
    assert "supplied to neither model" in metadata["causal_scope"]
    assert "not evaluation data" in metadata["kind"]
    assert "no QRC or QELM predictions" in metadata["scope"]
    geometry = metadata["plot_geometry"]
    for key, expected in {
        "canvas_cm": [17.9197, 9.6],
        "canvas_bounds_cm": [0, 3.6, 17.9197, 13.2],
        "input_axis_count": 1,
        "sample_axes_side_cm": 4.15,
        "full_input_axes_bounds_cm": [1.045, 6.725, 5.195, 10.875],
        "window_view": "highlighted and bracketed within the shared input axis; no separate duplicate plot",
        "output_axis_count": 2,
        "amplitude_limits": [0.3, 1.3],
        "amplitude_ticks": [0.4, 0.8, 1.2],
        "full_input_indices": [0, 200],
        "window_indices": [141, 180],
        "full_input_index_limits": [-6, 206],
        "window_bracket_index_extent": [140.5, 180],
    }.items():
        assert geometry[key] == expected
    assert geometry["output_axes_bounds_cm"] == output_provenance()["axes_bounds_cm"]
    assert "history_error_axes_bounds_cm" not in geometry
    assert "only the current scalar enters each update" in metadata["architecture_scope"]
    assert "No specific topology or gate count" in metadata["architecture_scope"]
    assert "Both models measure observables and apply a readout" in metadata["architecture_scope"]
    assert "target readout is fixed" in metadata["architecture_scope"]
    assert "only the emulator readout is trained" in metadata["architecture_scope"]
    assert "general architectures" in metadata["figure_notation"]
    assert "O_1,...,O_m label target observables" in metadata["figure_notation"]
    assert "P_1,...,P_B label measured QELM features, including checkpoints" in metadata["figure_notation"]
    assert "m and B denote output dimension and measured-feature count" in metadata["figure_notation"]
    assert "output labels are y_t and widehat y_t" in metadata["figure_notation"]
    assert "not identical operators or an equal-budget claim" in metadata["figure_notation"]
    assert "small yellow readout triangles" in metadata["figure_notation"]
    assert "history_error_curve" not in metadata
    assert metadata["output_comparisons"] == output_provenance()

def test_one_larger_input_plot_keeps_raw_values_and_causal_context() -> None:
    source = CANONICAL.read_text().split("% Deterministic illustrative output overlays", 1)[0]
    boxes = re.findall(
        r"\\draw\[ink,line width=\.6pt\] "
        r"\(([\d.]+),([\d.]+)\) rectangle \(([\d.]+),([\d.]+)\);",
        source,
    )
    assert len(boxes) == 1
    assert np.asarray(boxes, dtype=float) == pytest.approx(np.asarray([
        [1.045, 6.725, 5.195, 10.875],
    ]))
    left, bottom, right, top = map(float, boxes[0])
    assert right - left == pytest.approx(4.15)
    assert top - bottom == pytest.approx(4.15)
    input_source = source
    assert input_source.count("y=4.15cm]") == 1
    assert input_source.count(r"in \MGsamples{") == 1
    artwork = re.sub(r"%[^\n]*", "", source)
    assert not re.search(r"\{(?:zoom|schematic)\}", artwork)
    tick_lists = re.findall(r"\\foreach \\val in \{([\d.,]+)\}", input_source)
    assert [[float(value) for value in ticks.split(",")] for ticks in tick_lists] == [
        [0.4, 0.8, 1.2],
    ]
    assert input_source.count(r"4.15*(\val-.3)") == 1
    assert r"{$\val$}" not in input_source and r"{$\kk$}" not in input_source
    assert input_source.count(r"{\val};") == 1
    assert r"\foreach \kk in {0,100}" in input_source
    assert r"\foreach \kk in {141,160}" not in input_source
    selected_ticks = re.findall(
        r"\\node\[anchor=north,text=purple\] at "
        r"\(\{([\d.]+)\+([\d.]+)\*([\d.]+)/([\d.]+)\},([\d.]+)\) \{\$t\$\}", input_source,
    )
    selected_tick_coords = [(a + b * c / d, y) for a, b, c, d, y in (
        map(float, point) for point in selected_ticks
    )]
    np.testing.assert_allclose(selected_tick_coords, [
        [1.045 + 4.15 * (180 + 6) / 212, 6.615],
    ])
    assert re.search(r"\\node\[fill=\\tracecolor,circle,[^\]]+\] at \(\\kk,\\val\)",
                     input_source)
    # Only straight segments connect the numerical samples. The curved history
    # annotation and separate schematic curves are not numerical traces.
    trace_scopes = re.findall(
        r"\\begin\{scope\}\[shift=\{\([\d.]+,[^\n]+\n(.*?)\\end\{scope\}",
        input_source, re.S,
    )
    assert len(trace_scopes) == 1
    assert all(r"in \MGsamples{" in trace for trace in trace_scopes)
    assert all(".. controls" not in trace for trace in trace_scopes)
    assert "smooth" not in re.sub(r"%[^\n]*", "", input_source)
    transforms = re.findall(
        r"shift=\{\(([\d.]+),([\d.]+)\)\},x=([\d.]+)cm,y=([\d.]+)cm",
        input_source,
    )
    assert len(transforms) == 1
    x_origin, origin, x_scale, scale = map(float, transforms[0])
    assert x_origin + x_scale * -6 == pytest.approx(left)
    assert x_origin + x_scale * 206 == pytest.approx(right)
    assert left < x_origin < x_origin + x_scale * 200 < right
    assert origin + scale * 0.3 == pytest.approx(bottom)
    assert origin + scale * 1.3 == pytest.approx(top)
    # Enlargement changes the physical display size, never the source values.
    values = samples()
    assert np.all((values > 0.3) & (values < 1.3))
    np.testing.assert_allclose(
        origin + scale * values, 6.725 + 4.15 * (values - 0.3),
    )
    # At native 11pt the ket and its inner padding fit in a conservative 0.60cm
    # envelope. Leave that envelope right of the input frame instead of relying
    # on visual overlap being hidden by a filled gate or a white label.
    reset_source = source.split("% Fresh reset,", maxsplit=1)[1]
    ket = re.search(r"\\node\[anchor=east\] at \(([\d.]+),\\yy\) \{\$\|0\\rangle\$\}", reset_source)
    assert ket and float(ket[1]) - 0.60 - right >= 0.20
    reset_region = re.search(
        r"\\fill\[neutral,[^\]]+\] \(([\d.]+),([\d.]+)\) rectangle", reset_source,
    )
    assert reset_region and float(reset_region[1]) > right


def test_spacing_refinement_preserves_type_and_opens_model_rows() -> None:
    source = CANONICAL.read_text()
    rows = re.findall(r"\\foreach \\yy in \{([\d.,]+)\}", source)
    assert len(rows) == 2
    qrc, reset = (list(map(float, row.split(","))) for row in rows)
    assert qrc[1] - reset[1] >= 4.60
    bounds = provenance(CANONICAL)["plot_geometry"]
    left, bottom, right, top = bounds["full_input_axes_bounds_cm"]
    assert (bottom + top) / 2 == pytest.approx((qrc[1] + reset[1]) / 2)
    assert (left + right) / 2 == pytest.approx(3.12)
    assert r"font=\fontsize{11}{12.5}\selectfont" in source
    assert r"rectangle (17.9197,13.2)" in source


def test_reference_palette_and_quantum_processing_motifs_are_explicit() -> None:
    source = CANONICAL.read_text()
    colors = dict(re.findall(r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-F]{6})\}", source))
    assert {name: colors[name] for name in ("ink", "blue", "green", "orange", "gold", "purple")} == {
        "ink": "20252B", "blue": "D97632", "green": "3868D9",
        "orange": "D97632", "gold": "FFC943", "purple": "20252B",
    }
    metadata = provenance(CANONICAL)
    assert metadata["diagram_encoding"] == {
        "earlier_input": "#20252B",
        "carried_state": "#D97632",
        "explicit_window": "#3868D9",
        "classical_input_injection": "#20252B",
        "measured_quantities_and_readouts": "#FFC943",
        "target_output": "#D97632",
        "ideal_window_output": "#3868D9",
    }
    wire_rows = re.findall(r"\\foreach \\yy in \{([\d.,]+)\}", source)
    assert len(wire_rows) == 2
    assert all(len(row.split(",")) == 3 for row in wire_rows)
    assert source.count(r"\fill[palegold") == 2
    assert source.count(r"{$\langle O_1\rangle$}") == 1
    assert source.count(r"{$\langle O_m\rangle$}") == 1
    assert source.count(r"{$\langle P_1\rangle$}") == 1
    assert source.count(r"{$\langle P_B\rangle$}") == 1
    assert r"{$y_t$}" in source
    assert r"{$\widehat y_t$}" in source


def test_input_provenance_rejects_a_changed_sample(tmp_path: Path) -> None:
    source = CANONICAL.read_text()
    changed = tmp_path / "changed.tex"
    first_sample = f"0/{samples()[0]:.17g}"
    assert first_sample in source
    changed.write_text(source.replace(first_sample, "0/0.5", 1))
    with pytest.raises(ValueError, match="differ from the declared generator"):
        provenance(changed)


def test_shared_legend_leaves_all_output_ink_at_least_two_tenths_mm_clear() -> None:
    source = CANONICAL.read_text()
    metadata = output_provenance()
    assert r"line width=.70pt,inner sep=1.45pt" in source
    assert r"\node[fill=blue,circle,inner sep=.78pt]" in source
    assert r"rounded corners=1.3pt,line width=.75pt" in source
    assert r"\draw[blue,line width=.55pt] (\pk,\ptarget)--(\kk,\target);" in source
    assert r"\draw[green,line width=.65pt]" in source

    # 1/72 inch is slightly larger than a TeX point (1/72.27 inch), so these
    # radii conservatively include the empty TikZ circle's diagonal padding
    # and the open ring's half-stroke. Use the union of the key and its shadow.
    pt_cm = 2.54 / 72
    ring_radius = (np.sqrt(2) * 1.45 + 0.70 / 2) * pt_cm
    dot_radius = np.sqrt(2) * 0.78 * pt_cm
    key_half_stroke = 0.75 / 2 * pt_cm
    left, bottom, right, top = metadata["legend_geometry"]["bounds_cm"]
    shadow_dx, shadow_dy = metadata["legend_geometry"]["shadow_offset_cm"]
    key_left = min(left - key_half_stroke, left + shadow_dx)
    key_right = max(right + key_half_stroke, right + shadow_dx)
    key_bottom = min(bottom - key_half_stroke, bottom + shadow_dy)
    key_top = max(top + key_half_stroke, top + shadow_dy)
    x_min, x_max = metadata["time_limits"]
    y_min, y_max = metadata["amplitude_limits"]
    width, height = metadata["axes_size_cm"]
    minimum_clearance_cm = 0.02

    for case in metadata["case_order"]:
        bounds = metadata["cases"][case]["axes_bounds_cm"]
        values = output_samples(case)
        times = np.arange(len(values))
        x = bounds[0] + width * (times - x_min) / (x_max - x_min)
        y = bounds[1] + height * (values - y_min) / (y_max - y_min)

        def check_clearance(points, radius, *, connected=False):
            if connected:
                # Every straight segment lies between its endpoint extrema.
                # Include both endpoints of every segment whose ink can enter
                # the key's horizontal span, including the discrete expiry jump.
                first, last = points[:-1], points[1:]
                overlap = (
                    (np.maximum(first[:, 0], last[:, 0]) + radius >= key_left)
                    & (np.minimum(first[:, 0], last[:, 0]) - radius <= key_right)
                )
                ordinates = np.concatenate((first[overlap, 1], last[overlap, 1]))
            else:
                overlap = (
                    (points[:, 0] + radius >= key_left)
                    & (points[:, 0] - radius <= key_right)
                )
                ordinates = points[overlap, 1]
            assert ordinates.size
            clearance = (
                ordinates.min() - radius - key_top
                if case == "fading"
                else key_bottom - ordinates.max() - radius
            )
            assert clearance >= minimum_clearance_cm, (case, radius, clearance)

        target_points = np.column_stack((x, y[:, 0]))
        prediction_points = np.column_stack((x, y[:, 1]))
        check_clearance(target_points, dot_radius)
        check_clearance(prediction_points[::2], ring_radius)
        check_clearance(target_points, 0.55 / 2 * pt_cm, connected=True)
        expiry = metadata["memory_access"]["first_excluded_step"]
        prediction_path = np.insert(
            prediction_points, expiry, [x[expiry], y[expiry, 0]], axis=0,
        )
        check_clearance(prediction_path, 0.65 / 2 * pt_cm, connected=True)
