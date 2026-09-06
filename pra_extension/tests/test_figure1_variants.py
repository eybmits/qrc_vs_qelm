from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

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
    N_SAMPLES,
    OUTPUT_BEGIN,
    OUTPUT_END,
    OUTPUT_MEMORY_AMPLITUDE,
    OUTPUT_N_SAMPLES,
    OUTPUT_SEED,
    OUTPUT_WINDOW,
    OUTPUT_WRITE_STEP,
    PARAMETERS,
    SEED,
    SELECTED_STEP,
    WINDOW,
    WINDOW_START,
    output_baseline,
    output_provenance,
    output_samples,
    provenance,
    samples,
    tikz_output_sample_block,
    tikz_sample_block,
)
from qrc_vs_qelm.core.tasks import mackey_glass_series  # noqa: E402

DESIGNS = ROOT / "paper/figures/figure1_variants"
CANONICAL = ROOT / "paper/figures/information_interface/fig01_information_interface.tex"
COMPACT_ASPECT = 17.9197 / 13.0
INPUT_SHA256 = "bf2e22ef87be4390ad5bd7e084ad8b39ef8337e1707cf1f957d11b285bc7a9f8"
LEGACY_INPUT_SHA256 = "279ffd9bff72e717ab0a77396efdd92c4495e57d417f65f2bdddcc05680e25ba"
OUTPUT_SHA256 = "375f0f9f1116189c9a9b3d7875a23ff7d0b92c04440dcf5dc0932c3021216faf"
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
    font_sizes = [float(size) for size in re.findall(r"\\fontsize\{([\d.]+)\}", source)]
    assert font_sizes
    assert min(font_sizes) >= 9.5
    assert max(font_sizes) <= 12
    main = (ROOT / "paper/canonical_emulation/main.tex").read_text()
    introduction = (ROOT / "paper/canonical_emulation/sections/introduction.tex").read_text()
    # The manuscript retains its canonical artifact path; archived alternatives
    # do not introduce duplicate figure inclusions after the user's selection.
    assert name not in main + introduction
    assert "fig01_information_interface.pdf" in introduction


def test_selected_architecture_source_is_the_canonical_overview() -> None:
    manifest = json.loads((DESIGNS / "design_manifest.json").read_text())
    selected = DESIGNS / f"{manifest['selected_variant']}.tex"
    assert CANONICAL.read_bytes() == selected.read_bytes()
    assert ROOT / manifest["canonical_source"] == CANONICAL
    digest = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
    assert manifest["canonical_source_sha256"] == digest
    assert manifest["variants"]["variant_a_architectures"]["artifacts"][
        "variant_a_architectures.tex"
    ] == digest


def test_canonical_overview_uses_letters_not_narrative_panel_titles() -> None:
    source = CANONICAL.read_text()
    panel_nodes = re.findall(
        r"\\node\[([^\]]*)\]\s+at\s+\([^;]*?\)\s+\{([abc])\};",
        source,
    )
    assert [label for _, label in panel_nodes] == ["a", "b"]
    bold_styles = set(re.findall(r"([\w-]+)/\.style\s*=\s*\{[^}]*\\bfseries", source))
    assert all(
        r"\bfseries" in options
        or bool({option.strip() for option in options.split(",")} & bold_styles)
        for options, _ in panel_nodes
    )
    assert not re.search(r"\\(?:sub)?title\{", source)


def test_selected_overview_keeps_larger_native_labels_and_measurement_symbols() -> None:
    source = CANONICAL.read_text()
    assert r"font=\fontsize{11}{12.5}\selectfont" in source
    assert r"font=\bfseries\fontsize{12}{13}\selectfont" in source
    font_sizes = [float(size) for size in re.findall(r"\\fontsize\{([\d.]+)\}", source)]
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
    # Queue the full-width overview in the introduction so it can appear on
    # page 2; the compiled PDF still needs final-size visual placement review.
    before_figure = introduction.split(r"\begin{figure*}", maxsplit=1)[0]
    assert len(before_figure.splitlines()) <= 15


def test_compact_canvas_keeps_native_geometry_without_scaling() -> None:
    source = CANONICAL.read_text()
    assert "x=1cm,y=1cm" in source
    assert r"\useasboundingbox (0,0) rectangle (17.9197,13.0);" in source
    assert not re.search(r"(?:^|[\[,\s])(?:xscale|yscale|scale)\s*=", source)


def test_selected_b4_panel_distinguishes_tolerance_from_persistent_error() -> None:
    complete_source = CANONICAL.read_text()
    assert re.search(r"\\begin\{scope\}\[xshift=4cm\]\s*% b:", complete_source)
    source = complete_source.split("% b:", maxsplit=1)[1]
    coordinate = r"\(([\d.]+),([\d.]+)\)"
    axes = re.findall(
        rf"\\draw\[flow,ink\] {coordinate}--{coordinate};", source,
    )
    assert len(axes) == 2
    horizontal, vertical = (tuple(map(float, axis)) for axis in axes)
    origin_x, origin_y, right, bottom = horizontal
    assert bottom == origin_y
    assert vertical[:3] == pytest.approx((origin_x, origin_y, origin_x))
    assert right - origin_x == pytest.approx(6.40)
    assert vertical[3] - origin_y == pytest.approx(2.40)

    band = re.search(
        rf"\\fill\[green!8\] {coordinate} rectangle {coordinate};", source,
    )
    assert band
    left, lower, band_right, tolerance = map(float, band.groups())
    assert (left, lower) == pytest.approx((origin_x, origin_y))
    assert origin_x < band_right < right
    assert origin_y < tolerance < vertical[3]
    threshold = re.search(
        rf"\\draw\[muted,densely dashed,[^\]]+\] {coordinate}--{coordinate};",
        source,
    )
    assert threshold
    assert tuple(map(float, threshold.groups())) == pytest.approx(
        (origin_x, tolerance, band_right, tolerance),
    )

    fading = re.search(r"\\draw\[blue,line width=[^\]]+\](.*?);", source, re.S)
    assert fading
    assert ".. controls" not in fading[1]
    assert fading[1].count("plot") == 1
    domain = re.search(r"domain=([\d.]+):([\d.]+),samples=(\d+)", fading[1])
    assert domain
    start, stop = map(float, domain.groups()[:2])
    assert (start, stop) == pytest.approx((1.66, 7.52))
    assert int(domain[3]) == 120
    number = r"(?:\d+(?:\.\d*)?|\.\d+)"
    expression = re.search(
        rf"\(\\x,\{{({number})\+({number})\*exp\(-({number})\*\(\\x-({number})\)\)\}}\)",
        re.sub(r"\s+", "", fading[1]),
    )
    assert expression
    baseline, amplitude, decay, offset = map(float, expression.groups())
    assert baseline == origin_y
    assert amplitude > 0 and decay > 0
    assert offset == start
    x = np.linspace(start, stop, int(domain[3]))
    y = baseline + amplitude * np.exp(-decay * (x - offset))
    assert np.all(np.diff(y) < 0)
    # One analytic function has no join kink and tends to the zero-error axis.
    assert np.all(-decay * (y - baseline) < 0)

    orange = re.search(r"\\draw\[orange,line width=[^\]]+\](.*?);", source, re.S)
    assert orange
    persistent = [tuple(map(float, point)) for point in re.findall(coordinate, orange[1])]
    assert len(persistent) >= 4
    assert np.all(np.diff([xx for xx, _ in persistent]) > 0)
    assert np.all(np.diff([yy for _, yy in persistent]) < 0)
    assert (start, y[0]) == pytest.approx(persistent[0])
    assert origin_y < y[-1] < tolerance < persistent[-1][1]
    crossing = re.search(rf"\\fill\[green\] {coordinate} circle", source)
    assert crossing
    crossing_point = tuple(map(float, crossing.groups()))
    assert crossing_point == pytest.approx((4.76, 1.47))
    assert crossing_point[1] == tolerance
    assert start < crossing_point[0] < stop
    expected_decay = np.log(amplitude / (tolerance - baseline)) / (crossing_point[0] - offset)
    assert decay == pytest.approx(expected_decay, abs=1e-14)
    assert baseline + amplitude * np.exp(-decay * (crossing_point[0] - offset)) == pytest.approx(
        tolerance, abs=1e-12,
    )
    assert "not the first strict integer d_epsilon" in source
    artwork = re.sub(r"%[^\n]*", "", source)
    assert r"d_\varepsilon" not in artwork
    assert r"d_{\varepsilon}" not in artwork
    assert "{crossing}" in artwork


def test_output_toy_is_reproducible_and_obeys_the_balanced_bit_information_cut(monkeypatch) -> None:
    assert OUTPUT_N_SAMPLES == 101 and OUTPUT_SEED == 2
    assert OUTPUT_WINDOW == 40 and OUTPUT_WRITE_STEP == 0
    assert OUTPUT_MEMORY_AMPLITUDE == 0.25
    k = np.arange(101, dtype=float)
    raw = mackey_glass_series(101, rng=np.random.default_rng(2), **PARAMETERS)
    baseline = 0.4 + 0.65 * (raw - raw.min()) / (raw.max() - raw.min())
    target = baseline + 0.25
    other_history = baseline - 0.25
    prediction = baseline + 0.25 * (k < 40)
    expected = np.column_stack((target, prediction))
    np.testing.assert_array_equal(output_baseline(), baseline)
    np.testing.assert_array_equal(output_samples(), expected)
    assert hashlib.sha256(expected.astype("<f8").tobytes()).hexdigest() == OUTPUT_SHA256
    assert baseline.min() == pytest.approx(0.4)
    assert baseline.max() == pytest.approx(1.05)
    np.testing.assert_array_equal(prediction[:40], target[:40])
    np.testing.assert_allclose(prediction[40:], (target[40:] + other_history[40:]) / 2)
    np.testing.assert_allclose(target[40:] - prediction[40:], 0.25)
    np.testing.assert_allclose(prediction[40:] - other_history[40:], 0.25)
    # These windows include the bit exactly through time39, not at time40.
    for time in range(101):
        available_steps = range(time - OUTPUT_WINDOW + 1, time + 1)
        assert (OUTPUT_WRITE_STEP in available_steps) == (time < 40)
    # The balanced scalar pair makes the midpoint both Bayes-optimal and
    # minimax; its lower bound concerns both histories, not one chosen trace.
    pair_mse = 0.5 * ((target[40:] - prediction[40:]) ** 2 + (other_history[40:] - prediction[40:]) ** 2)
    np.testing.assert_allclose(pair_mse, 0.25 ** 2)
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
    np.testing.assert_array_equal(metadata["baseline_values"], baseline)
    np.testing.assert_array_equal(metadata["other_history_values"], other_history)
    access = metadata["memory_access"]
    assert access["bit_values"] == [-1, 1] and access["bit_probabilities"] == [0.5, 0.5]
    assert access["displayed_bit"] == 1
    assert access["write_step"] == 0 and access["first_excluded_step"] == 40
    assert access["window_length"] == 40 and access["memory_amplitude"] == 0.25
    assert access["post_expiry_pair_gap"] == 0.5
    assert access["post_expiry_pair_midpoint_error"] == 0.25
    assert "identical later inputs" in access["shared_future"]
    assert "independent of time and later inputs" in access["shared_future"]
    assert "cannot fit both histories exactly" in metadata["optimality_scope"]
    assert "not a fitted finite-feature QELM" in metadata["emulator_scope"]
    assert metadata["units"] == "arbitrary time and output units"


def test_tikz_contains_exact_two_column_output_toy_block() -> None:
    source = CANONICAL.read_text()
    literal = source[source.index(OUTPUT_BEGIN):source.index(OUTPUT_END) + len(OUTPUT_END)]
    assert literal == tikz_output_sample_block()
    rows = re.findall(r"^(\d+)/([\d.eE+-]+)/([\d.eE+-]+),?$", literal, re.M)
    assert [int(row[0]) for row in rows] == list(range(101))
    recorded = np.array([row[1:] for row in rows], dtype=float)
    np.testing.assert_array_equal(recorded, output_samples())
    metadata = provenance(CANONICAL)["output_comparisons"]
    assert metadata == output_provenance()
    assert metadata["column_order"] == ["target", "prediction"]
    assert metadata["float64_le_sha256"] == OUTPUT_SHA256
    assert metadata["sample_indices"] == list(range(101))
    np.testing.assert_array_equal(metadata["values"], recorded)


@pytest.mark.parametrize("change", ("altered", "duplicated", "missing"))
def test_output_provenance_rejects_corrupted_blocks(tmp_path: Path, change: str) -> None:
    source = CANONICAL.read_text()
    block = tikz_output_sample_block()
    replacement = {
        "altered": block.replace(f"0/{output_samples()[0, 0]:.17g}", "0/0.1", 1),
        "duplicated": block + "\n" + block,
        "missing": "",
    }[change]
    changed = tmp_path / "changed.tex"
    changed.write_text(source.replace(block, replacement, 1))
    with pytest.raises(ValueError, match="output"):
        provenance(changed)


def test_overview_adds_whitespace_without_shrinking_its_contents() -> None:
    source = CANONICAL.read_text()
    architecture = source.split("% b:", maxsplit=1)[0]
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
    assert scalar_node and float(scalar_node[1]) > 5.32 and float(scalar_node[2]) > 11.0
    assert r"\ifnum\kk=180 \coordinate (currentinput)" in qrc_source
    selected_x = 0.92 + 4.4 * (SELECTED_STEP + 6) / 212
    assert 0.92 < selected_x < 5.32
    guide = re.search(r"\\draw\[muted,densely dashed,[^\]]+\] ([^;]+);", qrc_source)
    assert guide
    guide_points = re.findall(
        r"\(\{([\d.]+)\+([\d.]+)\*([\d.]+)/([\d.]+)\},([\d.]+)\)", guide[1],
    )
    assert len(guide_points) == 2
    actual_guide = [(a + b * c / d, y) for a, b, c, d, y in (
        map(float, point) for point in guide_points
    )]
    np.testing.assert_allclose(actual_guide, [(selected_x, 6.60), (selected_x, 11.0)])
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
        (0.92 + 4.4 * 6 / 212, 11.14),
        (0.92 + 4.4 * 146.5 / 212, 11.14),
    ])
    assert r"past in $\rho_{t-1}$" in qrc_source
    assert "incomingstate" not in qrc_source
    channel = re.search(
        r"\\draw\[ink,fill=blue!12,[^\]]+\]\s*"
        r"\(([\d.]+),([\d.]+)\) rectangle \(([\d.]+),([\d.]+)\)", qrc_source,
    )
    assert channel
    channel_left, channel_bottom, channel_right, _ = map(float, channel.groups())
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
        assert "The QRC carries state" in caption
        assert "bracketed green window" in caption
        assert "grey continuation" in caption.lower() and "excluded" in caption.lower()
        assert re.search(r"target readout (?:is |stays )?fixed during emulation", caption.lower())


def test_observable_banks_feed_small_readouts_and_one_shared_comparison() -> None:
    architecture = CANONICAL.read_text().split("% b:", maxsplit=1)[0]
    coordinate = r"\(([\d.]+),([\d.]+)\)"
    triangles = re.findall(
        rf"\\draw\[ink,fill=gold,[^\]]+\] {coordinate}--{coordinate}--{coordinate}--cycle;",
        architecture,
    )
    assert len(triangles) == 2
    triangles = [tuple(map(float, values)) for values in triangles]
    output_positions = []
    for names, label in ((("qo1", "qom"), "$Y_t$"), (("qf1", "qfb"), r"$\widehat Y_t$")):
        ends = []
        bank_y = []
        for name in names:
            node = re.search(rf"\({name}\) at {coordinate}", architecture)
            assert node
            bank_x, y = map(float, node.groups())
            branch = re.search(
                rf"\\draw\[ink\] \({name}\.east\)--{coordinate}--{coordinate};",
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
            rf"\\draw\[flow,ink\] {coordinate}--{coordinate};", architecture,
        )]
        arrows = [arrow for arrow in arrows if arrow[:2] == (tip_x, tip_y)]
        assert len(arrows) == 1
        start_x, start_y, tip_x, tip_y = arrows[0]
        assert 0 < tip_x - start_x < 1.0
        assert tip_y == start_y
        output = re.search(
            rf"\\node\[anchor=west\] at {coordinate} \{{{re.escape(label)}\}};",
            architecture,
        )
        assert output
        output_x, output_y = map(float, output.groups())
        assert tip_x < output_x < 12.40
        assert output_y == tip_y
        output_positions.append((output_x, output_y))
    brace = re.search(
        rf"\\draw\[muted,decorate,[^\n]+\] {coordinate}--{coordinate};", architecture,
    )
    assert brace
    x1, top, x2, bottom = map(float, brace.groups())
    assert x1 == x2 == 12.40
    assert top > output_positions[0][1] > output_positions[1][1] > bottom
    connector = re.findall(
        rf"\\draw\[flow,muted\] {coordinate}--{coordinate};", architecture,
    )
    assert len(connector) == 1
    x0, y0, x1, y1 = map(float, connector[0])
    assert 12.40 < x0 < x1 < 13.40
    assert y0 == y1
    assert 6.60 < y0 < 11.00  # connector enters the single output comparison
    assert "classical outputs; it is not a dynamical connection or quantum-state mixing" in architecture


def test_one_output_timeline_renders_the_discrete_information_cut_at_expiry() -> None:
    source = CANONICAL.read_text()
    comparison = source.split("% Deterministic illustrative output overlays", 1)[1].split(
        "% Panel b", 1,
    )[0]
    assert r"\foreach \base/\mode/\case" not in comparison
    assert r"\begin{scope}[shift={(13.40,6.60)}]" in comparison
    assert comparison.count(r"(0,0) rectangle (4.4,4.4)") == 1
    metadata = output_provenance()
    assert metadata["axis_count"] == 1
    np.testing.assert_allclose(metadata["axes_bounds_cm"], [13.40, 6.60, 17.80, 11.00])
    assert r"\foreach \val in {0.4,0.8,1.2}" in comparison
    assert r"4.4*(\val-.3)/1.2" in comparison
    assert r"\foreach \kk/\lab in {0/0,40/d,80/2d}" in comparison
    assert r"4.4*(\kk+3)/106" in comparison
    assert r"{$\val$}" not in comparison and r"{$\kk$}" not in comparison
    assert comparison.count(r"{time $t$}") == 1
    transform = re.search(r"\\begin\{scope\}\[([^\]]*xshift=[^\]]+)\]", comparison)
    assert transform
    scales = {key: float(value) for key, value in re.findall(
        r"(x|y|xshift|yshift)=(-?[\d.]+)cm", transform[1],
    )}
    assert set(scales) == {"x", "y", "xshift", "yshift"}
    x_shift, x_scale, y_scale, y_shift = (scales[key] for key in ("xshift", "x", "y", "yshift"))
    assert x_shift + -3 * x_scale == pytest.approx(0)
    assert x_shift + 103 * x_scale == pytest.approx(4.4)
    assert 0 < x_shift < x_shift + 100 * x_scale < 4.4
    assert y_shift + y_scale * 0.3 == pytest.approx(0)
    assert y_shift + y_scale * 1.5 == pytest.approx(4.4)
    assert np.all((output_samples() > 0.3) & (output_samples() < 1.5))
    assert comparison.count(r"in \OutputSamples{") == 1
    assert r"\foreach \kk/\target/\prediction" in comparison
    assert r"\ifnum\kk=40" in comparison
    assert r"(\pk,\pprediction)--(\kk,\target)--(\kk,\prediction);" in comparison
    assert r"(\pk,\pprediction)--(\kk,\prediction);" in comparison
    assert r"(\pk,\ptarget)--(\kk,\target);" in comparison
    assert re.search(r"\\node\[fill=ink,circle,[^\]]+\] at \(\\kk,\\target\)", comparison)
    assert re.search(r"\\node\[draw=green,fill=white,circle,[^\]]+\] at \(\\kk,\\prediction\)", comparison)
    artwork = re.sub(r"%[^\n]*", "", comparison)
    assert ".. controls" not in artwork and "smooth" not in artwork
    cutoff = re.search(
        r"\\draw\[muted,densely dashed,[^\]]+\]\s*"
        r"\(\{([\d.]+)\*([\d.]+)/([\d.]+)\},0\)--"
        r"\(\{([\d.]+)\*([\d.]+)/([\d.]+)\},([\d.]+)\)", comparison,
    )
    assert cutoff
    a, b, c, aa, bb, cc, height = map(float, cutoff.groups())
    assert a * b / c == pytest.approx(x_shift + 40 * x_scale)
    assert aa * bb / cc == pytest.approx(x_shift + 40 * x_scale)
    assert 0 < height < 4.4
    assert r"{old input\\leaves window}" in comparison
    legend = comparison.split("% Compact inset key", 1)[1]
    assert "{QRC}" in legend and "{QELM}" in legend
    assert r"\fill[ink]" in legend and r"\draw[green,fill=white" in legend
    assert r"\draw[green,line width=.65pt]" in legend
    legend_box = re.search(
        r"\\draw\[ink,fill=white,[^\]]+\] \(([\d.]+),([\d.]+)\) rectangle \(([\d.]+),([\d.]+)\)",
        legend,
    )
    assert legend_box
    _, legend_bottom, _, _ = map(float, legend_box.groups())
    assert y_shift + y_scale * np.max(output_samples()) < legend_bottom
    assert r"font=\fontsize{11}{12.5}\selectfont" in source
    assert r"font=\bfseries\fontsize{12}{13}\selectfont" in source


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
    assert "carried state" in labels and "fresh state" in labels
    assert "fixed gates" in labels
    assert labels.count("{observables}") == 2
    assert "measured features" not in labels
    architecture = re.sub(r"%[^\n]*", "", source.split("% b:", maxsplit=1)[0])
    assert not re.search(
        r"\\node[^;]*\{[^{}]*(?:trained|readout)[^{}]*\};", architecture, re.I,
    )
    assert "declared output" not in architecture and "emulated output" not in architecture
    assert r"{$Y_t$}" in architecture and r"{$\widehat Y_t$}" in architecture
    assert r"\mathcal C" not in architecture
    assert source.count(r"{$\langle O_1\rangle$}") == 2
    assert source.count(r"{$\langle O_m\rangle$}") == 2
    assert "O_i are model-specific schematic labels, not equal-budget operator identities" in source
    # A single causal toy replaces unrelated state-pair or mismatch panels.
    assert r"{$\Delta\rho$}" not in source
    assert "state 1" not in labels and "state 2" not in labels
    assert "can differ" not in labels and "identical" not in labels
    assert r"{old input leaves window}" in labels
    assert r"{QRC retains its influence}" in labels
    assert r"\equaloutputs" not in source and r"\sharedstart" not in source
    assert "% c:" not in source
    assert "best possible history error" in labels
    assert "scalar optimum" in labels and r"$D_d/2$" in source
    assert "persistent floor" in labels and "past influence fades" in labels
    assert "within tolerance" in labels and r"$\varepsilon$" in source
    for relative in (
        "paper/canonical_emulation/sections/introduction.tex",
        "paper/icaart2027/sections/introduction.tex",
    ):
        text = (ROOT / relative).read_text()
        caption = " ".join(text.split(r"\caption{", 1)[1].split(r"\label{", 1)[0].split())
        # The compact caption states the mechanism; model details and trace
        # provenance belong in the accompanying model section, not the artwork.
        assert len(caption.split()) <= 130
        assert "model-specific observables" in caption
        assert "trainable readouts" in caption
        assert re.search(r"target readout (?:is |stays )?fixed during emulation", caption.lower())
        assert "$D_d$ is the largest output gap between histories" in caption
        assert re.search(r"sharing (?:the|their) last \$d\$ inputs but differing earlier", caption)
        assert "For scalar outputs, $D_d/2$" in caption
        assert "smallest worst-case absolute error" in caption
        assert "any window predictor" in caption
        assert r"$\varepsilon$" in caption and "floor" in caption
        assert "schematic" in caption
        assert "ideal window predictor" in caption
        assert "not measured QELM performance" in caption
        assert "two possible values" in caption and "output midpoint" in caption
        assert "one history" in caption and "leaves the window" in caption
        assert "mismatch or close agreement" not in caption
        assert "Alternative input-created state pairs" not in caption
        assert not re.search(r"(?:\\textbf\{c\}|\bc)\s+Alternative", caption)
        models = " ".join((ROOT / relative).with_name("models.tex").read_text().split())
        assert "201 illustrative Mackey--Glass samples" in models
        assert "$t=180$" in models and "inputs 141--180" in models
        assert "Grey future samples are excluded" in models
        assert "separate output example" in models and "two histories" in models
        assert "differing only in one old bit" in models
        assert "identical later inputs" in models
        assert "ideal window predictor" in models and "output midpoint" in models
        assert "Only one history is drawn" in models
        assert "40-input window" in models
        assert re.search(r"not a (?:trained|fitted) QELM", models)
        assert "baseline is independent of the left trace" in models
        assert "neither plot is evaluation data" in models.lower()
        assert "independent 33-symbol inputs" in models
        assert "dissipative QRC channels" in models
        assert "target readout is the identity" in models
        assert "once trained and held fixed" in models
        assert "absorbing its weights into the observables" in models
        assert "does not require reconstruction of the full quantum state" in models


def test_illustrative_input_is_reproducible_raw_mackey_glass() -> None:
    assert N_SAMPLES == 201
    assert WINDOW == 40
    assert SELECTED_STEP == 180 and WINDOW_START == 141
    assert SEED == 1
    assert PARAMETERS == {
        "tau": 17, "beta": 0.2, "gamma": 0.1, "n_exp": 10,
        "dt": 1.0, "sample_every": 1, "x0": 1.2, "discard": 1000,
    }
    expected = mackey_glass_series(
        201, rng=np.random.default_rng(1), **PARAMETERS,
    )
    np.testing.assert_array_equal(samples(), expected)
    np.testing.assert_array_equal(samples(), samples())
    assert hashlib.sha256(expected.astype("<f8").tobytes()).hexdigest() == INPUT_SHA256
    assert hashlib.sha256(expected[:120].astype("<f8").tobytes()).hexdigest() == LEGACY_INPUT_SHA256
    # In particular, the displayed values are not min-max normalized to [0, 1].
    assert expected.min() == pytest.approx(0.38764364216757924)
    assert expected.max() == pytest.approx(1.2840136891526057)
    assert expected[180] == pytest.approx(1.0811295768379747)
    window = expected[141:181]
    assert len(window) == 40
    assert window.min() == pytest.approx(0.43105017534770285)
    assert window.max() == pytest.approx(1.0811295768379747)
    assert len(expected[181:]) == 20


def test_tikz_contains_exact_samples_and_only_the_causal_forty_sample_window() -> None:
    source = CANONICAL.read_text()
    literal = source[source.index(BEGIN):source.index(END) + len(END)]
    assert literal == tikz_sample_block()
    pairs = re.findall(r"^(\d+)/([\d.eE+-]+),?$", literal, flags=re.MULTILINE)
    assert [int(index) for index, _ in pairs] == list(range(201))
    recorded = np.array([float(value) for _, value in pairs])
    np.testing.assert_array_equal(recorded, samples())
    assert r"\def\tracecolor{ink}" in source
    assert r"\ifnum\kk>140 \def\tracecolor{green}\fi" in source
    assert r"\ifnum\kk>180 \def\tracecolor{muted!60}\fi" in source
    assert source.index(r"\ifnum\kk>140 \def\tracecolor{green}") < source.index(
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
    brace_indices = [(x - 0.92) * 212 / 4.4 - 6 for x, _ in actual_brace]
    # A half-step left edge encloses sample141; the right edge stops exactly
    # at the selected step180. Grey future samples never enter this bracket.
    assert brace_indices == pytest.approx([WINDOW_START - 0.5, SELECTED_STEP])
    assert actual_brace[0][1] == actual_brace[1][1] < 6.60
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
    assert metadata["plot_geometry"] == {
        "canvas_cm": [17.9197, 13.0],
        "input_axis_count": 1,
        "sample_axes_side_cm": 4.40,
        "full_input_axes_bounds_cm": [0.92, 6.60, 5.32, 11.0],
        "window_view": "highlighted and bracketed within the shared input axis; no separate duplicate plot",
        "history_error_axes_cm": [6.40, 2.40],
        "output_axis_count": 1,
        "output_axes_side_cm": 4.40,
        "output_axes_bounds_cm": [13.40, 6.60, 17.80, 11.00],
        "amplitude_limits": [0.3, 1.3],
        "amplitude_ticks": [0.4, 0.8, 1.2],
        "full_input_indices": [0, 200],
        "window_indices": [141, 180],
        "full_input_index_limits": [-6, 206],
        "window_bracket_index_extent": [140.5, 180],
    }
    assert "only the current scalar enters each update" in metadata["architecture_scope"]
    assert "No specific topology or gate count" in metadata["architecture_scope"]
    assert "Both models measure observables and apply a readout" in metadata["architecture_scope"]
    assert "target readout is fixed" in metadata["architecture_scope"]
    assert "only the emulator readout is trained" in metadata["architecture_scope"]
    assert "general architectures" in metadata["figure_notation"]
    assert "model-specific schematic observable labels" in metadata["figure_notation"]
    assert "not identical operators or an equal-budget claim" in metadata["figure_notation"]
    assert "small gold readout triangles" in metadata["figure_notation"]
    curve = metadata["history_error_curve"]
    assert curve["kind"] == "analytic conceptual exponential; not an input-data fit"
    assert curve["local_y_expression"] == "0.85+2.19*exp(-0.40707656282948684*(x-1.66))"
    assert curve["local_domain_cm"] == [1.66, 7.52]
    assert curve["panel_translation_cm"] == [4.0, 0.0]
    assert curve["render_samples"] == 120
    assert curve["zero_error_canvas_y_cm"] == 0.85
    assert curve["local_tolerance_crossing_cm"] == [4.76, 1.47]
    assert curve["canvas_tolerance_crossing_cm"] == [8.76, 1.47]
    assert "straight segments without smoothing" in curve["scope"]


def test_one_larger_input_plot_keeps_raw_values_and_causal_context() -> None:
    source = CANONICAL.read_text().split("% Deterministic illustrative output overlays", 1)[0]
    boxes = re.findall(
        r"\\draw\[ink,line width=\.6pt\] "
        r"\(([\d.]+),([\d.]+)\) rectangle \(([\d.]+),([\d.]+)\);",
        source,
    )
    assert len(boxes) == 1
    assert np.asarray(boxes, dtype=float) == pytest.approx(np.asarray([
        [0.92, 6.60, 5.32, 11.0],
    ]))
    left, bottom, right, top = map(float, boxes[0])
    assert right - left == pytest.approx(4.40)
    assert top - bottom == pytest.approx(4.40)
    input_source = source.split("% b:", maxsplit=1)[0]
    assert input_source.count("y=4.4cm]") == 1
    assert input_source.count(r"in \MGsamples{") == 1
    artwork = re.sub(r"%[^\n]*", "", source)
    assert not re.search(r"\{(?:zoom|schematic)\}", artwork)
    tick_lists = re.findall(r"\\foreach \\val in \{([\d.,]+)\}", input_source)
    assert [[float(value) for value in ticks.split(",")] for ticks in tick_lists] == [
        [0.4, 0.8, 1.2],
    ]
    assert input_source.count(r"4.4*(\val-.3)") == 1
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
        [0.92 + 4.4 * (180 + 6) / 212, 6.49],
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
        origin + scale * values, 6.60 + 4.4 * (values - 0.3),
    )
    # At native 11pt the ket and its inner padding fit in a conservative 0.60cm
    # envelope. Leave that envelope right of the input frame instead of relying
    # on visual overlap being hidden by a filled gate or a white label.
    reset_source = source.split("% Fresh reset,", maxsplit=1)[1]
    ket = re.search(r"\\node\[anchor=east\] at \(([\d.]+),\\yy\) \{\$\|0\\rangle\$\}", reset_source)
    assert ket and float(ket[1]) - 0.60 > right
    reset_region = re.search(
        r"\\fill\[neutral,[^\]]+\] \(([\d.]+),([\d.]+)\) rectangle", reset_source,
    )
    assert reset_region and float(reset_region[1]) > right


def test_reference_palette_and_quantum_processing_motifs_are_explicit() -> None:
    source = CANONICAL.read_text()
    colors = dict(re.findall(r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-F]{6})\}", source))
    assert {name: colors[name] for name in ("ink", "blue", "green", "orange", "gold", "purple")} == {
        "ink": "000000", "blue": "3B86FF", "green": "00A14B",
        "orange": "FC4D30", "gold": "FFC943", "purple": "A93BFF",
    }
    metadata = provenance(CANONICAL)
    assert metadata["diagram_encoding"] == {
        "earlier_input": "#000000",
        "carried_state": "#3B86FF",
        "explicit_window": "#00A14B",
        "classical_input_injection": "#A93BFF",
        "measured_quantities_and_readouts": "#FFC943",
        "persistent_gap": "#FC4D30",
    }
    wire_rows = re.findall(r"\\foreach \\yy in \{([\d.,]+)\}", source)
    assert len(wire_rows) == 2
    assert all(len(row.split(",")) == 3 for row in wire_rows)
    assert source.count(r"\fill[palegold") == 2
    assert source.count(r"{$\langle O_1\rangle$}") == 2
    assert source.count(r"{$\langle O_m\rangle$}") == 2
    assert r"{$Y_t$}" in source
    assert r"{$\widehat Y_t$}" in source


def test_input_provenance_rejects_a_changed_sample(tmp_path: Path) -> None:
    source = CANONICAL.read_text()
    changed = tmp_path / "changed.tex"
    changed.write_text(source.replace("0/0.38764364216757924", "0/0.5", 1))
    with pytest.raises(ValueError, match="differ from the declared generator"):
        provenance(changed)
