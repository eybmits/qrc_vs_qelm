from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "pra_extension" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from render_information_interface_figures import (  # noqa: E402
    DEFAULT_DARK_SUMMARY,
    DEFAULT_DENSE_PHASE_SOURCE,
    DEFAULT_DENSE_SUFFIX_SOURCE,
    DEFAULT_DENSE_N4_PHASE_SOURCE,
    DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    DEFAULT_PHASE_SOURCE,
    FIGURE_SIZES,
    PALETTE,
    STYLE,
    _base_style,
    _resource_diagnosis_axes,
    _validated_dense_phase_source,
    _validated_dense_n4_phase_source,
    _validated_dense_merged_source,
    render_all,
    simplified_phase_table,
)


EXPECTED_ARTIFACTS = {
    "fig01_information_interface.pdf",
    "fig01_information_interface.png",
    "fig01_information_interface_source.json",
    "fig02_information_frontier.pdf",
    "fig02_information_frontier.png",
    "fig02_information_frontier_source.csv",
    "fig03_resource_diagnosis.pdf",
    "fig03_resource_diagnosis.png",
    "fig03_resource_diagnosis_source.csv",
}


EXPECTED_STYLE = {
    "font_family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "font_size": 9.5,
    "axes_labelsize": 10.0,
    "axes_titlesize": 10.5,
    "tick_labelsize": 9.0,
    "legend_fontsize": 9.0,
    "axes_linewidth": 0.65,
    "primary_linewidth": 1.3,
    "grid_linewidth": 0.45,
}

EXPECTED_FIGURE_SIZES = {
    "fig1": (7.055, 4.40),
    "fig2": (7.055, 6.10),
    "fig3": (7.055, 3.50),
}

EXPECTED_PALETTE = {
    "ink": "#000000",
    "annotation": "#7F7F7F",
    "grid": "#B0B0B0",
    "neutral": "#F1F2F2",
    "history_missing": "#A7A9AC",
    "frozen": "#3B86FF",
    "frozen_light": "#EBF2FF",
    "onset": "#00A14B",
    "onset_light": "#E6F5EC",
    "scrambled": "#FC4D30",
    "scrambled_light": "#FEECE9",
    "yellow": "#FFC943",
    "pale_yellow": "#FFDD92",
    "orange": "#FF8431",
    "purple": "#A93BFF",
    "viridis_18": "#7AD151",
    "viridis_36": "#2A788E",
    "viridis_72": "#482475",
    "family_2": "#253494",
    "family_3": "#E8590C",
    "story_ink": "#20242A",
    "story_border": "#CBD5E1",
    "story_neutral": "#F8FAFC",
    "story_blue": "#3572C6",
    "story_blue_light": "#EDF3FC",
    "story_green": "#169B62",
    "story_green_light": "#EBF7F1",
    "story_orange": "#D97716",
    "story_orange_light": "#FFF1DF",
    "white": "#FFFFFF",
}


def test_resource_panels_share_one_row():
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=FIGURE_SIZES["fig3"])
    try:
        boxes = [ax.get_position() for ax in _resource_diagnosis_axes(fig)]
        assert len(boxes) == 3
        assert [b.y0 for b in boxes] == pytest.approx([boxes[0].y0] * 3)
        assert [b.height for b in boxes] == pytest.approx([boxes[0].height] * 3)
        assert [b.width for b in boxes] == pytest.approx([boxes[0].width] * 3)
        assert all(left.x1 < right.x0 for left, right in zip(boxes, boxes[1:]))
        assert all(0 <= b.x0 < b.x1 <= 1 and 0 <= b.y0 < b.y1 <= 1 for b in boxes)
    finally:
        plt.close(fig)


def _pdf_media_box(path: Path) -> tuple[float, float]:
    """Return the first PDF page width and height in PostScript points."""

    # TikZ stores the page object in a compressed stream; inspect the PDF using
    # Poppler rather than relying on Matplotlib's uncompressed object layout.
    info = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True
    ).stdout
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    match = re.search(
        r"Page size:\s*(" + number + r")\s*x\s*(" + number + r")\s+pts",
        info,
    )
    if match is None:
        raise AssertionError(f"{path.name} has no readable page dimensions")
    width, height = (float(value) for value in match.groups())
    return width, height


def test_reference_figure_style_contract_is_frozen() -> None:
    assert STYLE == EXPECTED_STYLE
    assert PALETTE == EXPECTED_PALETTE
    assert FIGURE_SIZES == EXPECTED_FIGURE_SIZES
    assert {width for width, _ in FIGURE_SIZES.values()} == {7.055}
    _base_style()
    assert list(matplotlib.rcParams["font.family"]) == STYLE["font_family"]
    assert matplotlib.rcParams["font.size"] == STYLE["font_size"]
    assert matplotlib.rcParams["axes.labelsize"] == STYLE["axes_labelsize"]
    assert matplotlib.rcParams["axes.titlesize"] == STYLE["axes_titlesize"]
    assert matplotlib.rcParams["xtick.labelsize"] == STYLE["tick_labelsize"]
    assert matplotlib.rcParams["ytick.labelsize"] == STYLE["tick_labelsize"]
    assert matplotlib.rcParams["legend.fontsize"] == STYLE["legend_fontsize"]
    assert matplotlib.rcParams["axes.linewidth"] == STYLE["axes_linewidth"]


def test_native_tikz_story_preserves_readable_labels_and_emulation_scope() -> None:
    tikz_source = (
        REPO_ROOT / "paper" / "figures" / "information_interface"
        / "fig01_information_interface.tex"
    ).read_text(encoding="utf-8")
    assert r"\usepackage{helvet}" in tikz_source
    assert r"\renewcommand{\familydefault}{\sfdefault}" in tikz_source
    font_sizes = [float(size) for size in re.findall(r"\\fontsize\{([\d.]+)\}", tikz_source)]
    assert font_sizes
    assert min(font_sizes) >= 9.5
    assert max(font_sizes) == 10.5
    assert r"\resizebox" not in tikz_source
    for required_story_label in (
        "Can finite context replace recurrent quantum memory?",
        "Which stored memory matters?",
        "When is replacement possible?",
        "carried state",
        "same reset state for each window",
        "Past inputs can create it",
        "Declared outputs can reveal it",
        "now or after later inputs",
        "Relevant memory fades uniformly",
        "universal reset-QELM family",
        "Visible memory persists",
        "for every finite window",
        "schematic",
    ):
        assert required_story_label in tikz_source


def test_plot_labels_use_plain_language_and_separate_resource_questions() -> None:
    renderer_source = (
        CODE_ROOT / "render_information_interface_figures.py"
    ).read_text(encoding="utf-8")
    assert r"\mathbf" not in renderer_source
    assert '"?"' not in renderer_source
    assert 'label="root split"' not in renderer_source
    assert 'label="unresolved"' not in renderer_source
    assert 'label="map limited"' not in renderer_source
    for removed_row_title in (
        "How much old history remains?",
        "Which reset-QELM reaches the target?",
    ):
        assert removed_row_title not in renderer_source
    for required_plot_label in (
        "Smallest successful feature count",
        "no tested QELM succeeds",
        "history target unmet",
    ):
        assert required_plot_label in renderer_source


def test_committed_figure_pdfs_preserve_width_aspect_and_font_contract() -> None:
    figure_dir = REPO_ROOT / "paper" / "figures" / "information_interface"
    pdfs = {
        "fig1": figure_dir / "fig01_information_interface.pdf",
        "fig2": figure_dir / "fig02_information_frontier.pdf",
        "fig3": figure_dir / "fig03_resource_diagnosis.pdf",
    }
    for key, path in pdfs.items():
        payload = path.read_bytes()
        assert b"dejavu" not in payload.lower(), path.name
        width_points, height_points = _pdf_media_box(path)
        target_width, target_height = FIGURE_SIZES[key]
        assert width_points / 72 == pytest.approx(target_width, abs=0.005)
        assert height_points / 72 == pytest.approx(target_height, abs=0.005)
        assert width_points / height_points == pytest.approx(
            target_width / target_height,
            rel=0.005,
        )


def test_information_interface_figures_are_complete_traceable_and_deterministic(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = render_all(output_dir=first_dir)
    second = render_all(output_dir=second_dir)

    assert set(first) == EXPECTED_ARTIFACTS
    assert first == second
    ledger = json.loads(
        (first_dir / "figure_sha256.json").read_text(encoding="utf-8")
    )
    assert ledger == first
    assert "figure_sha256.json" not in ledger
    for name, expected in first.items():
        path = first_dir / name
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if name.endswith(".pdf"):
            assert path.stat().st_size > 10_000
        if name.endswith(".png"):
            assert path.stat().st_size > 50_000

    story = json.loads(
        (first_dir / "fig01_information_interface_source.json").read_text(
            encoding="utf-8"
        )
    )
    assert story["figure_question"] == (
        "when can explicit finite context reproduce recurrent input-output behavior"
    )
    assert story["panels"] == {
        "a": "recurrent quantum state versus reset-window input-output emulation",
        "b": "input-reachable and future-output-visible memory directions",
        "c": "uniformly fading relevant memory versus persistent visible memory",
    }
    assert "future declared outputs" in story["memory_relevance"]
    assert "full-state forgetting is not required" in story["memory_relevance"]
    tikz_path = (
        REPO_ROOT / "paper" / "figures" / "information_interface"
        / "fig01_information_interface.tex"
    )
    assert story["tikz_source_sha256"] == hashlib.sha256(tikz_path.read_bytes()).hexdigest()
    assert story["typography"] == "native 9.5-10.5 pt text at 7.055 inch width"
    assert story["schematic_curves"] is True
    assert story["regimes"]["finite"]["condition"] == "D_d = 0 for some finite d"
    assert "positive at every finite d" in story["regimes"]["fading"]["condition"]
    assert "r_vis < 1" in story["regimes"]["fading"]["condition"]
    assert "Delta/2" in story["regimes"]["persistent"]["consequence"]
    assert "D_d tends to zero" in story["emulation_statement"]
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        story["upstream_dark_sector_summary_sha256"],
    )
    assert not {
        "balanced_mse_floor",
        "fixed_pair_construction",
        "output_gap",
        "suffix_scope",
    }.intersection(story)

    phase = pd.read_csv(first_dir / "fig02_information_frontier_source.csv")
    assert len(phase) == 198
    assert phase["evidence_cohort"].value_counts().to_dict() == {
        "frozen_dense_n2_n3": 132,
        "prospective_dense_n4_followup": 66,
    }
    assert phase["state"].value_counts().to_dict() == {
        "HISTORY_MISSING": 138,
        "FEASIBLE": 38,
        "UNRESOLVED": 21,
        "MAP_NOT_REPRESENTED": 1,
    }
    assert int(phase["any_root_disagreement"].sum()) == 77
    assert (phase["history_n_roots"] == 12).all()
    assert int(phase["history_n_roots"].sum()) == 2_376
    assert phase["display_outcome"].value_counts().to_dict() == {
        "HISTORY_NOT_ROBUST": 141,
        "QELM_36": 28,
        "NO_TESTED_QELM_ROBUST": 19,
        "QELM_18": 7,
        "QELM_72": 3,
    }
    assert (phase["history_root_count"] == 12).all()
    assert int(phase["history_root_count"].sum()) == 2_376
    assert phase["median_history_nrmse"].min() == pytest.approx(
        0.00110218849454
    )
    assert phase["median_history_nrmse"].max() == pytest.approx(
        0.874121688735
    )

    resources = pd.read_csv(first_dir / "fig03_resource_diagnosis_source.csv")
    assert len(resources) == 600
    assert resources["panel"].value_counts().to_dict() == {
        "history": 336,
        "shots": 192,
        "features": 72,
    }
    assert set(resources.loc[resources["panel"].eq("features"), "metric"]) == {
        "total_nrmse"
    }


def test_simplified_phase_table_preserves_registered_cell_grid() -> None:
    phase = simplified_phase_table(DEFAULT_PHASE_SOURCE)
    assert set(phase["teacher_n_qubits"]) == {2, 3}
    assert set(phase["teacher_gamma"]) == {0.0, 0.2, 0.5, 1.0, 2.0, 4.0}
    assert set(phase["window"]) == {1, 2, 4, 6}
    feasible = phase[phase["state"].eq("FEASIBLE")]
    counts = feasible.groupby(
        ["teacher_n_qubits", "minimum_feature_budget"]
    ).size()
    assert counts.to_dict() == {
        (2, 36.0): 3,
        (2, 72.0): 2,
        (3, 18.0): 1,
        (3, 36.0): 2,
        (3, 72.0): 2,
    }


def test_dense_phase_source_preserves_fresh_complete_grid() -> None:
    phase = _validated_dense_phase_source(
        DEFAULT_DENSE_PHASE_SOURCE,
        DEFAULT_DENSE_SUFFIX_SOURCE,
    )
    assert set(phase["teacher_n_qubits"]) == {2, 3}
    assert set(phase["teacher_gamma"]) == {
        0.0,
        0.2,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
    }
    assert set(phase["window"]) == {1, 2, 3, 4, 5, 6}
    assert phase["run_manifest_sha256"].nunique() == 1
    assert phase["root_set_sha256"].nunique() == 1
    assert phase.loc[
        phase["state"].eq("FEASIBLE"),
        ["emulated_n_B18", "emulated_n_B36", "emulated_n_B72"],
    ].max(axis=1).ge(10).all()
    robust_history = phase["history_sufficient_n"].ge(10)
    assert phase.loc[robust_history, "display_outcome"].ne(
        "HISTORY_NOT_ROBUST"
    ).all()
    assert phase.loc[~robust_history, "display_outcome"].eq(
        "HISTORY_NOT_ROBUST"
    ).all()
    excluded = phase[
        phase["teacher_n_qubits"].eq(3)
        & phase[["teacher_gamma", "window"]].apply(
            tuple,
            axis=1,
        ).isin({(1.0, 6), (1.25, 5), (1.5, 4)})
    ]
    assert excluded["display_outcome"].eq("HISTORY_NOT_ROBUST").all()


def test_dense_n4_source_preserves_separate_prospective_grid() -> None:
    phase = _validated_dense_n4_phase_source(
        DEFAULT_DENSE_N4_PHASE_SOURCE,
        DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    )
    assert len(phase) == 66
    assert set(phase["teacher_n_qubits"]) == {4}
    assert set(phase["teacher_gamma"]) == {
        0.0,
        0.2,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
    }
    assert set(phase["window"]) == {1, 2, 3, 4, 5, 6}
    assert phase["evidence_cohort"].eq("prospective_dense_n4_followup").all()
    assert phase["history_root_count"].eq(12).all()
    assert int(phase["history_root_count"].sum()) == 792
    assert phase["display_outcome"].value_counts().to_dict() == {
        "HISTORY_NOT_ROBUST": 47,
        "QELM_36": 10,
        "NO_TESTED_QELM_ROBUST": 7,
        "QELM_18": 2,
    }


def test_merged_dense_source_keeps_cohort_provenance_separate() -> None:
    phase = _validated_dense_merged_source(
        DEFAULT_DENSE_PHASE_SOURCE,
        DEFAULT_DENSE_SUFFIX_SOURCE,
        DEFAULT_DENSE_N4_PHASE_SOURCE,
        DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    )
    assert len(phase) == 198
    assert not phase.duplicated(
        ["teacher_n_qubits", "teacher_gamma", "window"]
    ).any()
    assert phase["evidence_cohort"].value_counts().to_dict() == {
        "frozen_dense_n2_n3": 132,
        "prospective_dense_n4_followup": 66,
    }
    for cohort in phase["evidence_cohort"].unique():
        selected = phase[phase["evidence_cohort"].eq(cohort)]
        assert selected["run_manifest_sha256"].nunique() == 1
        assert selected["root_set_sha256"].nunique() == 1
        assert selected["config_sha256"].nunique() == 1
    assert phase.groupby("evidence_cohort")["config_sha256"].first().nunique() == 2
    assert phase.groupby("evidence_cohort")["root_set_sha256"].first().nunique() == 2


def test_feature_panel_omits_only_uniformly_history_insufficient_low_loss_cells() -> None:
    phase = _validated_dense_merged_source(
        DEFAULT_DENSE_PHASE_SOURCE,
        DEFAULT_DENSE_SUFFIX_SOURCE,
        DEFAULT_DENSE_N4_PHASE_SOURCE,
        DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    )
    omitted = phase.loc[phase["teacher_gamma"].le(1.0)]
    assert len(omitted) == 90
    assert set(omitted["teacher_n_qubits"]) == {2, 3, 4}
    assert set(omitted["window"]) == {1, 2, 3, 4, 5, 6}
    assert omitted["display_outcome"].eq("HISTORY_NOT_ROBUST").all()
    assert omitted["history_sufficient_n"].lt(10).all()
    # Their measurements remain in the full figure source and upper curves.
    archived = pd.read_csv(
        REPO_ROOT / "paper" / "figures" / "information_interface"
        / "fig02_information_frontier_source.csv"
    )
    assert len(archived.loc[archived["teacher_gamma"].le(1.0)]) == len(omitted)


def test_feature_frontier_keeps_real_nonmonotonic_window_outcomes() -> None:
    phase = _validated_dense_merged_source(
        DEFAULT_DENSE_PHASE_SOURCE,
        DEFAULT_DENSE_SUFFIX_SOURCE,
        DEFAULT_DENSE_N4_PHASE_SOURCE,
        DEFAULT_DENSE_N4_SUFFIX_SOURCE,
    )
    cells = phase.loc[phase["teacher_gamma"].eq(2.0) & phase["window"].isin((5, 6))]
    assert len(cells) == 6
    assert cells["history_sufficient_n"].eq(12).all()
    outcomes = cells.set_index(["teacher_n_qubits", "window"])["display_outcome"]
    assert outcomes.to_dict() == {
        (2, 5): "QELM_72",
        (2, 6): "NO_TESTED_QELM_ROBUST",
        (3, 5): "QELM_72",
        (3, 6): "NO_TESTED_QELM_ROBUST",
        (4, 5): "QELM_36",
        (4, 6): "NO_TESTED_QELM_ROBUST",
    }
    archived = pd.read_csv(
        REPO_ROOT / "paper" / "figures" / "information_interface"
        / "fig02_information_frontier_source.csv"
    )
    archived_cells = archived.loc[
        archived["teacher_gamma"].eq(2.0) & archived["window"].isin((5, 6))
    ]
    assert archived_cells.set_index(
        ["teacher_n_qubits", "window"]
    )["display_outcome"].to_dict() == outcomes.to_dict()


def test_dense_phase_validator_rejects_missing_cell_and_mixed_run(tmp_path: Path) -> None:
    phase = pd.read_csv(DEFAULT_DENSE_PHASE_SOURCE)
    missing = tmp_path / "missing.csv"
    phase.iloc[:-1].to_csv(missing, index=False)
    with pytest.raises(ValueError, match="132 unique cells"):
        _validated_dense_phase_source(missing)

    mixed = phase.copy()
    mixed.loc[0, "run_manifest_sha256"] = "0" * 64
    mixed_path = tmp_path / "mixed.csv"
    mixed.to_csv(mixed_path, index=False)
    with pytest.raises(ValueError, match="run_manifest_sha256"):
        _validated_dense_phase_source(mixed_path)


def test_dense_n4_validator_rejects_missing_cell_and_mixed_run(tmp_path: Path) -> None:
    phase = pd.read_csv(DEFAULT_DENSE_N4_PHASE_SOURCE)
    missing = tmp_path / "missing_n4.csv"
    phase.iloc[:-1].to_csv(missing, index=False)
    with pytest.raises(ValueError, match="66 unique cells"):
        _validated_dense_n4_phase_source(missing)

    mixed = phase.copy()
    mixed.loc[0, "run_manifest_sha256"] = "0" * 64
    mixed_path = tmp_path / "mixed_n4.csv"
    mixed.to_csv(mixed_path, index=False)
    with pytest.raises(ValueError, match="run_manifest_sha256"):
        _validated_dense_n4_phase_source(mixed_path)


def test_dense_phase_validator_rejects_tampered_raw_suffix_source(
    tmp_path: Path,
) -> None:
    suffix = pd.read_csv(DEFAULT_DENSE_SUFFIX_SOURCE)
    suffix.loc[0, "history_nrmse"] = float(suffix.loc[0, "history_nrmse"]) + 0.01
    tampered = tmp_path / "suffix_decomposition.csv"
    suffix.to_csv(tampered, index=False)
    with pytest.raises(ValueError, match="registered SHA-256"):
        _validated_dense_phase_source(DEFAULT_DENSE_PHASE_SOURCE, tampered)


def test_dense_n4_validator_rejects_tampered_raw_suffix_source(
    tmp_path: Path,
) -> None:
    suffix = pd.read_csv(DEFAULT_DENSE_N4_SUFFIX_SOURCE)
    suffix.loc[0, "history_nrmse"] = float(suffix.loc[0, "history_nrmse"]) + 0.01
    tampered = tmp_path / "suffix_decomposition_n4.csv"
    suffix.to_csv(tampered, index=False)
    with pytest.raises(ValueError, match="registered SHA-256"):
        _validated_dense_n4_phase_source(DEFAULT_DENSE_N4_PHASE_SOURCE, tampered)


def test_renderer_rejects_a_tampered_protected_memory_gap(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_DARK_SUMMARY.read_text(encoding="utf-8"))
    payload["expected_prefix_gap"] = 0.2
    tampered = tmp_path / "dark_sector_summary.json"
    tampered.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unexpected protected-memory output gap"):
        render_all(dark_summary_path=tampered, output_dir=tmp_path / "figures")
