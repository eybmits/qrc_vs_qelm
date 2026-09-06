from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from PIL import Image, ImageStat
import pytest


ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from render_figure1_bc_candidates import (  # noqa: E402
    DESIGNS,
    NAMES,
    NATIVE_CM,
    PROTECTED,
    load_descriptions,
    pdf_dimensions,
    selection_tex,
    tex_escape,
)


def read_manifest() -> dict:
    return json.loads((DESIGNS / "candidate_manifest.json").read_text())


def test_exactly_ten_independent_unselected_editorial_candidates() -> None:
    manifest = read_manifest()
    assert manifest["schema"] == "figure1_bc_selection_v1"
    assert manifest["status"] == "awaiting_user_selection_no_manuscript_replacement"
    assert manifest["selected_b"] is None and manifest["selected_c"] is None
    assert tuple(manifest["variants"]) == NAMES
    assert manifest["native_size_cm"] == list(NATIVE_CM)
    assert manifest["label_size_pt"] == 10
    assert "not simulations" in manifest["scope"]
    assert manifest["reference"]["figure"] == 1
    assert "no reference-paper data reused" in manifest["reference"]["use"]
    guard = manifest["unchanged_during_build"]
    assert guard["before"] == guard["after"]
    assert set(guard["before"]) == set(PROTECTED)
    for paper in ("canonical_emulation", "icaart2027"):
        for source in (ROOT / "paper" / paper).rglob("*.tex"):
            assert "figure1_bc_candidates" not in source.read_text()


def test_selection_source_is_reproducible_native_scale_and_scoped() -> None:
    entries = load_descriptions()
    assert [entry["id"] for entry in entries] == [name.upper() for name in NAMES]
    assert len({entry["title_de"] for entry in entries}) == 10
    short_captions = json.loads((DESIGNS / "selection_guide.json").read_text())
    assert set(short_captions) == {name.upper() for name in NAMES}
    assert all(0 < len(caption) <= 140 for caption in short_captions.values())
    for entry in entries:
        assert len(entry["caption_de"].split()) <= 45
    source = selection_tex(entries)
    assert source == (DESIGNS / "selection.tex").read_text()
    assert source.count(r"\includegraphics[width=8.60cm]") == 10
    assert "Nummern müssen nicht übereinstimmen" in source
    assert "keine neuen Messdaten" in source
    assert "8,6 $\\times$ 5,6 cm" in source
    assert "10-pt-Beschriftung" in source
    assert pdf_dimensions(DESIGNS / "selection.pdf", pages=2) == pytest.approx(
        (595.276, 841.89), abs=.1,
    )
    text = subprocess.run(
        ["pdftotext", str(DESIGNS / "selection.pdf"), "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert all(name.upper() in text for name in NAMES)
    for panel in "bc":
        with Image.open(DESIGNS / f"overview_{panel}.png") as preview:
            assert ImageStat.Stat(preview.convert("L")).stddev[0] > 25


@pytest.mark.parametrize("name", NAMES)
def test_candidate_native_dimensions_fonts_and_editable_sources(name: str) -> None:
    manifest = read_manifest()
    source = (DESIGNS / f"{name}.tex").read_text()
    style = (DESIGNS / "style.tex").read_text()
    assert r"\input{style.tex}" in source
    assert rf"\begin{{candidate}}{{{name[0]}}}" in source
    assert r"\resizebox" not in source and r"\scalebox" not in source
    assert not re.search(r"\\(?:sub)?title\{", source)
    assert r"\useasboundingbox (0,0) rectangle (8.60,5.60);" in style
    assert r"\usepackage{helvet}" in style
    font_sizes = [float(value) for value in re.findall(r"\\fontsize\{([\d.]+)\}", style + source)]
    assert min(font_sizes) >= 10 and max(font_sizes) <= 12
    size = pdf_dimensions(DESIGNS / f"{name}.pdf")
    assert size == pytest.approx(tuple(cm * 72 / 2.54 for cm in NATIVE_CM), abs=.1)
    assert manifest["variants"][name]["pdf_size_points"] == pytest.approx(size)
    assert manifest["variants"][name]["conceptual_only"] is True
    with Image.open(DESIGNS / f"{name}.png") as preview:
        assert preview.width == 1400
        assert preview.height == pytest.approx(1400 * 5.60 / 8.60, abs=1)
    fonts = subprocess.run(
        ["pdffonts", str(DESIGNS / f"{name}.pdf")],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "Type 3" not in fonts
    assert "NimbusSan" in fonts or "Helvetica" in fonts


def test_candidate_manifest_checksums_and_no_missing_outputs() -> None:
    manifest = read_manifest()
    required = {f"{name}.{extension}" for name in NAMES for extension in ("tex", "pdf", "png")}
    required |= {"style.tex", "DESIGN_BRIEF.md", "README.md", "b_descriptions.json", "c_descriptions.json", "selection_guide.json",
                 "render_figure1_bc_candidates.py", "test_figure1_bc_candidates.py",
                 "selection.tex", "selection.pdf", "overview_b.png", "overview_c.png"}
    assert set(manifest["artifacts"]) == required
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((DESIGNS / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((CODE / "render_figure1_bc_candidates.py").read_bytes()).hexdigest() == (
        manifest["renderer_sha256"]
    )


def test_selection_captions_escape_tex_without_changing_meaning() -> None:
    assert tex_escape("10% & B_1") == r"10\% \& B\_1"
    assert tex_escape("Ausgabe → Messung") == r"Ausgabe $\to$ Messung"
    assert tex_escape("{x}") == r"\{x\}"
