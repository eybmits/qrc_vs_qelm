"""Small unit tests: no existing manuscript or TeX build is required."""

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("icaart_validator", Path(__file__).with_name("validate_submission.py"))
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


def test_all_unicode_whitespace_is_excluded():
    assert qa.nonspace_characters("a b\nc\td\u00a0e\u2003f") == 6


def test_abstract_counter_ignores_comments_and_citations():
    assert qa.abstract_words(r"A \textbf{reset-window} model \cite{ref1}. % hidden words") == [
        "A", "reset-window", "model"]


@pytest.mark.parametrize("text", ["Baumann", "github.com/eybmits/project", "qrc_vs_qelm", "/Users/example/file"])
def test_identifying_material_is_detected(text):
    assert qa.identifying_hits(text)


def test_general_qrc_language_is_not_an_identifier():
    assert not qa.identifying_hits("QRC and reset-window QELM emulation")


def test_font_parser_handles_multiword_font_types():
    output = "name type encoding emb sub uni object ID\n------------------\nABCD+Foo CID TrueType Identity-H yes yes no 10 0\nBar Type 1 WinAnsi no no yes 11 0\n"
    assert qa.parse_fonts(output) == [
        {"name": "ABCD+Foo", "embedded": True, "subset": True, "unicode": False},
        {"name": "Bar", "embedded": False, "subset": False, "unicode": True},
    ]


def test_font_parser_rejects_unknown_output():
    with pytest.raises(ValueError):
        qa.parse_fonts("header\nseparator\nmalformed row")


def test_log_reference_and_compilation_failures():
    assert qa.log_problems("LaTeX Warning: Reference `thm:x' on page 3 undefined on input line 20.")
    assert qa.log_problems("! Undefined control sequence.")
    assert not qa.log_problems("Output written on submission.pdf (10 pages).")


def test_margin_diagnostics_do_not_treat_glyph_excursions_as_clipping():
    result = qa.margin_diagnostics([{"size": qa.A4_POINTS, "boxes": [(50, 90, 550, 720)]}])
    assert result["template_margin_excursions"]
    assert not result["physical_page_clipping"]


def test_physical_clipping_is_reported():
    result = qa.margin_diagnostics([{"size": qa.A4_POINTS, "boxes": [(-4, 90, 600, 720)]}])
    assert result["physical_page_clipping"]


def test_template_hash_gate_checks_bytes(tmp_path):
    source = tmp_path / "SCITEPRESS.sty"
    source.write_text("official style")
    manifest = tmp_path / "template_manifest.json"
    manifest.write_text(json.dumps({"files": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}}))
    assert qa.check_template_manifest(manifest, tmp_path)["passed"]
    source.write_text("modified style")
    assert not qa.check_template_manifest(manifest, tmp_path)["passed"]


def test_template_manifest_cannot_escape_package(tmp_path):
    manifest = tmp_path / "template_manifest.json"
    manifest.write_text(json.dumps({"files": {"../outside": "bad"}}))
    with pytest.raises(ValueError, match="escapes"):
        qa.check_template_manifest(manifest, tmp_path)


def test_synthetic_pdf_backend_reads_a4_page(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    if not shutil.which("pdftotext"):
        pytest.skip("Optional Poppler backend unavailable")
    pdf = tmp_path / "synthetic.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=qa.A4_POINTS[0], height=qa.A4_POINTS[1])
    writer.add_metadata({"/Author": "Anonymous"})
    with pdf.open("wb") as stream:
        writer.write(stream)
    result = qa.read_pdf(pdf)
    assert len(result["pages"]) == 1
    assert result["pages"][0]["size"] == pytest.approx(qa.A4_POINTS)
    assert result["metadata"]["author"] == "Anonymous"


def test_all_validation_gates_are_included(tmp_path, monkeypatch):
    pdf = tmp_path / "submission.pdf"
    pdf.write_bytes(b"Synthetic file: PDF extraction is mocked.")
    pdf.with_suffix(".log").write_text("Output written on submission.pdf")
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "abstract.tex").write_text("word " * 70)
    style = tmp_path / "SCITEPRESS.sty"
    style.write_text("unchanged")
    (tmp_path / "template_manifest.json").write_text(json.dumps({
        "files": {style.name: hashlib.sha256(style.read_bytes()).hexdigest()}}))
    monkeypatch.setattr(qa, "read_pdf", lambda _: {
        "pages": [{"size": qa.A4_POINTS, "text": "x " * 10000, "boxes": []}],
        "metadata": {"author": ""}, "links": [], "attachments": 0, "backend": "mock"})
    monkeypatch.setattr(qa.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout="header\nseparator\nFont Type 1 WinAnsi yes yes yes 1 0\n"))
    result = qa.validate(pdf)
    assert result["format_passed"]
    assert not result["passed"]
    assert not result["policy_passed"]
    assert not result["submission_ready"]
    assert not result["policy_checks"]["required_ai_disclosure"]["passed"]
    assert "required_ai_disclosure_and_tool_citations_missing_or_incomplete" in result["submission_blockers"]
    assert len(result["checks"]) == 11
    (tmp_path / "sections" / "abstract.tex").write_text("word " * 201)
    assert not qa.validate(pdf)["format_passed"]


def test_absent_mentions_do_not_imply_no_disclosure_required(tmp_path):
    result = qa.check_required_disclosure("Recurrent quantum models and their outputs.", tmp_path)
    assert result["required_for_this_project"]
    assert not result["passed"]
    assert not any(result["components"].values())
    assert result["sources"] == list(qa.POLICY_SOURCES)


def test_disclosure_check_requires_more_than_a_tool_name_in_references(tmp_path):
    result = qa.check_required_disclosure("References. OpenAI Codex, 2026.", tmp_path)
    assert result["components"]["named_tool_in_pdf"]
    assert not result["components"]["disclosure_in_pdf"]
    assert not result["passed"]


def test_necessary_disclosure_presence_does_not_assess_adequacy(tmp_path):
    sections = tmp_path / "sections"
    sections.mkdir()
    for name in ("introduction", "models", "background", "dynamics", "evaluation", "conclusion", "appendix"):
        (sections / f"{name}.tex").write_text(r"Text \cite{openai2026codex}.")
    result = qa.check_required_disclosure(
        "AI-use disclosure. OpenAI Codex assisted preparation. Figure prompt: show both models.", tmp_path)
    assert result["passed"]
    assert "Human review" in result["note"]
    (sections / "models.tex").write_text(r"Text. % \cite{openai2026codex}")
    assert not qa.check_required_disclosure(
        "AI-use disclosure. OpenAI Codex assisted preparation. Figure prompt: show both models.", tmp_path)["passed"]


def test_editorial_paper_sources_omit_tool_mentions():
    base = Path(__file__).parent
    files = [base / "main.tex", base / "references.bib", *sorted((base / "sections").glob("*.tex"))]
    combined = "\n".join(path.read_text().lower() for path in files)
    for mention in ("aicredit", "openai", "codex", "llm", "ai-use", "ai-assisted"):
        assert mention not in combined
    assert not (base / "sections" / "ai-use.tex").exists()


def test_paper_figure_selection_excludes_archived_frontier():
    base = Path(__file__).parent
    spec = importlib.util.spec_from_file_location("icaart_assets", base / "prepare_assets.py")
    assets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assets)
    paper = "\n".join(path.read_text() for path in (base / "sections").glob("*.tex"))
    included = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", paper)
    assert len(included) == 3
    assert set(included) == set(assets.PAPER_FIGURES)
    assert "fig02_information_frontier.pdf" not in assets.PAPER_FIGURES
    # Removal from the paper must not delete the reproducible source artifact.
    archived = base.parent / "figures" / "information_interface"
    for suffix in (".pdf", ".png", "_source.csv"):
        assert (archived / f"fig02_information_frontier{suffix}").is_file()
