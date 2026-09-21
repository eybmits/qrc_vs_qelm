#!/usr/bin/env python3
"""Separate format/anonymity QA from venue-policy and submission readiness.

This does not determine scientific correctness or conference acceptance. It also
does not replace visual inspection. PyMuPDF is preferred; pypdf and Poppler form
an equivalent local fallback for structural checks when PyMuPDF is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent
A4_POINTS = (210 * 72 / 25.4, 297 * 72 / 25.4)
IDENTIFIERS = (
    r"eybmits", r"markus[._\s-]*baumann", r"\bbaumann\b",
    r"qrc[_\s-]*vs[_\s-]*qelm", r"/Users/", r"/home/", r"superposition",
)
PLACEHOLDERS = re.compile(r"\b(?:TODO|FIXME|TBD|PLACEHOLDER)\b", re.I)
LOG_ERRORS = (
    r"(?:Reference|Citation) .+? undefined", r"There were undefined references",
    r"Undefined control sequence", r"Emergency stop", r"Fatal error occurred",
)
POLICY_SOURCES = (
    "https://icaart.scitevents.org/Guidelines.aspx?y=2027",
    "https://icaart.scitevents.org/AiTools.aspx?y=2027",
)


def check_submission_approval(pdf: Path) -> dict:
    """Read explicit author decisions, valid only for the reviewed PDF bytes."""
    path = pdf.parent / "submission_approval.json"
    try:
        record = json.loads(path.read_text())
        current = (record.get("schema") == "icaart2027_author_approval_v1"
                   and record.get("pdf_sha256") == hashlib.sha256(pdf.read_bytes()).hexdigest())
        return {
            "record": path.name,
            "matches_pdf": current,
            "coauthors_approved": current and record.get("coauthor_approval") == "approved",
            "ai_disclosure_confirmed": current and record.get("ai_disclosure_review") == "confirmed_complete",
            "confirmed_on": record.get("confirmed_on"),
            "basis": "Explicit author confirmation; not an automated policy determination.",
        }
    except (OSError, ValueError, AttributeError) as exc:
        return {"record": path.name, "matches_pdf": False, "coauthors_approved": False,
                "ai_disclosure_confirmed": False, "error": str(exc)}


def check_required_disclosure(text: str, base: Path) -> dict:
    """Detect missing required disclosure for this project's known tool use.

    This is a necessary-content check, not an assessment of disclosure adequacy.
    Its result must never infer that no tool was used from absent manuscript text.
    """
    normalized = re.sub(r"\s+", " ", text).lower()
    sections = base / "sections"
    components = {
        "named_tool_in_pdf": "openai" in normalized and "codex" in normalized,
        "disclosure_in_pdf": bool(re.search(
            r"ai[ -]use disclosure|ai-assisted preparation|use of (?:generative )?ai",
            normalized)),
        "affected_section_tool_citations": all(
            (sections / f"{name}.tex").is_file()
            and bool(re.search(r"\\(?:cite\w*|aicredit)\b",
                               strip_comments((sections / f"{name}.tex").read_text())))
            and ("openai2026codex" in strip_comments((sections / f"{name}.tex").read_text())
                 or r"\aicredit" in strip_comments((sections / f"{name}.tex").read_text()))
            for name in ("introduction", "models", "background", "dynamics", "evaluation",
                         "conclusion", "appendix")),
        "conceptual_figure_credit_and_prompt": bool(re.search(
            r"(?:prompt|instruction).*?(?:codex|openai)|(?:codex|openai).*?(?:prompt|instruction)",
            normalized)),
    }
    return {
        "passed": all(components.values()),
        "required_for_this_project": True,
        "components": components,
        "known_use": "Research, text, code and conceptual-figure preparation included tool assistance.",
        "policy_checked_on": "2026-09-06",
        "sources": list(POLICY_SOURCES),
        "note": "Missing disclosure is a submission blocker, not a formatting defect. "
                "Human review must check accuracy, scope, placement and figure-credit adequacy.",
    }


def nonspace_characters(text: str) -> int:
    """Count Unicode characters, excluding every Unicode whitespace class."""
    return sum(not char.isspace() for char in text)


def strip_comments(source: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def abstract_words(source: str) -> list[str]:
    """Transparent prose-word estimate; citations/labels are not prose words."""
    source = strip_comments(source)
    source = re.sub(r"\\(?:cite\w*|label|ref)\*?(?:\[[^]]*\])?\{[^}]*\}", "", source)
    source = re.sub(r"\\[a-zA-Z]+\*?", " ", source)
    source = source.replace("~", " ").replace("{", " ").replace("}", " ")
    return re.findall(r"[^\W_]+(?:[-'\u2019][^\W_]+)*", source, re.UNICODE)


def identifying_hits(text: str) -> list[str]:
    return sorted({match.group(0) for pattern in IDENTIFIERS
                   for match in re.finditer(pattern, text, re.I)})


def parse_fonts(output: str) -> list[dict]:
    rows = []
    for line in output.splitlines()[2:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 8 or fields[-5] not in {"yes", "no"}:
            raise ValueError(f"Unrecognized pdffonts row: {line}")
        rows.append({"name": fields[0], "embedded": fields[-5] == "yes",
                     "subset": fields[-4] == "yes", "unicode": fields[-3] == "yes"})
    return rows


def log_problems(text: str) -> list[str]:
    return sorted({match.group(0) for pattern in LOG_ERRORS
                   for match in re.finditer(pattern, text, re.I)})


def margin_diagnostics(pages: list[dict]) -> dict:
    """Report template-margin excursions; only physical clipping is a gate.

    SCITEPRESS 2023 final style uses approximately 26 mm side, 33.4 mm top,
    and 42.6 mm bottom margins. Text glyph extents can differ from TeX's
    baseline/box coordinates, so this margin audit is deliberately diagnostic.
    """
    clipping, excursions = [], []
    for number, page in enumerate(pages, 1):
        width, height = page["size"]
        margin_box = (26 * 72 / 25.4, 33.4 * 72 / 25.4,
                      width - 26 * 72 / 25.4, height - 42.6 * 72 / 25.4)
        for box in page["boxes"]:
            x0, y0, x1, y1 = box
            if x0 < -1 or y0 < -1 or x1 > width + 1 or y1 > height + 1:
                clipping.append({"page": number, "box": list(box)})
            if (x0 < margin_box[0] - 4 or y0 < margin_box[1] - 4
                    or x1 > margin_box[2] + 4 or y1 > margin_box[3] + 4):
                excursions.append({"page": number, "box": list(box)})
    return {"physical_page_clipping": clipping, "template_margin_excursions": excursions,
            "note": "Glyph-box diagnostics only; inspect rendered pages for overlap and layout."}


def read_pdf(pdf: Path) -> dict:
    try:
        fitz = importlib.import_module("fitz")
    except ImportError:
        fitz = None
    if fitz is not None:
        with fitz.open(pdf) as document:
            pages = [{"size": (page.rect.width, page.rect.height), "text": page.get_text(),
                      "boxes": [word[:4] for word in page.get_text("words")]}
                     for page in document]
            links = [link.get("uri", "") for page in document for link in page.get_links()]
            return {"pages": pages, "metadata": document.metadata, "links": links,
                    "attachments": document.embfile_count(), "backend": "PyMuPDF"}
    from pypdf import PdfReader

    if not shutil.which("pdftotext"):
        raise RuntimeError("Install PyMuPDF or provide pypdf and Poppler pdftotext.")
    reader = PdfReader(pdf)
    result = subprocess.run(["pdftotext", "-bbox-layout", str(pdf), "-"],
                            check=True, text=True, capture_output=True)
    # Mathematical glyph encodings can produce XML 1.0-invalid control bytes.
    # They have no geometric meaning; remove only those invalid code points.
    xml = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", "", result.stdout)
    root = ET.fromstring(xml)
    page_elements = [element for element in root.iter() if element.tag.endswith("}page")]
    if len(page_elements) != len(reader.pages):
        raise ValueError("PDF and Poppler page counts disagree.")
    pages, links = [], []
    for page, element in zip(reader.pages, page_elements, strict=True):
        boxes = [tuple(float(word.attrib[key]) for key in ("xMin", "yMin", "xMax", "yMax"))
                 for word in element.iter() if word.tag.endswith("}word")]
        pages.append({"size": (float(page.mediabox.width), float(page.mediabox.height)),
                      "text": page.extract_text() or "", "boxes": boxes})
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A", {})
            if action.get("/URI"):
                links.append(str(action["/URI"]))
    metadata = {str(key).lstrip("/").lower(): str(value)
                for key, value in (reader.metadata or {}).items()}
    return {"pages": pages, "metadata": metadata, "links": links,
            "attachments": len(reader.attachments), "backend": "pypdf + Poppler"}


def check_template_manifest(path: Path, base: Path) -> dict:
    """Validate {files: {relative/path: sha256}} or list-of-file-records."""
    manifest = json.loads(path.read_text())
    entries = manifest.get("files", manifest.get("artifacts", {}))
    if isinstance(entries, dict):
        records = [{"path": name, "sha256": value.get("sha256") if isinstance(value, dict) else value}
                   for name, value in entries.items()]
    elif isinstance(entries, list):
        records = entries
    else:
        raise ValueError("Template manifest must provide files as a mapping or list.")
    results = []
    for record in records:
        relative = record.get("path", record.get("name", ""))
        target = (base / relative).resolve()
        if not target.is_relative_to(base.resolve()):
            raise ValueError("Template manifest path escapes the submission directory.")
        expected = record.get("sha256", "")
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        results.append({"path": relative, "matches": actual == expected, "sha256": actual})
    return {"passed": bool(results) and all(row["matches"] for row in results), "files": results}


def validate(pdf: Path, *, max_pages: int = 12, template_manifest: Path | None = None) -> dict:
    base = pdf.parent
    data = read_pdf(pdf)
    pages = data["pages"]
    text = "\n".join(page["text"] for page in pages)
    identifiers = identifying_hits(text + "\n" + json.dumps(data["metadata"])
                                   + "\n" + "\n".join(data["links"]))
    characters = nonspace_characters(text)
    author = str(data["metadata"].get("author", "")).strip()
    anonymous_author = not author or re.fullmatch(r"anonymous(?: authors?)?", author, re.I)
    geometry = margin_diagnostics(pages)
    checks = {
        "page_budget": {"passed": 1 <= len(pages) <= max_pages, "pages": len(pages), "maximum": max_pages},
        "a4_page_size": {"passed": all(abs(page["size"][i] - A4_POINTS[i]) <= 1
                                            for page in pages for i in (0, 1)),
                         "page_sizes_points": [list(page["size"]) for page in pages]},
        "character_budget": {"passed": 10000 <= characters <= 50000, "characters": characters,
                             "minimum": 10000, "maximum": 50000, "definition": "Unicode, whitespace excluded"},
        "anonymous_pdf": {"passed": not identifiers and bool(anonymous_author),
                          "identifying_hits": identifiers, "author_metadata": author},
        "no_embedded_attachments": {"passed": data["attachments"] == 0, "count": data["attachments"]},
        "no_placeholder_text": {"passed": not PLACEHOLDERS.search(text),
                                "hits": sorted(set(PLACEHOLDERS.findall(text)))},
        "no_physical_page_clipping": {"passed": not geometry["physical_page_clipping"],
                                      "hits": geometry["physical_page_clipping"]},
    }
    abstract_path = base / "sections" / "abstract.tex"
    if abstract_path.is_file():
        count = len(abstract_words(abstract_path.read_text()))
        checks["abstract_length"] = {"passed": 70 <= count <= 200, "words": count, "minimum": 70, "maximum": 200}
    else:
        checks["abstract_length"] = {"passed": False, "error": "Missing sections/abstract.tex"}
    log = pdf.with_suffix(".log")
    errors = log_problems(log.read_text(errors="replace")) if log.is_file() else ["Build log missing"]
    checks["resolved_latex_references"] = {"passed": not errors, "errors": errors}
    try:
        output = subprocess.run(["pdffonts", str(pdf)], check=True, text=True, capture_output=True)
        fonts = parse_fonts(output.stdout)
        checks["embedded_fonts"] = {"passed": bool(fonts) and all(font["embedded"] for font in fonts), "fonts": fonts}
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        checks["embedded_fonts"] = {"passed": False, "error": str(exc)}
    manifest_path = template_manifest or base / "template_manifest.json"
    if manifest_path.is_file():
        checks["official_template_unchanged"] = check_template_manifest(manifest_path, base)
    else:
        checks["official_template_unchanged"] = {"passed": False, "error": f"Missing {manifest_path.name}"}
    format_passed = all(check["passed"] for check in checks.values())
    approval = check_submission_approval(pdf)
    disclosure = check_required_disclosure(text, base)
    # Presence heuristics cannot decide which credits apply. Keep their findings
    # visible, but recognize an explicit author review of this exact PDF.
    disclosure["automated_presence_passed"] = disclosure["passed"]
    disclosure["assessment_basis"] = (
        "author_confirmed" if approval["ai_disclosure_confirmed"] else "automated_presence_check")
    disclosure["passed"] = disclosure["passed"] or approval["ai_disclosure_confirmed"]
    policy_checks = {"required_ai_disclosure": disclosure}
    policy_passed = all(check["passed"] for check in policy_checks.values())
    blockers = [] if format_passed else ["format_or_anonymity_checks_failed"]
    if not policy_passed:
        blockers.append("required_ai_disclosure_and_tool_citations_missing_or_incomplete")
    if not approval["coauthors_approved"]:
        blockers.append("human_scientific_author_approval_pending")
    return {"schema": "icaart2027_submission_qa_v3", "pdf": pdf.name,
            "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "extraction_backend": data["backend"],
            "passed": format_passed and policy_passed,
            "format_passed": format_passed, "policy_passed": policy_passed,
            "submission_ready": not blockers, "submission_blockers": blockers,
            "author_approval": approval,
            "checks": checks, "policy_checks": policy_checks,
            "geometry_diagnostics": geometry,
            "scope": "Separate format/anonymity checks, disclosure-presence diagnostics, and explicit "
                     "author review. Author decisions are bound to the PDF hash. Visual review, "
                     "repository CI and portal submission remain separate."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, nargs="?", default=BASE / "submission.pdf")
    parser.add_argument("--max-pages", type=int, default=12)
    parser.add_argument("--template-manifest", type=Path)
    parser.add_argument("--report", type=Path, help="Optionally save the JSON report.")
    parser.add_argument("--format-only", action="store_true",
                        help="Exit on format status only; policy blockers remain in the report.")
    args = parser.parse_args()
    try:
        report = validate(args.pdf.resolve(), max_pages=args.max_pages, template_manifest=args.template_manifest)
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.CalledProcessError) as exc:
        report = {"schema": "icaart2027_submission_qa_v2", "passed": False,
                  "format_passed": False, "submission_ready": False, "error": str(exc)}
    encoded = json.dumps(report, indent=2) + "\n"
    print(encoded, end="")
    if args.report:
        args.report.write_text(encoded)
    gate = "format_passed" if args.format_only else "passed"
    return 0 if report[gate] else 1


if __name__ == "__main__":
    sys.exit(main())
