#!/usr/bin/env python3
"""Derive an identified arXiv package from the unchanged anonymous source ZIP.

Only title-page authors, PDF metadata, and the repository-access sentence
differ. Scientific content, references, figures, and AI disclosure are preserved.
No upload or publication is performed.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

BASE = Path(__file__).resolve().parent
ANONYMOUS = BASE.parent / "icaart2027"
ANONYMOUS_CODE_NOTE = b"Code link withheld for anonymity."
AUTHORED_CODE_NOTE = (
    rb"Code repository: \href{https://github.com/eybmits/qrc_vs_qelm}"
    rb"{qrc\_vs\_qelm} (currently access-restricted).")
sys.path.insert(0, str(ANONYMOUS))
from validate_submission import log_problems, margin_diagnostics, parse_fonts, read_pdf


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tex(value: str) -> str:
    return "".join({"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
                    "ä": r'\"a', "ü": r'\"u', "ö": r'\"o'}.get(c, c) for c in value)


def valid_orcid(value: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", value):
        return False
    total = 0
    for char in value.replace("-", "")[:-1]:
        total = (total + int(char)) * 2
    result = (12 - total % 11) % 11
    return value[-1] == ("X" if result == 10 else str(result))


def authored_main(source: str, metadata: dict) -> str:
    authors = metadata["authors"]
    names = []
    for author in authors:
        name = rf'\href{{{author["profile_url"]}}}{{{tex(author["name"])}}}'
        if author["orcid"]:
            if not valid_orcid(author["orcid"]):
                raise ValueError(f'Invalid ORCID for {author["name"]}')
            name += rf'\,\orcidlink{{{author["orcid"]}}}'
        names.append(name)
    affiliation = metadata["affiliation"]
    corresponding = next(a for a in authors if a["corresponding"])
    author_block = "\n".join([
        r"\author{",
        r"\authorname{" + ", ".join(names[:-1]) + " and " + names[-1] + "}",
        rf'\affiliation{{\href{{{affiliation["laboratory_url"]}}}{{{tex(affiliation["laboratory"])}}}, '
        + tex(affiliation["department"]) + r"\\",
        rf'\href{{{affiliation["institution_url"]}}}{{{tex(affiliation["institution"])}}}, '
        + tex(affiliation["city"]) + ", " + tex(affiliation["country"]) + "}",
        r"\email{Correspondence: \href{mailto:" + corresponding["email"]
        + "}{" + tex(corresponding["email"]) + "}}",
        "}",
    ])
    replacements = {
        "% No author, affiliation, email, ORCID, date, or acknowledgements in review PDF.\n\\author{}":
            "% Author-identified preprint; generated from the anonymous manuscript.\n" + author_block,
        "pdfauthor={},pdfsubject={Anonymous editorial draft in ICAART 2027 format}":
            "pdfauthor={" + "; ".join(a["name"] for a in authors) + "},pdfsubject={Author-identified research preprint}",
        "% mathematical operator in this anonymous mathematical manuscript.":
            "% mathematical operator in this mathematical manuscript.",
        r"\graphicspath{{figures/}}": "\n".join([
            "% Preprint title spacing; preserve publisher font sizes and allow clear author separation.",
            r"\usepackage{etoolbox,orcidlink}",
            r"\makeatletter",
            r"\patchcmd{\maketitle}{\vskip 0.51in}{\vskip 0.12in}{}{\errmessage{Unexpected title layout}}",
            r"\patchcmd{\maketitle}{\vskip -0.07in}{\vskip 0pt}{}{\errmessage{Unexpected title-author layout}}",
            r"\renewcommand{\title}[1]{\gdef\@title{#1\\[12pt]}}",
            r"\renewcommand{\email}[1]{{\fontsize{9}{11}\selectfont\textit{#1}}\par}",
            r"\makeatother",
            r"\graphicspath{{figures/}}",
        ]),
    }
    for original, replacement in replacements.items():
        if source.count(original) != 1:
            raise ValueError(f"Anonymous source layout changed: {original}")
        source = source.replace(original, replacement)
    return source


def compile_paper(directory: Path) -> None:
    result = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=directory, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (BASE / "build.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError("arXiv build failed; inspect paper/arxiv/build.log")
    errors = log_problems((directory / "main.log").read_text(errors="replace"))
    if errors:
        raise RuntimeError(f"Unresolved LaTeX references: {errors}")


def write_inputs(directory: Path, inputs: dict[str, bytes]) -> None:
    for name, content in inputs.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Unsafe source archive path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def main() -> None:
    metadata_path = BASE / "authors.json"
    metadata = json.loads(metadata_path.read_text())
    manifest = json.loads((ANONYMOUS / "ICAART2027_MANIFEST.json").read_text())
    archive_path = ANONYMOUS / "ICAART2027_anonymous_source.zip"
    if digest(archive_path.read_bytes()) != manifest["files"][archive_path.name]:
        raise ValueError("Anonymous source archive no longer matches its manifest")
    anonymous_pdf = ANONYMOUS / "submission.pdf"
    before = digest(anonymous_pdf.read_bytes())
    if before != manifest["files"][anonymous_pdf.name]:
        raise ValueError("Anonymous PDF no longer matches its manifest")
    with zipfile.ZipFile(archive_path) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    # Fail if a local manuscript edit has not yet been built into the source ZIP.
    for name, content in original.items():
        if (ANONYMOUS / name).read_bytes() != content:
            raise ValueError(f"Anonymous source archive is stale: {name}")
    inputs = dict(original)
    inputs["main.tex"] = authored_main(original["main.tex"].decode(), metadata).encode()
    method = "sections/methodology.tex"
    if original[method].count(ANONYMOUS_CODE_NOTE) != 1:
        raise ValueError("Anonymous repository note changed")
    inputs[method] = original[method].replace(ANONYMOUS_CODE_NOTE, AUTHORED_CODE_NOTE)
    with tempfile.TemporaryDirectory(prefix="qrc-arxiv-build-") as temporary:
        working = Path(temporary)
        write_inputs(working, inputs)
        compile_paper(working)
        pdf = working / "main.pdf"
        data = read_pdf(pdf)
        full_text = "\n".join(p["text"] for p in data["pages"])
        first_text = re.sub(r"\s+", " ", data["pages"][0]["text"])
        fonts = parse_fonts(subprocess.run(["pdffonts", str(pdf)], check=True,
                                           text=True, capture_output=True).stdout)
        expected_links = [metadata["affiliation"]["laboratory_url"],
                          metadata["affiliation"]["institution_url"]]
        expected_links += [a["profile_url"] for a in metadata["authors"]]
        expected_links += ["https://orcid.org/" + a["orcid"] for a in metadata["authors"] if a["orcid"]]
        expected_links += ["mailto:" + a["email"] for a in metadata["authors"] if a["corresponding"]]
        expected_links.append("https://github.com/eybmits/qrc_vs_qelm")
        checks = {
            "all_authors_on_title_page": all(a["name"] in first_text for a in metadata["authors"]),
            "author_order_correct": [first_text.index(a["name"]) for a in metadata["authors"]]
                == sorted(first_text.index(a["name"]) for a in metadata["authors"]),
            "pdf_author_metadata": data["metadata"].get("author") == "; ".join(a["name"] for a in metadata["authors"]),
            "profile_lmu_and_orcid_links": set(expected_links).issubset(data["links"]),
            "no_page_clipping": not margin_diagnostics(data["pages"])["physical_page_clipping"],
            "embedded_fonts": bool(fonts) and all(font["embedded"] for font in fonts),
            "no_embedded_attachments": data["attachments"] == 0,
            "disclosure_preserved": "OpenAI Codex" in full_text and "AI-use disclosure" in full_text,
            "scientific_content_preserved": all(
                (inputs[n].replace(AUTHORED_CODE_NOTE, ANONYMOUS_CODE_NOTE) if n == method else inputs[n]) == b
                for n, b in original.items() if n != "main.tex"),
            "anonymous_pdf_unchanged": digest(anonymous_pdf.read_bytes()) == before,
        }
        inputs["main.bbl"] = (working / "main.bbl").read_bytes()
        pdf_bytes = pdf.read_bytes()
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, content in sorted(inputs.items()):
            item = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            output.writestr(item, content)
    with tempfile.TemporaryDirectory(prefix="qrc-arxiv-clean-") as temporary:
        clean = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(archive_buffer.getvalue())) as source:
            write_inputs(clean, {name: source.read(name) for name in source.namelist()})
        compile_paper(clean)
        checks["clean_source_rebuild_identical"] = (clean / "main.pdf").read_bytes() == pdf_bytes
    qa = {"schema": "arxiv_package_qa_v1", "checks": checks, "passed": all(checks.values()),
          "pages": len(data["pages"]), "pdf_sha256": digest(pdf_bytes),
          "anonymous_pdf_sha256": before, "source_zip_sha256": digest(archive_buffer.getvalue()),
          "orcid_not_supplied": [a["name"] for a in metadata["authors"] if not a["orcid"]],
          "scope": "Local source/build, identity/link, and geometry checks; arXiv processing and publication have not occurred."}
    if not qa["passed"]:
        raise RuntimeError("arXiv package QA failed: " + json.dumps(qa))
    delivered = {"main.tex": inputs["main.tex"], "preprint.pdf": pdf_bytes,
                 "arxiv_source.zip": archive_buffer.getvalue(),
                 "arxiv_qa.json": (json.dumps(qa, indent=2) + "\n").encode()}
    release = {"schema": "arxiv_preprint_package_v1", "anonymous_source_sha256": digest(archive_path.read_bytes()),
               "authors_sha256": digest(metadata_path.read_bytes()), "pages": qa["pages"],
               "parent_source_sha256": {n: digest(b) for n, b in original.items()},
               "administrative_changes": {
                   "main.tex": "Author header with compact title spacing, linked green ORCID icons, profile links, and PDF author/subject metadata",
                   method: "Anonymity-only code note replaced with a link and accurate restricted-access status"},
               "source_package_files": {n: digest(b) for n, b in inputs.items()},
               "files": {n: digest(b) for n, b in delivered.items()} | {
                   n: digest((BASE / n).read_bytes()) for n in ("authors.json", "build_arxiv.py")},
               "publication": "Not uploaded or published; no arXiv identifier has been assigned."}
    delivered["ARXIV_MANIFEST.json"] = (json.dumps(release, indent=2, sort_keys=True) + "\n").encode()
    # Build/rebuild failures above leave the previous deliverables intact.
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=BASE) as temporary:
        staging = Path(temporary)
        write_inputs(staging, delivered)
        for name in delivered:
            (staging / name).replace(BASE / name)
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
