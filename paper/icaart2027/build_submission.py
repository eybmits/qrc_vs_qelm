#!/usr/bin/env python3
"""Build an anonymous editorial draft; report submission-policy blockers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from validate_submission import validate
from prepare_assets import PAPER_FIGURES


BASE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_paper(directory: Path) -> None:
    result = subprocess.run(
        ["latexmk", "-pdf", "-jobname=submission", "-interaction=nonstopmode",
         "-halt-on-error", "main.tex"],
        cwd=directory, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (directory / "build.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"LaTeX build failed; inspect {directory / 'build.log'}")


def main() -> None:
    compile_paper(BASE)
    qa = validate(BASE / "submission.pdf")
    (BASE / "submission_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    if not qa["format_passed"]:
        raise RuntimeError("Submission format checks failed; inspect submission_qa.json")

    # Only compilation inputs enter the anonymous source package. In particular,
    # no author handoff, repository metadata, local paths, tests or old drafts.
    sources = [BASE / name for name in (
        "main.tex", "references.bib", "article.cls", "SCITEPRESS.sty",
        "apalike.sty", "apalike.bst",
    )]
    sources += sorted((BASE / "sections").glob("*.tex"))
    # Do not silently repackage archived figures left from an earlier draft.
    sources += [BASE / "figures" / name for name in PAPER_FIGURES]
    archive = BASE / "ICAART2027_anonymous_source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sources:
            item = zipfile.ZipInfo(str(path.relative_to(BASE)), (1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            output.writestr(item, path.read_bytes())

    with tempfile.TemporaryDirectory(prefix="icaart2027-clean-") as temporary:
        clean = Path(temporary)
        with zipfile.ZipFile(archive) as source_zip:
            source_zip.extractall(clean)
        compile_paper(clean)
        if digest(clean / "submission.pdf") != digest(BASE / "submission.pdf"):
            raise RuntimeError("Clean-extraction build differs from the delivered PDF")

    artifacts = sources + [archive, BASE / "submission.pdf", BASE / "submission_qa.json",
                           BASE / "template_manifest.json", BASE / "figure_provenance.json",
                           BASE / "build_submission.py", BASE / "prepare_assets.py",
                           BASE / "validate_submission.py", BASE / "test_validate_submission.py",
                           BASE / "README.md"]
    visual = BASE / "visual_review.json"
    visual_matches = False
    if visual.is_file():
        artifacts.append(visual)
        visual_matches = json.loads(visual.read_text()).get("pdf_sha256") == digest(BASE / "submission.pdf")
    manifest = {
        "schema": "icaart2027_editorial_draft_release_v2",
        "purpose": "Anonymous editorial draft in ICAART format; not authorized for submission",
        "status": "EDITORIAL_DRAFT_NOT_SUBMISSION_READY",
        "format_passed": qa["format_passed"],
        "policy_passed": qa["policy_passed"],
        "submission_ready": False,
        "submission_blockers": qa["submission_blockers"],
        "clean_extraction_pdf_identical": True,
        "visual_review_matches_pdf": visual_matches,
        "pages": qa["checks"]["page_budget"]["pages"],
        "non_whitespace_characters": qa["checks"]["character_budget"]["characters"],
        "abstract_words": qa["checks"]["abstract_length"]["words"],
        "author_approval": "pending",
        "portal_submission": "not performed",
        "files": {str(path.relative_to(BASE)): digest(path) for path in artifacts},
    }
    (BASE / "ICAART2027_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
