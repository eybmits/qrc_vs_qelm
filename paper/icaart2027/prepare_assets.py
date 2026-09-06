#!/usr/bin/env python3
"""Copy vector figures for anonymous review without project metadata.

Only the ICAART copies are written. Page content streams, sizes, and scientific
data are preserved; the canonical figures are never modified.
"""
from pathlib import Path
import hashlib
import json

from pypdf import PdfReader, PdfWriter


# Source filenames are stable audit identifiers; LaTeX assigns display numbers.
# The former three-size frontier remains in the repository, not in this paper.
PAPER_FIGURES = (
    "fig01_information_interface.pdf",
    "fig03_resource_diagnosis.pdf",
    "fig04_long_context_n5.pdf",
)


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root.parent / "figures" / "information_interface"
    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    entries = {}
    for filename in PAPER_FIGURES:
        original = (source.parent / "long_context_n5" / filename
                    if filename == "fig04_long_context_n5.pdf" else source / filename)
        reader = PdfReader(original)
        writer = PdfWriter(clone_from=reader)
        writer.metadata = None
        writer.add_metadata({"/Creator": "Scientific figure", "/Title": filename})
        target = destination / filename
        with target.open("wb") as stream:
            writer.write(stream)
        copied = PdfReader(target)
        assert len(reader.pages) == len(copied.pages)
        for a, b in zip(reader.pages, copied.pages):
            assert a.mediabox == b.mediabox
            assert a.get_contents().get_data() == b.get_contents().get_data()
        entries[filename] = {
            "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "anonymous_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "page_content_streams_identical": True,
        }
    (root / "figure_provenance.json").write_text(
        json.dumps({"figures": entries}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
