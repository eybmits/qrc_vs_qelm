"""Focused identity and author-separation checks; unit tests need no TeX run."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zipfile

import pytest


BASE = Path(__file__).resolve().parent
ANONYMOUS = BASE.parent / "icaart2027"
SPEC = importlib.util.spec_from_file_location("qrc_arxiv_builder", BASE / "build_arxiv.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

EXPECTED_AUTHORS = ["Markus Baumann", "Jonas Stein", "Claudia Linnhoff-Popien"]


@pytest.fixture
def metadata():
    return json.loads((BASE / "authors.json").read_text())


@pytest.fixture
def source():
    return (ANONYMOUS / "main.tex").read_text()


def test_authored_transform_preserves_scientific_body_and_disclosure(source, metadata):
    original_metadata = deepcopy(metadata)
    authored = builder.authored_main(source, metadata)
    # Everything from keywords onward includes the exact abstract call, every
    # scientific section input, bibliography, disclosure, and appendix inputs.
    start = r"\keywords{"
    assert authored[authored.index(start):] == source[source.index(start):]
    assert re.findall(r"\\title\{[^\n]+", authored) == re.findall(r"\\title\{[^\n]+", source)
    assert r"\let\sup\mathematicalsup" in authored
    disclosure = source[source.index(r"\par\noindent\textit{AI-use disclosure.}"):]
    assert authored.endswith(disclosure)
    assert metadata == original_metadata


def test_author_order_profile_links_affiliation_and_correspondence(source, metadata):
    assert [author["name"] for author in metadata["authors"]] == EXPECTED_AUTHORS
    authored = builder.authored_main(source, metadata)
    block = authored[authored.index(r"\author{"):authored.index(r"\keywords{")]
    positions = [block.index(name) for name in EXPECTED_AUTHORS]
    assert positions == sorted(positions)
    for author in metadata["authors"]:
        assert rf'\href{{{author["profile_url"]}}}{{{author["name"]}}}' in block
        if author["orcid"]:
            assert rf'\orcidlink{{{author["orcid"]}}}' in block
    for key in ("laboratory_url", "institution_url"):
        assert rf'\href{{{metadata["affiliation"][key]}}}' in block
    corresponding = [author for author in metadata["authors"] if author["corresponding"]]
    assert len(corresponding) == 1
    assert re.findall(r"mailto:([^}]+)", block) == [corresponding[0]["email"]]
    assert "pdfauthor={" + "; ".join(EXPECTED_AUTHORS) + "}" in authored
    assert "pdfsubject={Author-identified research preprint}" in authored


def test_missing_markus_orcid_is_allowed_without_assigning_one(source, metadata):
    metadata["authors"][0]["orcid"] = None
    authored = builder.authored_main(source, metadata)
    expected = [author["orcid"] for author in metadata["authors"] if author["orcid"]]
    assert re.findall(r"\\orcidlink\{([^}]+)\}", authored) == expected
    assert metadata["authors"][0]["orcid"] is None
    assert "orcid.org/None" not in authored


@pytest.mark.parametrize("identifier", ["0009-0007-3575-1006", "0000-0001-5727-9151", "0000-0001-6284-9286"])
def test_verified_author_orcids_pass_the_checksum(identifier):
    assert builder.valid_orcid(identifier)


def test_all_three_verified_author_ids_use_linked_orcid_icons(source, metadata):
    expected = ["0009-0007-3575-1006", "0000-0001-5727-9151", "0000-0001-6284-9286"]
    assert [author["orcid"] for author in metadata["authors"]] == expected
    authored = builder.authored_main(source, metadata)
    assert r"\usepackage{etoolbox,orcidlink}" in authored
    assert re.findall(r"\\orcidlink\{([^}]+)\}", authored) == expected
    assert r"\scriptsize iD" not in authored


@pytest.mark.parametrize("identifier", [
    "0000-0001-5727-9150",  # One-character checksum typo in Jonas Stein's ID.
    "0000-0001-6284-9287",  # One-character checksum typo in Claudia's ID.
    "https://orcid.org/0000-0001-5727-9151",
    "0000000157279151",
])
def test_invalid_orcid_is_rejected(identifier):
    assert not builder.valid_orcid(identifier)


def test_transform_rejects_real_orcid_checksum_typo(source, metadata):
    metadata["authors"][1]["orcid"] = "0000-0001-5727-9150"
    with pytest.raises(ValueError, match="Invalid ORCID for Jonas Stein"):
        builder.authored_main(source, metadata)


@pytest.mark.parametrize("anchor", [
    r"\author{}",
    "pdfauthor={},pdfsubject={Anonymous editorial draft in ICAART 2027 format}",
    "% mathematical operator in this anonymous mathematical manuscript.",
])
def test_source_anchor_drift_fails_closed(source, metadata, anchor):
    assert source.count(anchor) == 1
    with pytest.raises(ValueError, match="Anonymous source layout changed"):
        builder.authored_main(source.replace(anchor, "% changed source anchor"), metadata)


def test_duplicate_author_anchor_fails_closed(source, metadata):
    anchor = "% No author, affiliation, email, ORCID, date, or acknowledgements in review PDF.\n\\author{}"
    with pytest.raises(ValueError, match="Anonymous source layout changed"):
        builder.authored_main(source + "\n" + anchor, metadata)


@pytest.mark.parametrize("ending", [b"", b"\n", b"\n\n"])
def test_authored_methodology_only_appends_full_repository_note(ending):
    original = b"Hardware errors and measurement backaction are omitted." + ending
    authored = builder.authored_methodology(original)
    assert authored[:len(original)] == original
    assert authored[len(original):] == b"\n" + builder.AUTHORED_CODE_NOTE + b"\n"
    assert authored.count(builder.AUTHORED_CODE_NOTE) == 1
    assert rb"\url{https://github.com/eybmits/qrc_vs_qelm}" in builder.AUTHORED_CODE_NOTE
    assert b"access-restricted" not in authored
    assert b"anonymity" not in authored


@pytest.mark.parametrize("note", [
    b"Code link withheld for anonymity.",
    builder.AUTHORED_CODE_NOTE,
])
def test_authored_methodology_rejects_existing_repository_notes(note):
    with pytest.raises(ValueError, match="Anonymous methodology contains a repository note"):
        builder.authored_methodology(b"Scientific methodology.\n" + note + b"\n")


def _read_archive(path):
    if not path.is_file():
        pytest.skip(f"Build artifact is not available: {path.name}")
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        return {name: archive.read(name) for name in archive.namelist()}


def test_existing_anonymous_source_zip_has_no_authored_front_matter(metadata):
    archive = ANONYMOUS / "ICAART2027_anonymous_source.zip"
    original = _read_archive(archive)
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    anonymous_main = original["main.tex"].decode()
    assert r"\author{}" in anonymous_main and "pdfauthor={}" in anonymous_main
    for author in metadata["authors"]:
        assert author["name"] not in anonymous_main
        assert author["profile_url"] not in anonymous_main
        assert author["email"] not in anonymous_main
    assert not {"authors.json", "main_arxiv.tex", "preprint.pdf"}.intersection(original)
    assert all(Path(name).name != "authors.json" for name in original)
    builder.authored_main(anonymous_main, metadata)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before


def test_built_authored_zip_preserves_all_shared_scientific_sources(metadata):
    # Exercise the actual package once the independent build has finished.
    # Fresh source-only checkouts still run every transform/checksum test above.
    if not (BASE / "ARXIV_MANIFEST.json").is_file():
        pytest.skip("Authored package has not been built yet")
    anonymous = _read_archive(ANONYMOUS / "ICAART2027_anonymous_source.zip")
    authored = _read_archive(BASE / "arxiv_source.zip")
    assert set(authored) == set(anonymous) | {"main.bbl"}
    assert authored["main.bbl"].strip()
    assert authored["main.tex"] == builder.authored_main(anonymous["main.tex"].decode(), metadata).encode()
    method = "sections/methodology.tex"
    assert b"Code link withheld for anonymity." not in anonymous[method]
    assert builder.AUTHORED_CODE_NOTE not in anonymous[method]
    assert authored[method].count(builder.AUTHORED_CODE_NOTE) == 1
    assert rb"\url{https://github.com/eybmits/qrc_vs_qelm}" in builder.AUTHORED_CODE_NOTE
    assert b"access-restricted" not in builder.AUTHORED_CODE_NOTE
    assert authored[method] == anonymous[method] + b"\n" + builder.AUTHORED_CODE_NOTE + b"\n"
    changed = {name for name, content in anonymous.items() if authored[name] != content}
    assert changed == {"main.tex", method}
    manifest = json.loads((BASE / "ARXIV_MANIFEST.json").read_text())
    assert set(manifest["administrative_changes"]) == changed
