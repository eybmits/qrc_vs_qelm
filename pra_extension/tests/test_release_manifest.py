from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import build_release_manifest as release  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _fake_repository(root: Path) -> tuple[Path, Path]:
    canonical_files = (
        "configs/canonical_emulation.yaml",
        "pyproject.toml",
        "scripts/run_canonical_emulation.py",
        *release.CANONICAL_PROTOCOL_FILES,
        *release.CANONICAL_SCRIPTS,
        "src/qrc_vs_qelm/__init__.py",
        "tests/test_placeholder.py",
    )
    for relative in canonical_files:
        _write(root / relative, f"content for {relative}\n")

    for relative in release.CANONICAL_RESULT_FILES:
        path = root / relative
        _write(path, "{}\n" if path.suffix == ".json" else "value\n1\n")
    for relative in release.CANONICAL_FIGURE_FILES:
        if relative.endswith("figure_manifest.json"):
            continue
        path = root / relative
        if path.suffix == ".pdf":
            _write(path, b"%PDF-1.4\n%%EOF\n")
        elif path.suffix == ".png":
            _write(path, b"not-a-real-png-but-content-addressed")
        else:
            _write(path, "value\n1\n")

    for relative in release.DENSE_SECONDARY_FILES:
        path = root / relative
        _write(path, "{}\n" if path.suffix == ".json" else "value\n1\n")
    for relative in release.DENSE_N4_FILES:
        path = root / relative
        _write(path, "{}\n" if path.suffix == ".json" else "value\n1\n")

    for relative in release.DOCUMENTATION_FILES:
        _write(root / relative, f"documentation for {relative}\n")
    for relative in release.DARK_SECTOR_FILES:
        path = root / relative
        _write(path, "{}\n" if path.suffix == ".json" else "value\n1\n")
    for relative in release.COMPACT_PRA_FIGURE_FILES:
        path = root / relative
        if path.suffix == ".pdf":
            _write(path, b"%PDF-1.4\n%%EOF\n")
        elif path.suffix == ".png":
            _write(path, b"not-a-real-png-but-content-addressed")
        elif path.suffix == ".json":
            _write(path, "{}\n")
        else:
            _write(path, "value\n1\n")
    for relative in release.THRESHOLD_SENSITIVITY_FILES:
        _write(root / relative, f"analysis code for {relative}\n")
    for relative in release.HISTORICAL_REGRESSION_FILES:
        path = root / relative
        _write(path, "{}\n" if path.suffix == ".json" else "value\n1\n")
    for relative in release.REQUIRED_PAPER_SOURCES:
        _write(root / relative, f"paper source for {relative}\n")
    _write(
        root / "paper/canonical_emulation/sections/abstract.tex",
        "\\begin{abstract}test\\end{abstract}\n",
    )
    _write(root / "pra_extension/code/example.py", "VALUE = 1\n")
    _write(root / "pra_extension/tests/test_example.py", "def test_ok(): pass\n")

    for relative in release.AUTHORITATIVE_EXTENSION_FILES:
        path = root / relative
        if path.suffix == ".csv":
            _write(path, "value\n1\n")
        else:
            _write(path, "{}\n")

    material = release._canonical_material_manifest(root)
    captured_files: dict[str, dict[str, object]] = {}
    for label, digest in material["hashes"].items():
        if label == "configuration":
            relative = "configs/canonical_emulation.yaml"
        elif label == "project_metadata":
            relative = "pyproject.toml"
        elif label == "runner":
            relative = "scripts/run_canonical_emulation.py"
        elif label.startswith("protocol:") or label.startswith("script:"):
            relative = label.split(":", 1)[1]
        elif label.startswith("source:") or label.startswith("test:"):
            relative = label.split(":", 1)[1]
        else:  # pragma: no cover - the helper must track the production labels
            raise AssertionError(f"unmapped canonical material label: {label}")
        captured_files[label] = {
            "path": relative,
            "sha256": digest,
            "size_bytes": (root / relative).stat().st_size,
        }
    _write(
        root / "results_canonical_confirmatory_v1/canonical_manifest.json",
        json.dumps(
            {
                "reproducibility": {
                    "material": {
                        "file_count": material["file_count"],
                        "files": captured_files,
                        "identity_sha256": material["identity_sha256"],
                    }
                }
            }
        ),
    )
    config_hash = _sha(root / "configs/canonical_emulation.yaml")
    _write(
        root / "configs/canonical_emulation_freeze.json",
        json.dumps(
            {
                "schema_version": 1,
                "status": "FROZEN_CONFIRMATORY",
                "study": "canonical_qrc_to_parallel_qelm_emulation",
                "profile": "confirmatory",
                "config_sha256": config_hash,
                "material_identity_sha256": material["identity_sha256"],
                "confirmatory_roots_previously_opened": False,
                "frozen_at_utc": "2026-08-28T14:13:28Z",
            }
        ),
    )

    combined = root / release.EXPLORATORY_ANALYSES[0]
    _write(
        combined,
        json.dumps(
            {
                "status": "EXPLORATORY_POST_CONFIRMATORY",
                "source_artifacts": {
                    link[2]: _sha(root / link[3])
                    for link in release.SOURCE_HASH_LINKS
                    if link[0] == release.EXPLORATORY_ANALYSES[0]
                },
            }
        ),
    )
    spectrum = root / release.EXPLORATORY_ANALYSES[1]
    spectrum_source = [
        link for link in release.SOURCE_HASH_LINKS if link[0] == release.EXPLORATORY_ANALYSES[1]
    ][0]
    _write(
        spectrum,
        json.dumps(
            {
                "status": "EXPLORATORY_MECHANISM_DIAGNOSTIC",
                "artifacts": {spectrum_source[2]: _sha(root / spectrum_source[3])},
            }
        ),
    )
    for analysis_relative in release.EXPLORATORY_ANALYSES[2:]:
        links = [
            link for link in release.SOURCE_HASH_LINKS if link[0] == analysis_relative
        ]
        payload: dict[str, object] = {
            "status": "EXPLORATORY_POST_CONFIRMATORY"
        }
        for mapping_key in sorted({link[1] for link in links}):
            values = {
                link[2]: _sha(root / link[3])
                for link in links
                if link[1] == mapping_key
            }
            if mapping_key:
                payload[mapping_key] = values
            else:
                payload.update(values)
        _write(
            root / analysis_relative,
            json.dumps(payload),
        )

    external_ledgers = sorted(
        {
            link[0]
            for link in release.SOURCE_HASH_LINKS
            if link[0] not in release.EXPLORATORY_ANALYSES
        }
    )
    for ledger_relative in external_ledgers:
        ledger_links = [
            link for link in release.SOURCE_HASH_LINKS if link[0] == ledger_relative
        ]
        payload: dict[str, object] = {}
        for mapping_key in sorted({link[1] for link in ledger_links}):
            values = {
                link[2]: _sha(root / link[3])
                for link in ledger_links
                if link[1] == mapping_key
            }
            if mapping_key:
                payload[mapping_key] = values
            else:
                payload.update(values)
        _write(
            root / ledger_relative,
            json.dumps(payload),
        )

    figures = root / "pra_extension/final_figures"
    for name in sorted(release.REQUIRED_FINAL_FIGURES):
        content = (
            b"%PDF-1.4\n%%EOF\n"
            if name.endswith(".pdf")
            else b"not-a-real-png-but-content-addressed"
        )
        _write(figures / name, content)
    _write(
        figures / "figure_sha256.json",
        json.dumps(
            {
                name: _sha(figures / name)
                for name in sorted(release.REQUIRED_FINAL_FIGURES)
            }
        ),
    )
    paper = root / "paper/canonical_emulation/main.pdf"
    _write(paper, b"%PDF-1.4\n%%EOF\n")
    for relative in release.REQUIRED_PAPER_ARTIFACTS:
        _write(root / relative, b"%PDF-1.4\n%%EOF\n")

    return figures, paper


def test_release_manifest_is_deterministic_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figures, paper = _fake_repository(tmp_path)
    monkeypatch.setattr(release, "PINNED_AUTHORITATIVE_HASHES", {})
    monkeypatch.setattr(release, "PINNED_HISTORICAL_REGRESSION_HASHES", {})

    first = release.build_manifest(
        repo_root=tmp_path,
        figures_dir=figures,
        paper_pdf=paper,
    )
    second = release.build_manifest(
        repo_root=tmp_path,
        figures_dir=figures,
        paper_pdf=paper,
    )
    assert first == second
    assert first["canonical_frozen_identity"]["material_file_count"] > 0
    assert first["evidence_layers"] == {
        "canonical": "FROZEN_CONFIRMATORY",
        "dense_n4": "FROZEN_PROSPECTIVE_N4",
        "dense_secondary": "FROZEN_PROSPECTIVE_SECONDARY",
        "dark_sector": "FROZEN_VALIDATION",
        "pra_extension": "EXPLORATORY_POST_CONFIRMATORY",
    }
    assert "if-and-only-if growing reset-QELM emulation theorem" in first[
        "claim_boundary"
    ]
    assert "reachable-observable joint spectral radius and context law" in first[
        "claim_boundary"
    ]
    assert len(first["source_hash_checks"]) == len(release.SOURCE_HASH_LINKS)
    assert release.SHA256_RE.fullmatch(first["manifest_identity_sha256"])
    assert any(
        path.endswith("eiv_root_effects.csv") for path in first["artifacts"]
    )
    assert any(
        path.endswith("student_architecture_selected_effects.csv")
        for path in first["artifacts"]
    )
    assert any(
        path.endswith("observable_root_effects.csv") for path in first["artifacts"]
    )
    assert any(
        path.endswith("continuous_root_effects.csv") for path in first["artifacts"]
    )
    assert any(
        path.endswith("fig01_history_frontier_source.csv")
        for path in first["artifacts"]
    )
    assert "results_canonical_confirmatory_v1/stream_metrics.csv" in first["artifacts"]
    assert "results_dense_phase_secondary_v1/suffix_decomposition.csv" in first["artifacts"]
    assert (
        "results_dense_phase_secondary_v1_analysis/dense_phase_cells.csv"
        in first["artifacts"]
    )
    assert "results_dense_phase_n4_v1/suffix_decomposition.csv" in first["artifacts"]
    assert (
        "results_dense_phase_n4_v1_analysis/dense_n4_phase_cells.csv"
        in first["artifacts"]
    )
    assert (
        "pra_extension/outputs/dark_sector_validation/dark_sector_summary.json"
        in first["artifacts"]
    )
    assert "paper/canonical_emulation/main.pdf" in first["artifacts"]
    assert (
        "paper/figures/information_interface/phase_threshold_sensitivity.json"
        in first["artifacts"]
    )
    assert (
        first["artifacts"]["scripts/analyze_phase_threshold_sensitivity.py"]["role"]
        == "post_hoc_phase_threshold_analysis_code"
    )
    assert not any("supplement_main" in path for path in first["artifacts"])
    for relative in release.HISTORICAL_REGRESSION_FILES:
        assert first["artifacts"][relative]["role"] == (
            "historical_regression_fixture_only"
        )

    linked_source = tmp_path / release.SOURCE_HASH_LINKS[0][3]
    linked_source.write_text("value\nchanged\n", encoding="utf-8")
    with pytest.raises(release.ManifestError, match="embedded source hash mismatch"):
        release.build_manifest(
            repo_root=tmp_path,
            figures_dir=figures,
            paper_pdf=paper,
        )

    linked_source.write_text("value\n1\n", encoding="utf-8")
    paper.unlink()
    with pytest.raises(release.ManifestError, match="missing paper PDF"):
        release.build_manifest(
            repo_root=tmp_path,
            figures_dir=figures,
            paper_pdf=paper,
        )


def test_historical_regression_fixtures_are_separate_and_hash_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    historical = release.PINNED_HISTORICAL_REGRESSION_HASHES.copy()
    assert len(historical) == 7
    assert set(historical) == set(release.HISTORICAL_REGRESSION_FILES)
    assert not set(historical).intersection(release.AUTHORITATIVE_EXTENSION_FILES)
    assert not set(historical).intersection(release.PINNED_AUTHORITATIVE_HASHES)
    assert set(release.HISTORICAL_REGRESSION_FILES.values()) == {
        "historical_regression_fixture_only"
    }
    monkeypatch.setattr(release, "PINNED_AUTHORITATIVE_HASHES", {})
    for relative, expected in historical.items():
        assert _sha(root / relative) == expected
        fixture = tmp_path / relative
        _write(fixture, "original\n")
        monkeypatch.setattr(
            release, "PINNED_HISTORICAL_REGRESSION_HASHES", {relative: _sha(fixture)}
        )
        release._verify_pinned_hashes(tmp_path)
        _write(fixture, "mutated\n")
        with pytest.raises(
            release.ManifestError, match="pinned historical regression hash mismatch"
        ):
            release._verify_pinned_hashes(tmp_path)


def test_checked_in_release_manifest_matches_live_inputs() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = release.load_strict_json_object(
        root / "pra_extension/PRA_EXTENSION_RELEASE.json"
    )
    actual = release.build_manifest(repo_root=root)
    assert actual == expected
