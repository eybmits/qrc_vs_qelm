from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper" / "canonical_emulation"
MAIN_SECTION_NAMES = (
    "abstract.tex",
    "introduction.tex",
    "related-work.tex",
    "background.tex",
    "evaluation.tex",
    "conclusion.tex",
    "methodology.tex",
    "appendix.tex",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(relative: str) -> dict[str, object]:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def _paper_text() -> str:
    paths = [
        PAPER_ROOT / "main.tex",
        *(PAPER_ROOT / "sections" / name for name in MAIN_SECTION_NAMES),
    ]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _pdf_page_count(path: Path) -> int:
    """Count pdfTeX page objects without adding a non-frozen test dependency."""

    raw = path.read_bytes()
    payloads = [raw]
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.DOTALL):
        try:
            payloads.append(zlib.decompress(match.group(1).strip()))
        except zlib.error:
            continue
    return sum(len(re.findall(rb"/Type\s*/Page\b", data)) for data in payloads)


def test_authoritative_manuscript_claims_match_generated_artifacts() -> None:
    paper = _paper_text()
    compact_paper = re.sub(r"\s+", " ", paper)
    evaluation = (PAPER_ROOT / "sections" / "evaluation.tex").read_text(
        encoding="utf-8"
    )
    dark = _json(
        "pra_extension/outputs/dark_sector_validation/dark_sector_summary.json"
    )
    gates = _json("results_canonical_confirmatory_v1/canonical_gate_report.json")
    analysis = _json(
        "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
        "pra_extension_analysis.json"
    )
    eiv = _json(
        "pra_extension/outputs/finite_shots_eiv_n4_analysis_v6_final_helpers_"
        "strict/eiv_analysis.json"
    )
    bernstein = _json(
        "pra_extension/outputs/bernstein_qelm_validation_v1/"
        "bernstein_validation_summary.json"
    )

    # Exact protected-memory witness and its declared scope.
    assert dark["all_primary_passed"] is True
    assert dark["n_roots"] == 12
    assert dark["suffix_steps"] == 80
    assert f"{dark['expected_prefix_gap']:.10f}" in paper
    assert f"{dark['expected_balanced_mse_floor']:.10f}" in paper
    assert f"{dark['expected_worst_case_absolute_floor']:.7f}" in paper
    assert "time-homogeneous alphabet" in paper
    assert "differ at the old flagged input only" in compact_paper
    assert "outside the fading-memory regime" in compact_paper
    assert "every finite allowed suffix" in compact_paper
    assert "When Can a Reset-Window QELM Reproduce a Quantum Reservoir?" in compact_paper
    assert "fixed recurrent process and fixed trajectory pair" in compact_paper
    assert "Randomization does not remove the obstruction" in compact_paper
    assert r"\textit{Reset-window emulation.}" in compact_paper
    assert r"E_d&=\inf_{g_d}\sup_h" in compact_paper
    assert r"=\frac{D_d}{2}" in compact_paper
    assert r"B=(r+1)^d" in compact_paper
    assert "blockwise Hamming weights" in compact_paper
    assert r"\operatorname{Binomial}(r,x_i)" in compact_paper
    assert r"\label{eq:qrc-qelm-bound}" in compact_paper
    assert "continuous fading-memory scalar QRC filter" in compact_paper
    assert "one fixed six-qubit windowed QELM" in compact_paper
    assert "neither universality to a fixed QELM architecture" in compact_paper
    for name, expected in dark["artifact_sha256"].items():
        artifact = (
            REPO_ROOT / "pra_extension" / "outputs" / "dark_sector_validation" / name
        )
        assert _sha256(artifact) == expected
    assert _sha256(REPO_ROOT / "configs" / "dark_sector_validation.json") == dark[
        "config_sha256"
    ]

    # Deterministic constructive QRC-to-QELM audit.
    assert bernstein["schema_version"] == "bernstein_qelm_validation_v1"
    feature_audit = bernstein["feature_probability_audit"]
    assert feature_audit["max_quantum_vs_binomial_error"] < 1e-15
    qrc_rows = bernstein["qrc_bernstein_convergence"]
    history_rows = bernstein["qrc_history_convergence"]
    assert f"{qrc_rows[0]['max_abs_error']:.6f}" in compact_paper
    assert f"{qrc_rows[-1]['max_abs_error']:.7f}" in compact_paper
    assert f"{history_rows[0]['max_abs_error']:.6f}" in compact_paper
    assert all(
        row["max_abs_error"] <= row["certified_max_error_bound"]
        for row in qrc_rows
    )
    assert all(
        row["max_abs_error"] <= row["certified_max_history_error_bound"]
        for row in history_rows
    )

    # Frozen history-before-features phase map.
    assert gates["decision"] == "CONFIRMS_FINITE_RESOURCE_FRONTIER"
    assert all(gates["primary_gates"][f"G{index}"]["passed"] for index in range(7))
    g4 = gates["primary_gates"]["G4"]
    g5 = gates["primary_gates"]["G5"]
    assert f"{100 * g4['median_gain']:.2f}\\%" in evaluation
    assert f"{100 * g5['median_gain']:.2f}\\%" in evaluation
    assert g4["positive_roots"] == g5["positive_roots"] == 12
    assert "all 24 cases---12 independent roots for each of two" in evaluation
    assert "22/24 cells are representation" in evaluation
    assert "all 24 cells reach the $0.10$ tolerance" in evaluation
    classical = gates["secondary_qelm_vs_best_classical"]
    assert classical["passed"] is False
    assert classical["positive_roots"] == 0
    assert f"{classical['median_gain']:.3f}" in evaluation

    # Common-support dissipator comparisons, not unequal live-cell mixtures.
    common = analysis["observable_memory"]["common_live_support_sensitivity"]
    for family in ("local_loss", "collective_loss", "dephasing"):
        effect = common["effects"][family]["fractional_reduction"]
        assert common["effects"][family]["common_live_cells_total"] == 79
        assert f"{100 * effect['median']:.2f}\\%" in evaluation
        assert effect["positive_roots"] == 12
    assert "fig02_information_frontier.pdf" in paper
    assert "fig03_resource_diagnosis.pdf" in paper

    # Validation-selected errors-in-variables claim.
    policy = eiv["validation_selected_policy_Q19200_vs_Q300"]
    surface = eiv["validation_selected_policy_test_nrmse"]
    assert f"{surface['Q300']['median']:.3f}" in evaluation
    assert f"{surface['Q19200']['median']:.4f}" in evaluation
    assert f"{100 * policy['median']:.2f}\\%" in evaluation
    assert policy["positive_roots"] == policy["n_roots"] == 12

    # Claim boundaries and journal-format deliverables.
    assert "exact boundary" not in paper
    assert "genuinely quantum" not in paper
    assert "arbitrary 80-symbol suffixes" not in paper
    assert "quadratic delay" in compact_paper
    assert "Supplemental Material" not in paper
    assert "supplement" not in paper.lower()
    assert paper.count(r"\begin{figure*") == 3
    assert "fig01_information_interface.pdf" in paper
    assert "fig02_information_frontier.pdf" in paper
    assert "fig03_resource_diagnosis.pdf" in paper
    assert "fig01_protected_excited_visible.pdf" not in paper
    assert "fig02_phase_map_compact.pdf" not in paper
    assert "fig03_dissipation_window_scaling_compact.pdf" not in paper
    page_count = _pdf_page_count(PAPER_ROOT / "main.pdf")
    assert 0 < page_count <= 12
    bibliography = (PAPER_ROOT / "main.bbl").read_text(encoding="utf-8")
    reference_count = len(re.findall(r"^\\bibitem\b", bibliography, re.MULTILINE))
    assert 40 <= reference_count <= 50
    assert not (PAPER_ROOT / "supplement_main.pdf").exists()
    for name in (
        "fig01_information_interface.pdf",
        "fig02_information_frontier.pdf",
        "fig03_resource_diagnosis.pdf",
    ):
        assert (
            REPO_ROOT / "paper" / "figures" / "information_interface" / name
        ).is_file()


def test_main_manuscript_uses_direct_scientific_voice() -> None:
    paths = [
        PAPER_ROOT / "main.tex",
        *(
            PAPER_ROOT / "sections" / name
            for name in (
                "abstract.tex",
                "introduction.tex",
                "related-work.tex",
                "background.tex",
                "methodology.tex",
                "evaluation.tex",
                "conclusion.tex",
                "appendix.tex",
            )
        ),
    ]
    text = re.sub(
        r"\s+",
        " ",
        "\n".join(path.read_text(encoding="utf-8") for path in paths),
    ).lower()
    forbidden = (
        "to our knowledge",
        "first explicit",
        "novelty boundary",
        "exact contribution and scope",
        "narrow priority",
        "we claim",
        "we do not claim",
        "not a claim that",
        "does not establish quantum",
        "prevents a quantum-advantage interpretation",
        "the scope is precise",
        "referee",
        "qrc--qelm equivalence",
        "qrc--qelm universal equivalence",
        "new universal-approximation theorem",
    )
    for phrase in forbidden:
        assert phrase not in text
