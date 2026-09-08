from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper" / "canonical_emulation"
MAIN_SECTION_NAMES = (
    "abstract.tex",
    "introduction.tex",
    "related-work.tex",
    "models.tex",
    "background.tex",
    "dynamics.tex",
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
    compact_evaluation = re.sub(r"\s+", " ", evaluation)
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
    assert "time-homogeneous input alphabet" in compact_paper
    assert "Two histories differ only in one old flagged input" in compact_paper
    assert r"\label{eq:protected-mode}" in compact_paper
    assert r"\label{eq:visible-memory-theorem}" in compact_paper
    assert (
        "When Can Finite-Context QELMs Replace Quantum Reservoir Memory?"
        in compact_paper
    )
    assert "one fixed reservoir and a pair of trajectories" in compact_paper
    assert r"D_d\geq\delta\qquad\text{for every finite }d" in compact_paper
    assert "randomization cannot lower the scalar bound" in compact_paper
    assert "worst-case expected absolute error" in compact_paper
    assert "Identical windows therefore produce identical predictions for a fixed model" in compact_paper
    assert "their probability distributions remain identical" in compact_paper
    assert "With finitely sampled measurements" in compact_paper
    assert "Reset-window emulation theorem." in compact_paper
    # Prose stays normal-weight; only caption panel references remain bold.
    assert all(label in {"a", "b", "c", "a,b"}
               for label in re.findall(r"\\textbf\{([^{}]*)\}", paper))
    assert r"\bfseries" not in paper
    assert r"\inf_f\sup_h |y(h)-f(S_d(h))|" in compact_paper
    assert r"=\frac{D_d}{2}" in compact_paper
    assert r"(r+1)^d" in compact_paper
    assert "each block's Hamming weight" in compact_paper
    assert r"K_i\sim\operatorname{Binomial}(r,a_i)" in compact_paper
    assert r"\label{eq:qrc-qelm-bound}" in compact_paper
    assert r"D_d\longrightarrow0" in compact_paper
    assert "reachable-observable quotient" in compact_paper
    assert "joint spectral radius" in compact_paper
    assert "vector error between $D_d/2$ and $D_d$" in compact_paper
    assert "Any enclosing ball has radius at least half its diameter" in compact_paper
    assert "The six-qubit reset-QELM encodes" in compact_evaluation
    assert "These experiments evaluate a fixed feature architecture" in compact_evaluation
    assert re.search(r"\b(?:teacher|student)s?\b", paper, re.IGNORECASE) is None
    assert "representability with increasing resources" in compact_paper
    assert "does not guarantee efficient training" in compact_paper
    assert "or make the fixed six-qubit emulator" in compact_paper
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
    assert r"$8.33\times10^{-17}$" in compact_paper
    assert "every tested $(d,r)$ pair satisfies" in compact_paper
    assert qrc_rows[-1]["max_abs_error"] < qrc_rows[0]["max_abs_error"]
    assert history_rows[-1]["max_abs_error"] < history_rows[0]["max_abs_error"]
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
    assert r"At $B=72$ and $\gamma_T\in\{2,4\}$" in compact_evaluation
    assert r"increasing $d$ from one to six" in compact_evaluation
    assert "history limited in every root" in compact_evaluation
    assert r"the $B=18$ QELM usually fails while replay succeeds" in compact_evaluation
    assert "operational representation-limit diagnosis" in compact_evaluation
    assert "every root across the two-, three-, and four-qubit targets reaches the tolerance" in compact_evaluation
    classical = gates["secondary_qelm_vs_best_classical"]
    assert classical["passed"] is False
    assert classical["positive_roots"] == 0
    assert r"outperforms the $B=72$ QELM in every root" in compact_evaluation

    # Common-support dissipator comparisons, not unequal live-cell mixtures.
    common = analysis["observable_memory"]["common_live_support_sensitivity"]
    for family in ("local_loss", "collective_loss", "dephasing"):
        effect = common["effects"][family]["fractional_reduction"]
        assert common["effects"][family]["common_live_cells_total"] == 79
        assert effect["positive_roots"] == 12
    assert "All rootwise local-loss decay fits have negative slope" in compact_evaluation
    assert "fig02_information_frontier.pdf" not in paper
    assert "fig03_resource_diagnosis.pdf" in paper

    # Validation-selected errors-in-variables claim.
    policy = eiv["validation_selected_policy_Q19200_vs_Q300"]
    surface = eiv["validation_selected_policy_test_nrmse"]
    assert f"{surface['Q300']['median']:.3f}" in evaluation
    assert f"{surface['Q19200']['median']:.4f}" in evaluation
    assert f"{100 * policy['median']:.2f}\\%" in evaluation
    assert policy["positive_roots"] == policy["n_roots"] == 12
    assert "separate exploratory, post-confirmatory study" in compact_evaluation

    # Claim boundaries and journal-format deliverables.
    assert "exact boundary" not in paper
    assert "genuinely quantum" not in paper
    assert "arbitrary 80-symbol suffixes" not in paper
    assert "quadratic delay" in compact_paper
    assert "Supplemental Material" not in paper
    assert "supplement" not in paper.lower()
    assert paper.count(r"\begin{figure*") == 3
    assert "fig01_information_interface.pdf" in paper
    assert "fig02_information_frontier.pdf" not in paper
    assert "fig03_resource_diagnosis.pdf" in paper
    assert "fig04_long_context_n5.pdf" in paper
    assert "fig01_protected_excited_visible.pdf" not in paper
    assert "fig02_phase_map_compact.pdf" not in paper
    assert "fig03_dissipation_window_scaling_compact.pdf" not in paper
    for artifact in (
        PAPER_ROOT / "main.pdf",
        REPO_ROOT / "paper" / "icaart2027" / "submission.pdf",
    ):
        page_count = _pdf_page_count(artifact)
        assert 0 < page_count <= 12, f"{artifact.name}: {page_count} pages exceeds 12"
    # Check the delivered bibliography, not an ignored local BibTeX build file.
    destinations = subprocess.run(
        ["pdfinfo", "-dests", str(PAPER_ROOT / "main.pdf")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    reference_keys = set(re.findall(r'"cite\.([^"\n]+)"', destinations))
    cited_keys = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", paper)
        for key in group.split(",")
    }
    # Verify the delivered bibliography against the actual cited sources.
    assert reference_keys == cited_keys
    assert {
        "innocenti2023potential", "fujii2017harnessing", "boyd1985fading",
        "grigoryeva2018sas", "martinezpena2023finite", "mujal2023measurement",
        "kawanabe2026tdqelm", "goto2021universal", "yu2024nonasymptotic",
    } <= reference_keys
    assert not (PAPER_ROOT / "supplement_main.pdf").exists()
    for name in (
        "fig01_information_interface.pdf",
        "fig02_information_frontier.pdf",
        "fig03_resource_diagnosis.pdf",
    ):
        assert (
            REPO_ROOT / "paper" / "figures" / "information_interface" / name
        ).is_file()


def test_empirical_residual_is_not_claimed_as_pure_feature_approximation() -> None:
    equations = []
    for directory in (PAPER_ROOT, REPO_ROOT / "paper" / "icaart2027"):
        sections = {
            name: re.sub(r"\s+", " ", (directory / "sections" / f"{name}.tex")
                         .read_text(encoding="utf-8"))
            for name in ("abstract", "appendix", "methodology", "evaluation", "conclusion")
        }
        # The active ICAART revision groups this sampled diagnostic with its
        # numerical methods in E.1; the archived edition retains it in D.
        diagnostic = sections[
            "methodology" if directory.name == "icaart2027" else "appendix"
        ]
        assert re.search(
            r"[Rr]esidual(?: emulator)? error (?:can (?:reflect|include)|includes) "
            r"feature approximation",
            diagnostic,
        )
        for component in ("finite-data estimation", "ridge regularization",
                          "suffix-mean sampling variability"):
            assert component in diagnostic
        assert re.search(
            r"(?:without isolating feature approximation in the residual|"
            r"not an isolated feature[- ]approximation error)", diagnostic,
        )
        assert re.search(
            r"[Rr]epresentation limited (?:is an operational diagnosis|operationally means|means)",
            sections["methodology"],
        )
        assert "history passes, the fitted QELM fails, and replay succeeds" in sections["methodology"]
        assert re.search(
            r"(?:does not (?:isolate|identify)(?: the)?.*?"
            r"best approximation (?:attainable with the|using|available from the) QELM features|"
            r"not that the best approximation using QELM features fails)",
            sections["methodology"],
        )
        assert "common-suffix residual emulator NRMSE" in sections["evaluation"]
        assert ("the fitted model must still represent the resulting input--output "
                "relationship accurately") in sections["conclusion"]
        assert ("the theoretical existence result guarantees neither efficient learning "
                "nor resource savings") in sections["conclusion"]
        assert "The experiments therefore establish no quantum advantage." in sections["conclusion"]
        assert ("examine the practical roles of window length, feature capacity, "
                "and measurement budgets") in sections["abstract"]
        for name in ("evaluation", "conclusion"):
            assert "representation NRMSE" not in sections[name]
        # The empirical identity, unlike its interpretation, is unchanged.
        equation = re.search(
            r"\\begin\{equation\} ((?:(?!\\end\{equation\}).)*"
            r"\\label\{eq:panel-decomposition\}.*?)\\end\{equation\}", diagnostic,
        )
        assert equation is not None
        equations.append(equation.group(1))
        assert r"\frac1m\mathbb E_{s,k}\|y_{sk}-\widehat y_s\|_2^2" in equation.group(1)
        assert r"\frac1m\mathbb E_{s,k}\|y_{sk}-\bar y_s\|_2^2" in equation.group(1)
        assert r"\frac1m\mathbb E_s\|\bar y_s-\widehat y_s\|_2^2" in equation.group(1)
    assert equations[0] == equations[1]


def test_icaart_appendix_rewrite_covers_full_body_and_relocated_material() -> None:
    """The supplied A--E rewrite replaces the appendix, not the paper template."""

    directory = REPO_ROOT / "paper" / "icaart2027"
    appendix = (directory / "sections" / "appendix.tex").read_text(encoding="utf-8")
    methods = (directory / "sections" / "methodology.tex").read_text(encoding="utf-8")
    dynamics = (directory / "sections" / "dynamics.tex").read_text(encoding="utf-8")
    main = (directory / "main.tex").read_text(encoding="utf-8")
    combined = appendix + "\n" + methods

    titles = re.findall(r"\\subsection\{([^}]+)\}", appendix)
    expected_titles = (
        r"Optimal Window Error(?: and (?:QELM )?Emulation)?",
        r"Observable Memory and (?:Required )?Context",
        r"Explicit QELM Resource Bounds",
        r"Protected Memory(?: Beyond Every Finite Window)?",
    )
    assert len(titles) == len(expected_titles)
    assert all(re.fullmatch(pattern, title) for pattern, title in zip(expected_titles, titles))
    method_titles = re.findall(r"\\subsection\{([^}]+)\}", methods)
    assert len(method_titles) == 1
    assert re.fullmatch(r"Numerical Evaluation(?: and Reproducibility)?", method_titles[0])
    for label in (
        "app:window-laws", "app:observable-proof", "app:bernstein-resources",
        "app:protected-calibration", "app:protected-pulse", "app:protected-implementation",
    ):
        assert appendix.count(rf"\label{{{label}}}") == 1
        assert rf"\label{{{label}}}" not in methods
    for label in ("sec:methods", "app:common-suffix-diagnostic", "eq:panel-decomposition"):
        assert methods.count(rf"\label{{{label}}}") == 1
        assert rf"\label{{{label}}}" not in appendix

    # The implementation is D.2 and the sampled diagnostic is E.1.
    protected = appendix.split(r"\label{app:protected-calibration}", 1)[1]
    protected_parts = re.split(r"\\subsubsection\{", protected)
    assert len(protected_parts) == 3
    assert r"\label{app:protected-pulse}" in protected_parts[1]
    assert r"\label{app:protected-implementation}" in protected_parts[2]
    assert re.search(r"H\(u\)\s*&?=", protected_parts[2])
    method_parts = re.split(r"\\subsubsection\{", methods)
    assert len(method_parts) == 5
    assert r"\label{app:common-suffix-diagnostic}" in method_parts[1]
    assert r"\label{eq:panel-decomposition}" in method_parts[1]
    assert r"\label{eq:target-qrc-model}" in method_parts[2]
    # Cohorts and fitting remain explicit inside E.2, with compact run-in
    # headings instead of extra native subsection whitespace.
    assert r"\emph{Cohorts and held-out data.}" in method_parts[2]
    assert r"\emph{Fitting and decision rules.}" in method_parts[2]
    assert method_parts[3].startswith("Finite-Shot Extension}")
    assert method_parts[4].startswith("Long-Context Extension}")
    for topic in (
        "Why the midpoint is optimal.", "Why QELMs can approach it.",
        "When longer context suffices.", "Remove invisible differences.",
        "Measure the remaining memory.", "Relate memory size to output error.",
        "Stability and context length.", "Assumptions and replay error.",
        "Repeated encoding approximates replay.", "Three error contributions.",
        "Approaching the optimal information floor.",
    ):
        assert rf"\emph{{{topic}}}" in appendix
    assert r"\ref{app:protected-pulse}" in dynamics
    assert r"\ref{app:protected-implementation}" in dynamics
    assert r"\ref{sec:methods}" not in dynamics

    for preview_only in (
        "Proposed replacement text", "Appendix revision", r"\documentclass",
        r"\begin{document}", r"\end{document}", r"\textbf", r"\bfseries",
    ):
        assert preview_only not in combined
    assert main.index(r"\bibliography{references}") < main.index(r"\input{sections/appendix}")
    assert main.index(r"\input{sections/appendix}") < main.index(r"\input{sections/methodology}")


def test_icaart_appendix_rewrite_preserves_equation_sequence_12_to_26() -> None:
    directory = REPO_ROOT / "paper" / "icaart2027" / "sections"
    combined = "\n".join(
        (directory / f"{name}.tex").read_text(encoding="utf-8")
        for name in ("appendix", "methodology")
    )
    equations = re.findall(
        r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", combined, re.DOTALL,
    )
    labels = (
        "eq:replay-map", "eq:visible-quotient", "eq:visible-radius-formal",
        "eq:observable-norm", "eq:norm-equivalence", "eq:diameter-product",
        "eq:diameter-product-equivalence", "eq:incremental-lipschitz",
        "eq:weighted-input-bound",
        "eq:qrc-qelm-bound", "eq:midpoint-resource-bound",
        "eq:protected-pulse-population",
        "eq:protected-pair-mse", "eq:panel-decomposition", "eq:target-qrc-model",
    )
    assert len(equations) == len(labels) == 15
    for number, ((environment, body), label) in enumerate(zip(equations, labels), start=12):
        assert rf"\label{{{label}}}" in body, f"Equation ({number}): {label} moved"
        if environment == "align":
            # The target Hamiltonian and evolution share one equation number.
            assert number == 26
            assert body.count(r"\nonumber") == 1
    assert r"\sum_{i=1}^d\eta^{d-i}" in re.sub(r"\s+", "", equations[8][1])
    assert r"\sin^2(\theta/2)" in equations[11][1]
    assert not re.search(r"\\(?:setcounter|addtocounter)\{equation\}|\\tag\{", combined)


def test_conclusion_keeps_four_matching_normal_weight_paragraphs() -> None:
    versions = []
    for directory in (PAPER_ROOT, REPO_ROOT / "paper" / "icaart2027"):
        source = (directory / "sections" / "conclusion.tex").read_text(encoding="utf-8")
        body = re.sub(r"%[^\n]*", "", source)
        body = re.sub(r"\\section\{[^{}]+\}|\\label\{[^{}]+\}", "", body).strip()
        paragraphs = [re.sub(r"\s+", " ", paragraph).strip()
                      for paragraph in re.split(r"\n\s*\n", body) if paragraph.strip()]
        assert len(paragraphs) == 4
        assert r"\textbf" not in body and r"\bfseries" not in body
        assert r"\paragraph" not in body and r"\subsection" not in body
        assert r"Section~\ref{sec:criterion}" in body
        assert not re.search(r"Section\s*~?\s*\d", body)
        versions.append(paragraphs)
    assert versions[0] == versions[1]


def test_added_primary_references_are_cited_in_both_versions() -> None:
    additions = {
        "chen2020temporal", "sannia2024dissipation", "cindrak2024restriction",
        "zanardi1997noiseless", "hoeffding1963probability",
    }
    citation_sets = []
    for directory in (PAPER_ROOT, REPO_ROOT / "paper" / "icaart2027"):
        sections = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((directory / "sections").glob("*.tex"))
        )
        cited = {
            key.strip()
            for group in re.findall(r"\\cite\{([^}]+)\}", sections)
            for key in group.split(",")
        }
        assert additions <= cited
        citation_sets.append(cited)
        bibliography = (directory / "references.bib").read_text(encoding="utf-8")
        assert "Chen, Jiayin and Nurdin" in bibliography
        assert "Chen, Jinglei" not in bibliography
        assert "10.1080/01621459.1963.10500830" in bibliography
    assert citation_sets[0] == citation_sets[1]


def test_repository_reference_is_at_end_and_review_version_is_anonymous() -> None:
    journal_main = (PAPER_ROOT / "main.tex").read_text(encoding="utf-8")
    repository_url = "https://github.com/eybmits/qrc_vs_qelm"
    assert journal_main.index(repository_url) > journal_main.index(r"\bibliography{references}")
    methods = (PAPER_ROOT / "sections" / "methodology.tex").read_text(encoding="utf-8")
    assert repository_url not in methods
    conference = REPO_ROOT / "paper" / "icaart2027"
    main = (conference / "main.tex").read_text(encoding="utf-8")
    assert main.rfind(r"\input{sections/methodology}") > main.rfind(r"\bibliography{references}")
    conference_methods = (conference / "sections" / "methodology.tex").read_text(encoding="utf-8")
    assert "link is withheld for double-blind review" in conference_methods
    assert repository_url not in main + conference_methods


def test_main_figure_selection_preserves_archived_frontier_and_resolves_references() -> None:
    retained = [
        "fig01_information_interface.pdf",
        "fig03_resource_diagnosis.pdf",
        "fig04_long_context_n5.pdf",
    ]
    for directory in (PAPER_ROOT, REPO_ROOT / "paper" / "icaart2027"):
        main = (directory / "main.tex").read_text(encoding="utf-8")
        section_files = re.findall(r"\\input\{sections/([^}]+)\}", main)
        source = "\n".join([
            main,
            *((directory / "sections" / f"{name}.tex").read_text(encoding="utf-8")
              for name in section_files),
        ])
        assert source.count(r"\begin{figure*}") == 3, directory.name
        assert re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source,
        ) == retained, directory.name
        assert "fig02_information_frontier.pdf" not in source, directory.name
        assert "fig:information-frontier" not in source, directory.name
        labels = set(re.findall(r"\\label\{([^}]+)\}", source))
        references = set(re.findall(
            r"\\(?:ref|eqref|pageref|autoref)\{([^}]+)\}", source,
        ))
        assert references <= labels, (directory.name, sorted(references - labels))

    # Removing a redundant manuscript float does not delete its reproducible
    # graphic or source measurements. Renderer/evidence tests retain their
    # independent checksums and numerical contracts for these archived assets.
    archived = REPO_ROOT / "paper" / "figures" / "information_interface"
    for name in (
        "fig02_information_frontier.pdf",
        "fig02_information_frontier.png",
        "fig02_information_frontier_source.csv",
    ):
        assert (archived / name).is_file(), name
        assert (archived / name).stat().st_size > 0, name


def test_main_figures_are_spread_across_distinct_pages() -> None:
    """Keep full-width figures separate and clear them before the conclusion."""
    for directory, caption_prefix in (
        (PAPER_ROOT, r"FIG\."),
        (REPO_ROOT / "paper" / "icaart2027", "Figure"),
    ):
        filename = "main.pdf" if directory == PAPER_ROOT else "submission.pdf"
        pages = subprocess.run(
            ["pdftotext", "-layout", str(directory / filename), "-"],
            check=True, capture_output=True, text=True,
        ).stdout.split("\f")
        assert "??" not in "\n".join(pages), filename
        pattern = rf"(?m)^\s*{caption_prefix}\s+(\d+)[.:]\s"
        captions = [
            (int(match.group(1)), number)
            for number, page in enumerate(pages, 1)
            for match in re.finditer(pattern, page)
        ]
        assert [number for number, _ in captions] == [1, 2, 3], (filename, captions)
        positions = [page for _, page in captions]
        assert positions == sorted(set(positions)), (filename, positions)
        assert positions[-1] - positions[0] >= 3, (filename, positions)
        main = (directory / "main.tex").read_text(encoding="utf-8")
        assert r"\setcounter{dbltopnumber}{1}" in main
        assert main.index(r"\input{sections/evaluation}") < main.index(
            r"\FloatBarrier"
        ) < main.index(r"\input{sections/conclusion}")


def test_long_context_claims_preserve_the_separate_cohort_and_negative_controls() -> None:
    import pandas as pd

    study = REPO_ROOT / "results_n5_long_context_v1"
    summary = json.loads((study / "analysis/long_n5_summary.json").read_text())
    comparisons = pd.read_csv(study / "analysis/long_n5_classical_root_comparisons.csv")
    assert summary["integrity"]["passed"]
    assert summary["integrity"]["live_systems"] == 24
    assert summary["old_gates_changed"] is False
    assert len(comparisons) == 120
    assert int(comparisons["qelm_relative_gain_vs_poly2"].lt(0).sum()) == 118
    for version in ("canonical_emulation", "icaart2027"):
        text = " ".join((REPO_ROOT / f"paper/{version}/sections/evaluation.tex").read_text().split())
        assert "five-qubit targets, 12 new roots, and a ten-qubit QELM" in text
        assert "118 of 120 streaming root--condition comparisons" in text
        assert "no tested QELM budget meets the total-error rule" in text
        for gamma in (1, 4):
            effect = summary["analysis"]["paired_history"][f"history_gamma{gamma}_d2_to_d10"]
            assert effect["positive_roots"] == effect["n_roots"] == 12
            assert f"{100 * effect['median']:.2f}\\%" in text


def test_main_manuscript_uses_direct_scientific_voice() -> None:
    text = re.sub(r"\s+", " ", _paper_text()).lower()
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


def test_accessible_structure_introduces_models_before_limits_and_results() -> None:
    main = (PAPER_ROOT / "main.tex").read_text(encoding="utf-8")
    section_files = re.findall(r"\\input\{sections/([^}]+)\}", main)
    assert section_files == [
        "abstract", "introduction", "models", "background", "dynamics",
        "evaluation", "conclusion", "appendix", "methodology",
    ]
    sections = {
        name: (PAPER_ROOT / "sections" / f"{name}.tex").read_text(encoding="utf-8")
        for name in section_files
    }
    main_names = section_files[1:7]
    assert [
        re.findall(r"\\section\{([^}]+)\}", sections[name])
        for name in main_names
    ] == [
        ["Introduction"],
        ["Quantum Reservoirs and Reset-Window Models"],
        ["Limits of Finite-Window Emulation"],
        ["Dynamical Origins of Observable Memory"],
        ["Numerical Experiments"],
        ["Discussion and Conclusions"],
    ]
    for name in ("introduction", "models", "background", "conclusion"):
        assert r"\subsection{" not in sections[name]
    assert re.findall(r"\\subsection\{([^}]+)\}", sections["dynamics"]) == [
        "Observable Forgetting", "Required Window Length", "Protected Memory",
    ]
    assert re.findall(r"\\subsection\{([^}]+)\}", sections["evaluation"]) == [
        "Models and Evaluation", "History Requirements",
        "Feature Capacity and Classical Comparisons", "Measurement Budgets",
    ]
    # Declare the full-width overview early so it can float to page two.
    assert "fig01_information_interface.pdf" not in sections["models"]
    introduction = sections["introduction"]
    assert introduction.index("an appropriate input representation.") < (
        introduction.index("fig01_information_interface.pdf")
    ) < introduction.index("Quantum extreme learning machines (QELMs)")
    assert r"Figure~\ref{fig:information-interface}" in sections["models"]
    conference_sections = REPO_ROOT / "paper" / "icaart2027" / "sections"
    conference_introduction = (conference_sections / "introduction.tex").read_text(
        encoding="utf-8"
    )
    conference_models = (conference_sections / "models.tex").read_text(encoding="utf-8")
    assert "fig01_information_interface.pdf" not in conference_models
    assert conference_introduction.index(
        "an appropriate input representation."
    ) < conference_introduction.index(
        "fig01_information_interface.pdf"
    ) < conference_introduction.index("Quantum extreme learning machines (QELMs)")
    assert r"Figure~\ref{fig:information-interface}" in conference_models
    evaluation = sections["evaluation"]
    assert evaluation.index("Each independent realization block") < evaluation.index(
        "To distinguish missing history"
    )
    # The information-bound section retains its theorem content without the
    # redundant frontier float. Remaining declarations stay distributed.
    assert r"\label{eq:window-law}" in sections["background"]
    assert r"\begin{figure" not in sections["background"]
    assert "fig02_information_frontier.pdf" not in sections["background"]
    assert "fig03_resource_diagnosis.pdf" in sections["dynamics"]
    assert "fig04_long_context_n5.pdf" in evaluation
    assert re.findall(r"\\section\{([^}]+)\}", sections["appendix"]) == [
        "Mathematical proofs and constructions"
    ]
    assert re.findall(r"\\section\{([^}]+)\}", sections["methodology"]) == [
        "Numerical methods and reproducibility"
    ]
