from __future__ import annotations

import hashlib
import json
import math
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
            r"[Rr]esidual(?:(?: emulator)? error)? (?:can (?:reflect|include)|includes|combines) "
            r"feature approximation",
            diagnostic,
        )
        for component in ("finite-data estimation", "ridge regularization",
                          "suffix-mean sampling variability"):
            assert component in diagnostic
        assert re.search(
            r"(?:without isolating feature approximation in the residual|"
            r"not an isolated feature[- ]approximation error|"
            r"it does not isolate feature approximation)", diagnostic,
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
        if directory.name == "icaart2027":
            assert "fitted accuracy also depends on representation and measurement precision" in sections["conclusion"].lower()
            assert "representability, not efficient training or resource savings" in sections["conclusion"]
            assert re.search(r"(?:without establishing a quantum advantage|not (?:evidence of )?quantum advantage)",
                             sections["evaluation"])
        else:
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
    assert len(method_parts) == 6
    assert r"\label{app:common-suffix-diagnostic}" in method_parts[1]
    assert r"\label{eq:panel-decomposition}" in method_parts[1]
    assert r"\label{eq:target-qrc-model}" in method_parts[2]
    # Cohorts and fitting remain explicit inside E.2, with compact run-in
    # headings instead of extra native subsection whitespace.
    assert r"\emph{Cohorts and held-out data.}" in method_parts[2]
    assert r"\emph{Fitting and decision rules.}" in method_parts[2]
    assert method_parts[3].startswith("Finite-Shot Extension}")
    assert method_parts[4].startswith("Long-Context Extension}")
    assert method_parts[5].startswith("Time-Series Demonstrations}")
    assert r"\label{app:task-forecasting}" in method_parts[5]
    assert r"\label{app:laser-demonstration}" in method_parts[5]
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


def test_conclusion_keeps_a_plain_story_and_scoped_claims() -> None:
    for directory in (PAPER_ROOT, REPO_ROOT / "paper" / "icaart2027"):
        source = (directory / "sections" / "conclusion.tex").read_text(encoding="utf-8")
        body = re.sub(r"%[^\n]*", "", source)
        body = re.sub(r"\\section\{[^{}]+\}|\\label\{[^{}]+\}", "", body).strip()
        paragraphs = [re.sub(r"\s+", " ", paragraph).strip()
                      for paragraph in re.split(r"\n\s*\n", body) if paragraph.strip()]
        assert len(paragraphs) == (4 if directory == PAPER_ROOT else 3)
        assert r"\textbf" not in body and r"\bfseries" not in body
        assert r"\paragraph" not in body and r"\subsection" not in body
        compact = re.sub(r"\s+", " ", body)
        if directory == PAPER_ROOT:
            assert r"Section~\ref{sec:criterion}" in body
            assert "neither efficient learning nor resource savings" in compact
            assert "no quantum advantage" in compact
        else:
            # Active edition: the within-pair replacement decision comes first.
            # Exact scalar minimax formulas stay in the theory, while the
            # high-level iff claim retains growing capacity and exact means.
            assert paragraphs[0].startswith("For time-series emulation, the choice between")
            for qualifier in ("chosen outputs", "Under our assumptions", "precisely when",
                              "growing windows", "sufficient QELM capacity",
                              "exact expectations", "vanishes uniformly",
                              "error floor that no additional window-model capacity removes"):
                assert qualifier in paragraphs[0]
            assert "Consider a QELM" in paragraphs[0]
            assert "retain the QRC" in paragraphs[0]
            assert paragraphs[1].startswith("This is an information criterion, not a guarantee")
            for limit in ("fitted accuracy", "representation and measurement precision",
                          "not efficient training or resource savings"):
                assert limit in paragraphs[1].lower()
            assert "Quadratic delay models outperform" not in compact
            assert "no quantum advantage" not in compact
            assert paragraphs[2].startswith("The resulting workflow is to establish how much history is needed")
            assert "errors requiring access to older inputs" in paragraphs[2]
            assert paragraphs[2].endswith("a principled basis for retaining or replacing the QRC.")
            assert "$" not in body
            assert len(compact.split()) <= 220
        assert "uniformly" in compact
        assert not re.search(r"Section\s*~?\s*\d", body)


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
    # Only ICAART is active; its new measured-input experiment cites its source.
    assert citation_sets[1] == citation_sets[0] | {"weigend1993timeseries"}


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
    assert re.search(
        r"(?:link is withheld for double-blind review|Code link withheld for anonymity)",
        conference_methods,
    )
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
        expected = (retained[:2] + ["fig04_memory_prediction.pdf"]
                    if directory.name == "icaart2027" else retained)
        assert re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source,
        ) == expected, directory.name
        assert source.count(r"\begin{figure}") == 0
        if directory.name == "icaart2027":
            assert "fig05_exact_sharpness.pdf" not in source
            assert "fig:exact-sharpness" not in source
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
    """Keep the three full-width figures separate and all floats before conclusion."""
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
        if version == "canonical_emulation":
            assert "five-qubit targets, 12 new roots, and a ten-qubit QELM" in text
        else:
            assert "prospective four-/five-qubit follow-ups each use 12 roots" in text
            assert "five-qubit study uses a ten-qubit QELM" in text
            assert "cohorts and dense-map seeds are disjoint" in text
        assert "118 of 120 streaming root--condition comparisons" in text
        assert "no tested QELM budget meets the total-error rule" in text
        for gamma in (1, 4):
            effect = summary["analysis"]["paired_history"][f"history_gamma{gamma}_d2_to_d10"]
            assert effect["positive_roots"] == effect["n_roots"] == 12
            assert f"{100 * effect['median']:.2f}\\%" in text


def test_icaart_exact_benchmarks_keep_global_derivation_and_construction_scope() -> None:
    """Check the new controlled examples without conflating them with learned cohorts."""
    from sharpness_benchmarks import convergence_rows, zero_variance_probability

    directory = REPO_ROOT / "paper" / "icaart2027" / "sections"
    sections = {
        name: re.sub(r"\s+", " ", (directory / f"{name}.tex").read_text(encoding="utf-8"))
        for name in ("abstract", "introduction", "evaluation", "appendix", "methodology")
    }
    evaluation, appendix, methods = (sections[name] for name in ("evaluation", "appendix", "methodology"))
    benchmark = evaluation.split(r"\label{sec:sharpness}", 1)[1]
    # The author-restored Figure 3 again includes the analytical reference;
    # direct NARMA prediction remains a separate numerical follow-up.
    presentation = benchmark
    assert r"\subsection{Exact Sharpness Benchmarks}" in evaluation
    assert "Separate post hoc construction checks" in benchmark
    assert "classical leaky-memory process embedded in a qubit" in benchmark
    assert r"g(a)=a^2" in benchmark and r"(e^{2a}-1)/(e^2-1)" in benchmark
    assert r"D_d=\eta^d" in benchmark
    assert "two encoding copies per input and an analytical readout" in benchmark
    assert "attain $D_d/2$ exactly" in benchmark
    assert "constructed readouts, not learning or resource efficiency" in benchmark
    assert "factorized exact expectations" in presentation
    assert "not the best readout at each copy count" in presentation
    assert "no training" in presentation
    assert r"Figure~\ref{fig:long-context}(c)" in benchmark
    assert "$0.281322$ at one copy" in benchmark
    assert "$0.084753$ at 256 copies" in benchmark
    assert "Analytically constructed examples" in sections["abstract"]
    assert "Exactly solvable examples" in sections["introduction"]

    # The analytical interval and factorial moment establish sharpness;
    # testing a handful of windows alone does not certify a supremum.
    assert "in closure" in appendix
    assert r"\eta^d/2+\|f-m_d\|_\infty" in appendix
    assert r"\eta^d(1-\eta^L)" in appendix
    assert r"K_i(K_i-1)/[r(r-1)]" in appendix and r"r\geq2" in appendix
    assert r"\max_{a\in[0,1]}b_r(a)" in appendix
    assert re.search(
        r"(?:global sharpness follows from the identities|the identities establish global sharpness)",
        appendix,
    )
    assert "Midpoint coefficients" in appendix and "define a separate expectation" in appendix
    assert r"D_d\sqrt{p(1-p)}" in methods
    assert "Neither it nor the correction estimates $D_d$ or $r_{\\rm vis}$" in methods
    assert f"{100 * zero_variance_probability(0.01):.2f}\\%" in methods

    ledger = _json("paper/figures/sharpness/fig05_exact_sharpness_source.json")
    expected = convergence_rows(**ledger["parameters"])
    assert ledger["numeric_source"] == "sharpness_benchmarks.convergence_rows"
    assert any("not full statevectors" in caveat for caveat in ledger["scope"])
    assert any("not a capacity-optimal envelope" in caveat for caveat in ledger["scope"])
    assert ledger["parameters"] == {
        "d": 8, "eta": 0.8, "beta": 2.0, "copies": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    }
    assert len(ledger["rows"]) == len(expected)
    for recorded, recomputed in zip(ledger["rows"], expected):
        assert recorded.keys() == recomputed.keys()
        assert all(math.isclose(recorded[key], value, rel_tol=1e-12, abs_tol=1e-14)
                   for key, value in recomputed.items())
    assert f"{expected[0]['floor']:.8f}" in benchmark
    assert set(ledger["sources"]) == {
        "pra_extension/code/sharpness_benchmarks.py",
        "pra_extension/code/render_sharpness_figure.py",
        "pra_extension/code/paper_palette.py",
    }
    for relative, digest in ledger["sources"].items():
        assert _sha256(REPO_ROOT / relative) == digest, relative
    assert set(ledger["artifacts"]) == {"fig05_exact_sharpness.pdf", "fig05_exact_sharpness.png"}
    for filename, digest in ledger["artifacts"].items():
        assert _sha256(REPO_ROOT / "paper" / "figures" / "sharpness" / filename) == digest, filename


def test_combined_figure_preserves_all_original_rows_and_separates_exact_reference() -> None:
    import pandas as pd

    from sharpness_benchmarks import convergence_rows

    ledger = _json("paper/figures/long_context_sharpness/fig04_long_context_sharpness_source.json")
    old_directory = REPO_ROOT / "paper" / "figures" / "long_context_n5"
    for key, filename, row_count, sort_columns in (
        ("window_rows", "fig04_long_context_n5_source.csv", 360,
         ["root_id", "teacher_gamma", "window", "series"]),
        ("feature_rows", "fig04_long_context_n5_features_source.csv", 72,
         ["root_id", "teacher_gamma", "window", "feature_budget"]),
    ):
        archived = pd.read_csv(old_directory / filename, float_precision="round_trip")
        combined = pd.DataFrame(ledger[key])
        assert len(combined) == len(archived) == row_count
        assert set(combined.columns) == set(archived.columns)
        combined = combined[archived.columns].sort_values(sort_columns).reset_index(drop=True)
        archived = archived.sort_values(sort_columns).reset_index(drop=True)
        pd.testing.assert_frame_equal(combined, archived, check_exact=False, rtol=1e-13, atol=1e-15)
    expected = convergence_rows()
    assert len(ledger["convergence_rows"]) == len(expected) == 9
    for recorded, recomputed in zip(ledger["convergence_rows"], expected):
        assert recorded.keys() == recomputed.keys()
        assert all(math.isclose(recorded[key], value, rel_tol=1e-12, abs_tol=1e-14)
                   for key, value in recomputed.items())
    # Different metrics are intentionally not pooled or placed on one scale.
    assert set(ledger["panels"]) == {"a", "b", "c", "d"}
    assert "NRMSE" in ledger["panels"]["a"]["metric"]
    assert "NRMSE" in ledger["panels"]["b"]["metric"]
    assert "NRMSE" in ledger["panels"]["c"]["metric"]
    assert "absolute scalar error" in ledger["panels"]["d"]["metric"]
    assert {
        "pra_extension/code/render_long_context_sharpness.py",
        "pra_extension/code/sharpness_benchmarks.py",
        "pra_extension/code/render_long_context_n5.py",
        "pra_extension/code/paper_palette.py",
    } <= set(ledger["sources"])
    for relative, digest in ledger["sources"].items():
        assert _sha256(REPO_ROOT / relative) == digest, relative
    assert set(ledger["artifacts"]) == {
        "fig04_long_context_sharpness.pdf", "fig04_long_context_sharpness.png",
    }
    for filename, digest in ledger["artifacts"].items():
        assert _sha256(REPO_ROOT / "paper" / "figures" / "long_context_sharpness" / filename) == digest


def test_icaart_abstract_and_short_captions_keep_plain_explanations_and_scope() -> None:
    """Keep the time-series framing readable without dropping the claim boundaries."""
    directory = REPO_ROOT / "paper" / "icaart2027" / "sections"
    abstract = re.sub(r"\s+", " ", (directory / "abstract.tex").read_text(encoding="utf-8"))
    assert (
        "Different pasts can produce identical recent inputs but different QRC outputs, "
        "creating an unavoidable error for a window model."
    ) in abstract
    assert (
        "Under our assumptions, sufficiently expressive reset-QELMs evaluated through exact expectations "
        "permit arbitrarily accurate emulation precisely when the missing past's influence on the "
        "chosen outputs vanishes uniformly as the window grows."
    ) in abstract
    assert "consider a QELM when an affordable window contains enough history for the required accuracy" in abstract
    assert "retain the QRC when essential information must be carried beyond that window" in abstract
    assert "Sufficient history enables replacement in principle" in abstract
    assert "representation, training, and measurement determine its practical success" in abstract
    introduction = re.sub(r"\s+", " ", (directory / "introduction.tex").read_text(encoding="utf-8"))
    assert "in a time-series application?" in introduction
    assert "Our theoretical target is the QRC's chosen output" in introduction
    assert "Separate forecasting experiments" in introduction
    assert "Consider a brief event in a sensor stream." in introduction
    assert r"Under the assumptions of Section~\ref{sec:criterion}" in introduction
    assert "growing windows and sufficient QELM capacity" in introduction
    assert "emulation with exact expectations" in introduction
    # The practitioner-facing sections do not claim that quantum memory is
    # necessary among all model classes, or give a scalar equality for vectors.
    for name in ("abstract", "introduction", "conclusion"):
        prose = (directory / f"{name}.tex").read_text(encoding="utf-8")
        assert r"\textbf" not in prose and r"\bfseries" not in prose
        assert "$" not in prose
        assert not re.search(r"(?:only a QRC|quantum memory is necessary)", prose, re.I)

    captions = []
    for name in ("introduction", "dynamics", "evaluation"):
        source = (directory / f"{name}.tex").read_text(encoding="utf-8")
        source = re.sub(r"(?<!\\)%[^\n]*", "", source)
        starts = list(re.finditer(r"\\caption\{", source))
        assert len(starts) == 1, name
        start, depth = starts[0].end(), 1
        for end in range(start, len(source)):
            if source[end - 1] != "\\":
                depth += (source[end] == "{") - (source[end] == "}")
            if depth == 0:
                break
        assert depth == 0, f"{name}: unclosed caption"
        caption = re.sub(r"\s+", " ", source[start:end]).strip()
        assert 1 <= len(caption.split()) <= 150, (name, len(caption.split()))
        captions.append(caption)

    interface, resources, combined = captions
    assert "QRC carries its quantum state forward" in interface
    assert "reset-QELM receives a window of recent inputs" in interface
    assert "influence has vanished by then, so the curves stay identical" in interface
    assert "influence becomes more visible and a gap remains" in interface
    assert "a gap remains" in interface
    assert "schematic examples, not trained-model results" in interface
    assert "time advances while the window length stays fixed" in interface
    models = re.sub(r"\s+", " ", (directory / "models.tex").read_text(encoding="utf-8"))
    assert "fully forgotten by window expiry, a special case of fading memory" in models
    assert "Sufficiency here concerns these example histories, not uniform accuracy" in models

    assert "Three resources, three different questions" in resources
    assert "Blue windows meet the 10\\% old-history criterion in all three target sizes" in resources
    assert "More features help" in resources
    assert "validation-selected feature counts versus measurement budget" in resources
    assert "sampled NRMSE percentages, not worst-case bounds" in resources
    assert "12 realizations per cohort" in resources
    assert "10th--90th percentiles" in resources and "not confidence intervals" in resources
    assert "bands in b, rounded ranges in c" in resources

    assert "Memory in emulation and prediction" in combined
    assert "emulation, not laser forecasting" in combined
    assert "full-test NRMSE" in combined and "output scales differ" in combined
    assert "12 per loss rate in c, 24 in b" in combined
    assert "not confidence intervals" in combined
    assert "Synthetic NARMA10 targets and predictions use identical output scales" in combined
    assert "red unavoidable window error" in combined
    assert "not a laser or NARMA bound" in combined
    assert "analytic readout is not learned" in combined
    assert "medians and 10th--90th root percentiles" in combined
    assert "Replay refits readouts and costs extra evolution" in combined
    assert len(combined.split()) <= 130 and "$" not in combined
    # Detailed construction, cohort and normalization scope remains in the text.
    evaluation = " ".join((directory / "evaluation.tex").read_text().split())
    methods = " ".join((directory / "methodology.tex").read_text().split())
    assert "not the best readout at each copy count" in evaluation
    assert "one-qubit channel" in evaluation
    assert "five-qubit study uses a ten-qubit QELM" in evaluation
    assert "moves from emulation to actual task targets" in evaluation
    assert "Across 24 fresh roots" in evaluation
    assert "3.10\\%" in evaluation and "27.88\\%" in evaluation
    assert "59.7\\%" in evaluation and "77.3\\%" in evaluation
    assert "NRMSE uses training-target standard deviation" in methods
    assert "first root's first 128 scored rows" in methods
    assert "all 1,920 test rows" in methods
    assert "only the QELM readout is fitted" in methods
    assert "never past targets" in methods
    assert "not global $D_d/2$ or NRMSE" in methods
    assert "replicate an earlier 12-root study without pooling" in methods


def test_icaart_finite_resources_bridge_keeps_stronger_assumptions_and_pointwise_scope() -> None:
    """The short bridge summarizes the appendix bound, not a new error identity."""
    directory = REPO_ROOT / "paper" / "icaart2027"
    main = (directory / "main.tex").read_text(encoding="utf-8")
    sections = {
        name: (directory / "sections" / f"{name}.tex").read_text(encoding="utf-8")
        for name in ("background", "dynamics", "appendix")
    }
    assert main.index(r"\input{sections/dynamics}") < main.index(r"\input{sections/evaluation}")
    assert re.findall(r"\\subsection\{([^}]+)\}", sections["dynamics"]) == [
        "Observable Forgetting", "Required Window Length", "Protected Memory", "Finite Resources",
    ]
    assert sections["dynamics"].count(r"\label{sec:finite-resources}") == 1
    bridge = re.sub(r"\s+", " ", sections["dynamics"].split(r"\label{sec:finite-resources}", 1)[1])
    assert "For inputs in $[0,1]$ and a scalar output" in bridge
    assert "midpoint construction assumes" in bridge
    assert "stronger than observable forgetting" in bridge
    assert "uniform state contraction and Lipschitz dependence on inputs" in bridge
    assert r"\eqref{eq:incremental-lipschitz}" in bridge
    assert "$r$ encoding copies per input" in bridge
    assert "$Q$ independent samples of its bounded readout" in bridge
    assert r"\eqref{eq:midpoint-resource-bound}" in bridge
    assert r"\widehat f^{\rm mid}" in bridge
    compact_math = re.sub(r"\s+", "", bridge)
    for term in (
        r"\underbrace{D_d/2}_{\text{missinghistory}}",
        r"\underbrace{\varepsilon_{\rmrep}}_{\text{representation}}",
        r"\underbrace{\varepsilon_{\rmshot}}_{\text{measurements}}",
    ):
        assert term in compact_math
    assert r"\leq" in bridge
    assert "For each fixed history" in bridge
    assert "probability at least $1-\\zeta$" in bridge
    assert "failure tolerance $0<\\zeta<1$" in bridge
    assert "not simultaneously over all histories" in bridge
    assert "$1/\\sqrt r$ and $1/\\sqrt Q$, respectively" in bridge
    assert "at fixed window and confidence" in bridge
    assert "$rd$ qubits" in bridge and "$(r+1)^d$" in bridge
    assert "reachable-output extrema" in bridge
    assert "computation is not included in those resource counts" in bridge
    assert "constructive upper bound" in bridge
    assert "not an efficient learning method or an exact decomposition of fitted-model error" in bridge

    # Keep the numbered full bound in Appendix C; the bridge is deliberately
    # unnumbered so equations (12)--(26) and all existing references stay fixed.
    assert not re.search(r"\\begin\{(?:equation|align|gather)\}", bridge)
    assert not re.search(r"\\(?:setcounter|addtocounter)\{equation\}|\\tag\{", bridge)
    assert sections["appendix"].count(r"\label{eq:midpoint-resource-bound}") == 1
    assert r"\label{eq:midpoint-resource-bound}" not in sections["dynamics"]
    background = re.sub(r"\s+", " ", sections["background"])
    assert r"Section~\ref{sec:finite-resources} summarizes its resource bounds" in background
    assert r"Appendix~\ref{app:bernstein-resources} gives the construction" in background
    assert re.search(
        r"does not (?:guarantee efficient training or )?make the fixed six-qubit emulator .*? universal",
        background,
    )


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
        r"\label{sec:introduction}"
    ) < conference_introduction.index(
        "fig01_information_interface.pdf"
    ) < conference_introduction.index("A QELM transforms each window")
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
