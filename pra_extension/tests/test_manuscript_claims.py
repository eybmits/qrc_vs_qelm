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
        assert "history passes, the fitted QELM fails, and replay succeeds" in sections["methodology"]
        if directory.name == "icaart2027":
            # The revised E.1 states the mixed origin of the empirical residual
            # directly; the operational diagnosis is not a feature-family bound.
            for component in ("feature", "fitting", "regularization", "estimation"):
                assert component in diagnostic
            assert "not a pure representation error" in diagnostic
            assert r"\emph{representation-limited} diagnosis" in diagnostic
            assert "does not establish a lower bound for all readouts using those features" in diagnostic
            assert "four conditionally independent prefixes" in diagnostic
            assert "$4/3$" in diagnostic and "uncorrected terms" in diagnostic
            assert "Neither quantity estimates $D_d$ or $r_{\\rm vis}$" in diagnostic
        else:
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
            assert re.search(
                r"(?:does not (?:isolate|identify)(?: the)?.*?"
                r"best approximation (?:attainable with the|using|available from the) QELM features|"
                r"not that the best approximation using QELM features fails)",
                sections["methodology"],
            )
        assert "common-suffix residual emulator NRMSE" in sections["evaluation"]
        if directory.name == "icaart2027":
            # The revised framing retains the practical resource distinctions
            # without prescribing a particular editorial sentence.
            for resource in ("history", "representation", "learning", "measurements"):
                assert resource in sections["conclusion"].lower()
            assert re.search(
                r"representability, not efficient (?:learning|training) or resource savings",
                sections["conclusion"],
            )
            assert re.search(r"(?:without establishing a quantum advantage|not (?:evidence of )?quantum advantage)",
                             sections["evaluation"])
            for resource in ("context", "features", "measurements"):
                assert resource in sections["abstract"]
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
    # Line breaks and alignment may differ between the journal and conference
    # layouts; compare every mathematical token after removing spacing only.
    equation_tokens = [re.sub(r"\\\\|&|\\(?:quad|qquad)\b|\s", "", equation)
                       for equation in equations]
    assert equation_tokens[0] == equation_tokens[1]


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
        r"An Explicit QELM Construction",
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
    # Models, train-only preprocessing, validation selection, and operational
    # decision rules remain in E.2 after the complete user-supplied rewrite.
    fitting = re.sub(r"\s+", " ", method_parts[2])
    for condition in ("QRC targets", "Reset-QELM", "seed sets are disjoint",
                      "Training data alone standardize features",
                      "Validation selects", "refitted on training and validation data",
                      "same windows and splits", "10 of 12 roots"):
        assert condition in fitting
    assert method_parts[3].startswith("Finite Measurements}")
    assert method_parts[4].startswith("Longer Windows}")
    assert method_parts[5].startswith("Time-Series Demonstrations}")
    assert r"\label{app:task-forecasting}" in method_parts[5]
    assert r"\label{app:laser-demonstration}" in method_parts[5]
    parts = re.split(r"\\subsection\{", appendix)[1:]
    window, memory, construction, protection = [re.sub(r"\s+", " ", part) for part in parts]
    for condition in ("scalar output", "reachable closure", "Uniform QELM density",
                      "$D_d/2+\\varepsilon$", "infimum", "finite model need not attain",
                      "For vector outputs", "lower bound $D_d/2$", "error at most $D_d$",
                      "same prediction distribution", "Randomized predictions cannot lower"):
        assert condition in window
    memory_math = re.sub(r"\s+", "", memory)
    for identity in (r"W=V/\mathcalN", r"\|x\|_{\rmobs}=\sup_w\|\overline{\mathcalC}A_wx\|",
                     r"\kappa_-c_d\leqD_d\leq\kappa_+c_d", r"c_{d+\ell}\leqc_dc_\ell",
                     r"\lim_{d\to\infty}D_d^{1/d}=r_{\rmvis}",
                     r"\frac{d_\varepsilon}{\log(1/\varepsilon)}=\frac{1}{-\logr_{\rmvis}}"):
        assert identity in memory_math
    for condition in ("empty word", "norm ball", "independent of $d$",
                      "freely concatenable inputs", "constrained input grammars"):
        assert condition in memory
    assert "petreczky2026stability" in memory
    assert r"\widehat f_r" in construction
    for condition in (r"\tfrac12\sin^2(\theta/2)", r"$D_d\geq\delta$",
                      "at least $\\delta/2$ for every finite $d$",
                      "arbitrarily many shared inputs", "not for every possible input sequence"):
        assert condition in protection
    dark = _json("pra_extension/outputs/dark_sector_validation/dark_sector_summary.json")
    assert f"{dark['expected_prefix_gap']:.4f}" in protection
    assert f"{dark['expected_worst_case_absolute_floor']:.4f}" in protection
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


def test_icaart_appendix_rewrite_preserves_referenced_equation_sequence_12_to_15() -> None:
    directory = REPO_ROOT / "paper" / "icaart2027" / "sections"
    combined = "\n".join(
        (directory / f"{name}.tex").read_text(encoding="utf-8")
        for name in ("appendix", "methodology")
    )
    equations = re.findall(
        r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", combined, re.DOTALL,
    )
    labels = (
        "eq:incremental-lipschitz", "eq:midpoint-resource-bound",
        "eq:panel-decomposition", "eq:target-qrc-model",
    )
    assert len(equations) == len(labels) == 4
    for number, ((environment, body), label) in enumerate(zip(equations, labels), start=12):
        assert rf"\label{{{label}}}" in body, f"Equation ({number}): {label} moved"
        if environment == "align":
            # The target Hamiltonian and evolution share one equation number.
            assert number == 15
            assert body.count(r"\nonumber") == 1
    assert r"\eta D_{\rm tr}(\rho,\sigma)+L_{\rm in}|a-a'|" in equations[0][1]
    assert r"\widehat f_r(S_d(h))" in equations[1][1]
    assert r"D_d/2+\varepsilon_{\rm rep}+\varepsilon_{\rm shot}" in equations[1][1]
    assert not re.search(r"\\(?:setcounter|addtocounter)\{equation\}|\\tag\{", combined)


def test_conclusion_keeps_a_plain_story_and_scoped_claims() -> None:
    for directory in (PAPER_ROOT, REPO_ROOT / "paper" / "icaart2027"):
        source = (directory / "sections" / "conclusion.tex").read_text(encoding="utf-8")
        body = re.sub(r"%[^\n]*", "", source)
        body = re.sub(r"\\section\{[^{}]+\}|\\label\{[^{}]+\}", "", body).strip()
        paragraphs = [re.sub(r"\s+", " ", paragraph).strip()
                      for paragraph in re.split(r"\n\s*\n", body) if paragraph.strip()]
        assert len(paragraphs) == 4
        assert r"\textbf" not in body and r"\bfseries" not in body
        assert r"\paragraph" not in body and r"\subsection" not in body
        compact = re.sub(r"\s+", " ", body)
        if directory == PAPER_ROOT:
            assert r"Section~\ref{sec:criterion}" in body
            assert "neither efficient learning nor resource savings" in compact
            assert "no quantum advantage" in compact
        else:
            # Guard the information criterion in the revised plain-language
            # conclusion; numerical task limits remain beside those results.
            for qualifier in ("QRC outputs", "Under our assumptions", "precisely when",
                              "as their windows grow", "sufficiently expressive",
                              "exact expectations", "fades uniformly"):
                assert qualifier in paragraphs[0]
            for limit in ("two-qubit construction", "arbitrarily long delays",
                          "Every fixed-window predictor", "positive worst-case error floor",
                          "neither additional features nor a longer finite window can remove",
                          "rather than between quantum and classical computation",
                          "representability, not efficient learning or resource savings"):
                assert limit in paragraphs[1]
            evaluation = re.sub(r"\s+", " ", (directory / "sections" / "evaluation.tex")
                                .read_text(encoding="utf-8"))
            for limit in ("not laser forecasting", "Synthetic NARMA10 prediction",
                          "changing features and readouts prevent causal isolation",
                          "neither quantum necessity nor failure of every finite window follows"):
                assert limit in evaluation
            assert "Quadratic delay models outperform" not in compact
            for resource in ("missing history", "insufficient features", "finite measurements",
                             "Missing information requires longer context or retained state"):
                assert resource in paragraphs[2]
            assert "$" not in body
            assert len(compact.split()) <= 250
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
    # The active measured-input experiment adds its data source; the revised
    # Appendix B omits the former constrained-switching digression/citation.
    assert citation_sets[1] == (
        citation_sets[0] - {"philippe2016constrained"}
    ) | {"weigend1993timeseries"}


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
    from sharpness_benchmarks import convergence_rows

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
    assert r"Figure~\ref{fig:long-context}(b)" in benchmark
    assert "$0.281322$ at one copy" in benchmark
    assert "$0.084753$ at 256 copies" in benchmark
    assert "Analytical benchmarks and simulations examine three distinct limitations" in sections["abstract"]
    assert "positive worst-case error floor, regardless of its model capacity" in sections["abstract"]
    assert "Analytical benchmarks and simulations" in sections["introduction"]

    # The revised C keeps the compatible-range and factorial-moment argument,
    # rather than requiring the removed auxiliary finite-history calibration.
    assert "reachable closure does not change this supremum" in appendix
    assert "compatible populations have width $D_d=\\eta^d$" in appendix
    assert r"\eta^d/2+\|f-m_d\|_\infty" in appendix
    assert r"K_i(K_i-1)/[r(r-1)]" in appendix and r"r\geq2" in appendix
    assert "unbiased readout" in appendix and "attaining the information floor exactly" in appendix
    assert "midpoint coefficients give a nonnegative Bernstein bias" in appendix
    assert "one-dimensional maximization of that bias" in appendix
    assert "analytical readouts and binomial expectations, not training" in appendix
    assert "Neither quantity estimates $D_d$ or $r_{\\rm vis}$" in methods
    assert "rare histories can be missed" in methods

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
    """Keep the finite-context framing readable without dropping claim boundaries."""
    directory = REPO_ROOT / "paper" / "icaart2027" / "sections"
    abstract = re.sub(r"\s+", " ", (directory / "abstract.tex").read_text(encoding="utf-8"))
    for condition in ("Observable memory determines when",
                      "sufficiently expressive QELMs with growing input windows",
                      "evaluated through exact expectations",
                      "can approximate QRC outputs arbitrarily well",
                      "uniformly fading influence of the missing past",
                      "identical recent inputs", "different outputs",
                      "information limit that additional features cannot remove",
                      "Observable forgetting then determines how much context is needed",
                      "we construct a QRC",
                      "an old input remains relevant to later outputs over arbitrarily long delays",
                      "For this reservoir, every fixed-window QELM faces a positive worst-case error floor",
                      "regardless of its model capacity",
                      "missing history, insufficient features, and finite measurements",
                      "when longer windows can replace reservoir memory",
                      "persistent observable memory prevents arbitrarily accurate finite-window emulation"):
        assert condition in abstract
    for omitted in ("For scalar outputs", "half the largest", "two-qubit",
                    "under stated assumptions", "classical"):
        assert omitted not in abstract
    introduction = re.sub(r"\s+", " ", (directory / "introduction.tex").read_text(encoding="utf-8"))
    for condition in ("sensor event", "identical recent inputs", "different outputs",
                      "Under our stated assumptions", "sufficiently expressive QELMs",
                      "exact expectations", "growing windows", "vanishes uniformly",
                      "For scalar outputs", "half the largest output difference"):
        assert condition in introduction
    # Scope is explicit in the sections defining outputs and reporting tasks,
    # without requiring the introduction to repeat those qualifications.
    models = re.sub(r"\s+", " ", (directory / "models.tex").read_text(encoding="utf-8"))
    assert "emulation simulations use observable expectations directly" in models
    assert "forecasting tests fit both readouts to external task targets" in models
    evaluation = re.sub(r"\s+", " ", (directory / "evaluation.tex").read_text(encoding="utf-8"))
    assert "practical emulation limits, not quantum advantage" in evaluation
    # The task implication uses exact QRC means and a stateless comparison over
    # unbounded delays; it does not apply when the window covers the horizon.
    dynamics = re.sub(r"\s+", " ", (directory / "dynamics.tex").read_text(encoding="utf-8"))
    memory_task = dynamics.split(r"\emph{Memory-task implication.}", 1)[1].split(
        "Matched controls", 1,
    )[0]
    for condition in ("Exact QRC expectations", "stateless fixed-window predictor",
                      r"worst-case error at least $\delta/2$", "unbounded delays",
                      "not when the window covers the task horizon"):
        assert condition in memory_task
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
    assert "Test error falls with more shots" in resources
    assert "diamonds use validation-selected feature counts" in resources
    assert "Points are offset within shot tiers" in resources
    assert "horizontal dashed line marks 10\\% error" in resources
    assert "sampled NRMSE percentages, not worst-case bounds" in resources
    assert "12 realizations per cohort" in resources
    assert "10th--90th percentiles" in resources and "not confidence intervals" in resources
    assert "bands in b, whiskers in c" in resources

    assert "Memory in emulation and prediction" in combined
    assert "QELM emulation of laser-driven QRC" in combined
    assert "not laser forecasting" in combined
    assert "full-test NRMSE" in combined and "differing output scales" in combined
    assert "12 roots/loss in b; 24 in c" in combined
    assert "not confidence intervals" in combined
    assert "Synthetic NARMA10 traces share output scales" in combined
    assert "red analytical window-error floor" in combined
    assert "not a laser/NARMA bound" in combined
    assert "without readout training" in combined
    assert "Medians and 10th--90th root percentiles" in combined
    assert "Replay resets the same QRC" in combined
    assert "on its last $d$ inputs and refits its readout" in combined
    assert r"at every $d=1,\ldots,48$" in combined
    assert "Red dashes: empirical full-history QRC reference, not a floor" in combined
    assert "costing extra evolution" in combined
    assert len(combined.split()) <= 150
    assert re.findall(r"\$[^$]+\$", combined) == ["$d$", r"$d=1,\ldots,48$"]
    # Detailed construction, cohort and normalization scope remains in the text.
    evaluation = " ".join((directory / "evaluation.tex").read_text().split())
    methods = " ".join((directory / "methodology.tex").read_text().split())
    assert "not the best readout at each copy count" in evaluation
    assert "one-qubit channel" in evaluation
    assert "five-qubit study uses a ten-qubit QELM" in evaluation
    assert "Synthetic NARMA10 prediction" in evaluation
    assert "across 24 fresh roots" in evaluation
    assert "3.10\\%" in evaluation and "27.88\\%" in evaluation
    assert "59.7\\%" in evaluation and "77.3\\%" in evaluation
    assert "NRMSE uses the training-target standard deviation" in methods
    assert "4096/1024/2048-sample training/validation/test blocks" in methods
    assert "training-only standardization" in methods
    assert "QELM predicts the QRC's final $Z_0$ expectation, not the next laser value" in methods
    assert "never past targets" in methods
    assert "half the output gap bounds one of the two absolute prediction errors, not global NRMSE" in methods
    assert "Twenty-four fresh roots" in methods
    assert "at least 20 of 24 positive roots" in methods
    assert "without simultaneous coverage" in methods


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
    assert r"\widehat f_r" in bridge
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

    # Keep the numbered bound in Appendix C and its short main-text bridge
    # unnumbered; the revised appendix retains four referenced equations.
    assert not re.search(r"\\begin\{(?:equation|align|gather)\}", bridge)
    assert not re.search(r"\\(?:setcounter|addtocounter)\{equation\}|\\tag\{", bridge)
    assert sections["appendix"].count(r"\label{eq:midpoint-resource-bound}") == 1
    assert r"\label{eq:midpoint-resource-bound}" not in sections["dynamics"]
    construction = re.sub(r"\s+", " ", sections["appendix"].split(
        r"\label{app:bernstein-resources}", 1,
    )[1].split(r"\subsection{", 1)[0])
    for condition in ("For each fixed history", "probability at least $1-\\zeta$",
                      "exact expectations", "holds uniformly over histories",
                      "finite-shot guarantee is pointwise, not simultaneous over all histories",
                      "independent readouts", "each readout lies in $[-M,M]$",
                      "$rd$ qubits", "$(r+1)^d$ midpoint coefficients",
                      "reachable-output extrema", "not included in the resource count",
                      "representability, not an efficient training procedure"):
        assert condition in construction
    construction_math = re.sub(r"\s+", "", construction)
    for term in (r"f_r(s)=\mathbbE\bigl[m_d(K_1/r,\ldots,K_d/r)\bigr]",
                 r"\frac{ML_{\rmin}(1-\eta^d)}{(1-\eta)\sqrtr}",
                 r"M\sqrt{\frac{2\log(2/\zeta)}{Q}}"):
        assert term in construction_math
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
    # Declare the overview immediately after the opening model distinction,
    # before the motivation and related-work paragraphs, to retain its early float.
    opening, after_overview = conference_introduction.split(r"\end{figure*}", 1)
    assert opening.count(r"\begin{figure*}") == 1
    assert opening.index("which reset for each prediction") < opening.index(
        "fig01_information_interface.pdf"
    )
    assert after_overview.lstrip().startswith("This distinction matters")
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
