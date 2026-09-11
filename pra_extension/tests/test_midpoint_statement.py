"""Regression checks for the scalar midpoint construction, not study data.

The analytic toy below checks elementary minimax and Bernstein identities;
it is not a numerical QRC experiment or evidence of a trainable emulator.
"""

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_VERSIONS = ("canonical_emulation", "icaart2027")


def _section(version: str, name: str) -> str:
    path = REPO_ROOT / "paper" / version / "sections" / f"{name}.tex"
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def _equation(text: str, label: str) -> str:
    equations = re.findall(
        r"\\begin\{equation\}(.*?)\\end\{equation\}", text, re.DOTALL
    )
    matches = [item for item in equations if rf"\label{{{label}}}" in item]
    assert len(matches) == 1, f"Expected exactly one equation labelled {label}"
    return re.sub(r"\s+", "", matches[0])


@pytest.mark.parametrize("version", PAPER_VERSIONS)
def test_scalar_qelm_theorem_states_fixed_window_information_optimum(version: str) -> None:
    background = _section(version, "background")
    equation = _equation(background, "eq:qrc-qelm-equivalence")

    assert r"\boxed" in equation
    assert r"\inf_{f\in\mathcal Q_d}".replace(" ", "") in equation
    assert r"\sup_h" in equation
    assert "y(h)-f(S_d(h))" in equation
    assert r"\frac{D_d}{2}" in equation
    assert r"D_d\longrightarrow0" not in equation
    assert r"\label{eq:window-law}" in background
    assert r"D_d\longrightarrow0" in background


@pytest.mark.parametrize("version", PAPER_VERSIONS)
def test_section_three_preserves_equations_citations_and_automatic_appendix_references(
    version: str,
) -> None:
    background = _section(version, "background")
    appendix = _section(version, "appendix")
    assert background.count(r"\label{sec:criterion}") == 1
    equation_labels = re.findall(r"\\label\{(eq:[^}]+)\}", background)
    assert equation_labels == [
        "eq:diameter", "eq:window-law", "eq:qrc-qelm-equivalence",
    ]
    diameter = _equation(background, "eq:diameter")
    window_law = _equation(background, "eq:window-law")
    assert r"D_d=\sup_{S_d(h)=S_d(h')}\|y(h)-y(h')\|" in diameter
    assert r"\inf_f\sup_h|y(h)-f(S_d(h))|=\frac{D_d}{2}" in window_law
    assert r"\cite{goto2021universal,yu2024nonasymptotic}" in background
    bibliography = (REPO_ROOT / "paper" / version / "references.bib").read_text()
    assert "@article{goto2021universal," in bibliography
    assert "@inproceedings{yu2024nonasymptotic," in bibliography
    for label in ("app:window-laws", "app:bernstein-resources"):
        assert appendix.count(rf"\label{{{label}}}") == 1
    # Journal references include the containing letter and subsection number;
    # conference appendix subsections carry their own letters. Do not hardcode
    # either artifact's generated appendix identifiers into Section 3.
    assert r"Appendix~\ref{app:bernstein-resources}" in background
    if version == "canonical_emulation":
        assert r"Appendix~\ref{app:proofs}" in background
    else:
        assert r"Appendix~\ref{app:window-laws}" in background
    assert r"Section~\ref{sec:numerics}" in background


@pytest.mark.parametrize("version", PAPER_VERSIONS)
def test_fixed_window_statement_retains_scope_and_nonattainment(version: str) -> None:
    background = _section(version, "background")
    appendix = _section(version, "appendix")
    combined = (background + " " + appendix).lower()

    for requirement in ("scalar", "compact", "continuous", "dense", "exact"):
        assert requirement in background.lower()
    assert "growing" in background.lower()
    assert "concaten" in combined or "appendable" in combined
    assert "infimum" in background.lower()
    assert re.search(r"(?:need not|not necessarily).{0,100}attain", combined)
    assert "finite vector" in combined
    assert "replay" in appendix.lower()
    assert re.search(
        r"(?:without a guarantee of efficient fitting|"
        r"efficient fitting remains a separate question)", appendix.lower(),
    )
    assert "every finite input suffix can follow every admissible prehistory" in background
    assert "For each fixed window length $d$" in background
    assert "continuous window functions under the uniform norm" in background
    assert "Features are evaluated through exact expectations" in background
    assert "single finite QELM need not attain it exactly" in background
    assert "does not guarantee efficient training" in background
    assert "fixed six-qubit emulator" in background
    assert "fixed-window optimum lies between $D_d/2$ and $D_d$" in background
    assert "randomization cannot lower the scalar bound on worst-case expected absolute error" in background


@pytest.mark.parametrize("version", PAPER_VERSIONS)
def test_midpoint_proof_uses_common_compact_reachable_state_domain(version: str) -> None:
    appendix = _section(version, "appendix")
    compact_math = appendix.replace(" ", "")

    assert "reachable" in appendix.lower()
    assert "compact" in appendix.lower()
    assert r"\mathcalR" in compact_math
    assert "m_d" in appendix
    assert "y_-(s)" in appendix
    assert "y_+(s)" in appendix
    assert r"\min" in appendix
    assert r"\max" in appendix
    assert "uniform" in appendix.lower()
    assert "continuous" in appendix.lower()
    assert r"\frac{D_d}{2}" in compact_math


@pytest.mark.parametrize("version", PAPER_VERSIONS)
def test_observable_criterion_explicitly_attributes_existing_stability_result(
    version: str,
) -> None:
    dynamics = _section(version, "dynamics")
    # The revised section attributes the criterion immediately after its
    # statement and before discussing the required window length.
    attribution = dynamics.split(r"\subsection{Required Window Length}")[0]
    assert r"\label{eq:visible-memory-theorem}" in attribution

    assert "Observable-memory criterion." in attribution
    assert "Observable-memory theorem." not in dynamics
    assert "petreczky" in attribution.lower()
    assert "III.1" in attribution
    assert re.search(r"\\cite[^{}]*\{[^}]*petreczky[^}]*\}", attribution)
    assert re.search(r"\\cite[^{}]*\{[^}]*grigoryeva[^}]*2021[^}]*\}", attribution)


@pytest.mark.parametrize("version", PAPER_VERSIONS)
def test_midpoint_resource_bound_is_separate_from_preserved_replay_bound(version: str) -> None:
    background = _section(version, "background")
    appendix = _section(version, "appendix")
    replay = _equation(appendix, "eq:qrc-qelm-bound")
    midpoint = _equation(appendix, "eq:midpoint-resource-bound")
    # Definitions can precede the replay certificate in an unnumbered display.
    # They must still appear before the separate midpoint construction.
    replay_construction = appendix.split(r"\label{eq:qrc-qelm-bound}")[0]
    replay_definitions = re.sub(r"\s+", "", replay_construction).replace("&", "")

    # Bernstein grid coefficients require the whole cube, not just an arbitrary
    # compact subset of the scalar input interval.
    assert "full scalar input alphabet $[0,1]$" in background
    assert re.search(
        r"(?:full scalar input alphabet|Take inputs in) \$\[0,1\]\$", appendix,
    )
    assert r"\in[0,1]^d" in appendix.replace(" ", "")

    # The existing replay certificate and its validation are not retrospectively
    # relabelled as a midpoint implementation.
    assert r"\varepsilon_{\rmhist}=2M\eta^d" in replay_definitions
    assert (r"\varepsilon_{\rmrep}=\frac{ML_{\rmin}(1-\eta^d)}"
            r"{(1-\eta)\sqrtr}") in replay_definitions
    assert (r"\varepsilon_{\rmshot}=M\sqrt{\frac{2\log(2/\zeta)}{Q}}"
            in replay_definitions)
    assert r"\varepsilon_{\rmhist}+\varepsilon_{\rmrep}" in replay
    assert r"\frac{D_d}{2}" not in replay

    assert r"\frac{D_d}{2}" in midpoint
    assert r"\varepsilon_{\rmrep}" in midpoint
    assert r"\varepsilon_{\rmshot}" in midpoint
    if version == "canonical_emulation":
        # The archived edition retains the probability inside its display.
        assert r"1-\zeta" in midpoint
    else:
        # The compact ICAART displays retain both inequalities; their
        # probability qualifiers are immediately following prose, not a
        # claim of simultaneous high-probability control over all histories.
        assert r"\sup_h|y(h)-f^{\rmmid}(S_d(h))|" in midpoint
        assert r"|y(h)-\widehatf^{\rmmid}(S_d(h))|" in midpoint
        for label, expected_scope in (
            (
                "eq:qrc-qelm-bound",
                "The first holds uniformly; the second holds with probability at least "
                "$1-\\zeta$ for each fixed history $h$, not all histories simultaneously.",
            ),
            (
                "eq:midpoint-resource-bound",
                "The second bound again holds with probability at least $1-\\zeta$ "
                "for each fixed history, using independent midpoint readouts.",
            ),
        ):
            after_label = appendix.split(rf"\label{{{label}}}", 1)[1]
            after_equation = after_label.split(r"\end{equation}", 1)[1].lstrip()
            assert after_equation.startswith(expected_scope)
    assert "m_d" in appendix
    assert "coefficient" in appendix.lower()
    assert "extrema" in appendix.lower()
    assert re.search(r"fixed.{0,35}(?:history|prediction|input)", appendix.lower())
    assert "independent" in appendix.lower() or "iid" in appendix.lower()
    assert r"$8.33\times10^{-17}$" in appendix
    assert r"every tested $(d,r)$ pair satisfies" in appendix.lower()


def _midpoint(x: np.ndarray, eta: float, magnitude: float) -> float:
    """Exact suffix midpoint of a bounded scalar affine-state toy."""

    weights = eta ** np.arange(len(x) - 1, -1, -1)
    return float(magnitude * (1.0 - eta) * (weights @ (2.0 * x**2 - 1.0)))


@pytest.mark.parametrize("d,eta", [(1, 0.0), (1, 0.7), (2, 0.4), (3, 0.8)])
def test_analytic_scalar_midpoint_has_half_diameter_minimax_error(
    d: int, eta: float,
) -> None:
    magnitude = 1.7
    x = np.linspace(0.15, 0.85, d)
    midpoint = _midpoint(x, eta, magnitude)
    radius = magnitude * eta**d
    endpoints = np.array([midpoint - radius, midpoint + radius])
    predictions = midpoint + np.array([-0.37, -0.1, 0.0, 0.22])
    worst_errors = np.abs(predictions[:, None] - endpoints).max(axis=1)

    np.testing.assert_allclose(worst_errors, radius + np.abs(predictions - midpoint))
    assert worst_errors[2] == pytest.approx(np.ptp(endpoints) / 2.0)
    assert worst_errors[2] == pytest.approx(worst_errors.min())
    assert np.abs(endpoints).max() <= magnitude + 1e-14


@pytest.mark.parametrize("d,r,eta", [(1, 1, 0.0), (2, 3, 0.4), (3, 6, 0.8)])
def test_midpoint_bernstein_coefficients_preserve_range_and_lipschitz_error_bound(
    d: int, r: int, eta: float,
) -> None:
    magnitude = 1.7
    x = np.linspace(0.15, 0.85, d)
    y = np.linspace(0.75, 0.25, d)
    grid = np.array(list(itertools.product(range(r + 1), repeat=d)))
    coefficients = np.array([_midpoint(k / r, eta, magnitude) for k in grid])
    probabilities = np.array([
        math.prod(
            math.comb(r, int(k_i)) * x_i**k_i * (1.0 - x_i) ** (r - k_i)
            for k_i, x_i in zip(k, x, strict=True)
        )
        for k in grid
    ])
    weights = eta ** np.arange(d - 1, -1, -1)
    exact_midpoint = _midpoint(x, eta, magnitude)
    bernstein = float(probabilities @ coefficients)
    exact_bias = float(
        2.0 * magnitude * (1.0 - eta) * (weights @ (x * (1.0 - x))) / r
    )
    # For this toy, the input Lipschitz constant in the paper's convention
    # is L_in = 2(1-eta); both endpoint functions and their midpoint share it.
    input_lipschitz = 2.0 * (1.0 - eta)
    weighted_lipschitz_bound = (
        2.0 * magnitude * input_lipschitz * (weights @ np.abs(x - y))
    )
    representation_bound = (
        magnitude * input_lipschitz * (1.0 - eta**d)
        / ((1.0 - eta) * math.sqrt(r))
    )

    assert np.abs(coefficients).max() <= magnitude + 1e-14
    assert probabilities.sum() == pytest.approx(1.0)
    assert bernstein - exact_midpoint == pytest.approx(exact_bias)
    assert abs(bernstein - exact_midpoint) <= representation_bound + 1e-14
    assert abs(exact_midpoint - _midpoint(y, eta, magnitude)) <= weighted_lipschitz_bound
