"""Keep symbol meanings aligned between the two manuscript versions."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_VERSIONS = ("canonical_emulation", "icaart2027")


def _section(version: str, name: str) -> str:
    source = REPO_ROOT / "paper" / version / "sections" / f"{name}.tex"
    return re.sub(r"\s+", " ", source.read_text(encoding="utf-8"))


@pytest.mark.parametrize("version", PAPER_VERSIONS)
def test_models_define_outputs_features_and_prediction_before_window_law(version: str) -> None:
    models = _section(version, "models")
    background = _section(version, "background")
    compact = re.sub(r"\s+", "", models)

    # An input-history function and its value at the current step are the same
    # target; the measured feature count is independent of output dimension.
    assert r"$y(h)=y_t$" in models
    assert r"y_t=\mathcalC(\rho_t)" in compact
    assert r"S_d(h)=(a_{t-d+1},\ldots,a_t)" in compact
    if version == "icaart2027":
        # The accessible model definition needs only the state/output/window
        # maps. Feature-count and encoding notation belong to the methods.
        for meaning in ("specified initial state", "fixed map", "observable expectations",
                        "weights fixed during emulation", "input steps, not qubits",
                        "No additional recurrent quantum or classical state",
                        "identical prediction distributions"):
            assert meaning in models
        assert "finite vector of observable expectations" in models
        caption_source = _section(version, "introduction")
        for symbol_definition in ("$E$ encodes the window", "$O_j$ and $P_j$",
                                  "$m$ and $B$ count QRC outputs and QELM features",
                                  "$k$ indexes inputs while $t$ marks the current step"):
            assert symbol_definition in caption_source
        assert "$B$" not in models and "$P_j$" not in models
    else:
        assert r"$j=1,\ldots,m$" in models
        assert r"$P_j$, $j=1,\ldots,B$" in models
        assert "measurement checkpoints" in models
        assert "$B$ counts measured expectations, not qubits" in models
        assert r"$E(S_d(h))$" in models
    assert r"\widehaty_t=f(S_d(h))" in compact
    # Introductory definitions are displayed without shifting the theorem's
    # established equation numbering.
    assert models.count(r"\[") == models.count(r"\]")
    if version == "icaart2027":
        # The short window definition can be inline to avoid a layout orphan.
        assert 1 <= models.count(r"\[") <= 2
    else:
        assert models.count(r"\[") == 3
    assert r"\begin{equation}" not in models
    assert r"\textbf{" not in models
    assert re.search(r"(?:Identical windows therefore produce identical predictions for a fixed model|"
                     r"window-only model must give both histories the same prediction)", models)
    assert r"\cite{fujii2017harnessing,mujal2021opportunities}" in models
    assert r"\cite{innocenti2023potential,kawanabe2026tdqelm}" in models
    assert re.search(r"fixed norm|norm is fixed", background)
    assert re.search(r"absolute value for (?:a scalar output|scalar outputs)", background)

    # Q remains a shot budget; it cannot also name a predictor in the theorem.
    for legacy in ("Y(h)", "Y_t", r"\widehat Y", r"\inf_{g_d}", r"\inf_{Q"):
        assert legacy not in models + background
    assert r"\inf_{f\in\mathcal Q_d}" in background


@pytest.mark.parametrize("version", PAPER_VERSIONS)
def test_general_inputs_and_physical_time_have_distinct_meanings(version: str) -> None:
    models = _section(version, "models")
    dynamics = _section(version, "dynamics")
    appendix = _section(version, "appendix")
    methods = _section(version, "methodology")
    protected = appendix if version == "icaart2027" else methods

    if version == "icaart2027":
        assert "history $h$ ending at step $t$" in models
    else:
        assert "discrete step $t$" in models
        assert "$k$ indexes displayed steps and $t$ marks the current one" in models
    for theory in (dynamics, appendix):
        assert r"\Phi_a" in theory
        assert r"A_a" in theory
        for obsolete in (r"\Phi_u", r"\Phi_{u", "A_u", "A_{u"):
            assert obsolete not in theory
    # The protected-memory model still reserves u for its physical drive.
    if version == "icaart2027":
        assert "Each input contains a drive value $u\\in[0,1]$ and a binary flag $q$" in dynamics
        assert "input rule is unchanged over time" in dynamics
    else:
        assert "$a=(u,q)$" in dynamics
    assert re.search(r"H\(u\)\s*&?=", protected)
    assert r"H_T(a)" in methods
    assert r"\sigma_i^-=|0\rangle\langle1|_i" in protected
    if version == "icaart2027":
        # The revised protocol uses conventional dot notation without adding a
        # second time variable; discrete sample t remains defined in Models.
        assert r"\dot\rho&=-i[H_T(a),\rho]" in methods
        assert "each step lasts 0.5" in methods
        assert r"d\rho/dt" not in methods and r"d\rho/d t" not in methods
        assert re.search(r"(?:lowering operator|lowers qubit \$i\$)", protected)
    else:
        assert r"\dot\rho=d\rho/d\tau" in methods
        assert r"physical time $\tau$" in methods
        assert re.search(r"\$t\$ indexes discrete (?:input|step)", methods)
        assert "lowering operator on qubit $i$" in protected


@pytest.mark.parametrize("version", PAPER_VERSIONS)
def test_appendix_separates_reachable_geometry_and_sampled_predictors(version: str) -> None:
    appendix = _section(version, "appendix")
    compact = appendix.replace(" ", "")

    if version == "icaart2027":
        # The quotient and norm equivalence are stated in words; removed aliases
        # are not scientific prerequisites, but their definitions remain explicit.
        for definition in ("real span", "quotient space $W$",
                           "under every finite future input sequence",
                           "including the empty sequence", "induced updates $A_a$",
                           "fixed positive factors independent of $d$"):
            assert definition in appendix
        assert re.search(r"reuse \$\\mathcal C\$.*?output map.*?\$\\Delta\$.*?image.*?\$W\$", appendix)
        midpoint = re.sub(r"\s|&|\\\\", "", appendix)
        assert (r"m_d(s)=\tfrac12\big[\min_{\rho\in\mathcalR}\mathcalC\Phi_s(\rho)"
                r"+\max_{\rho\in\mathcalR}\mathcalC\Phi_s(\rho)\big]") in midpoint
    else:
        assert r"V=\operatorname{span}_{\mathbbR}\Delta" in compact
        assert r"\kappa_-c_d\leqD_d\leq\kappa_+c_d" in compact
        assert "y_-(s)" in appendix and "y_+(s)" in appendix
        midpoint = compact.replace("&", "")
        assert any(expression in midpoint for expression in (
            "m_d(s)=[y_-(s)+y_+(s)]/2",
            r"m_d(s)=\tfrac12[y_-(s)+y_+(s)]",
            r"m_d(s)=\frac12[y_-(s)+y_+(s)]",
            r"m_d(s)=\tfrac{1}{2}[y_-(s)+y_+(s)]",
            r"m_d(s)=\frac{1}{2}[y_-(s)+y_+(s)]",
        ))
    assert r"K_i\sim\operatorname{Binomial}(r,a_i)" in appendix
    if version == "icaart2027":
        # Reachable states define the interval midpoint; f_r is its exact
        # Bernstein expectation and the hat identifies Q-sample estimates.
        assert r"f_r(s)=\mathbbE\bigl[m_d(K_1/r,\ldots,K_d/r)\bigr]" in compact
        assert r"$Q$ independent readouts gives $\widehat f_r$" in appendix
        assert re.search(r"\$Q\$ independent (?:samples of its|measurements of this) bounded readout",
                         _section(version, "dynamics"))
    else:
        assert r"\mathcalY_d(s)=\{y(h):S_d(h)=s\}" in compact
        assert r"g_d(a_1,\ldots,a_d)" in compact
        assert r"\rho_t^{(d)}" in appendix
        assert r"f^{\rm mid}" in appendix and r"\widehat f^{\rm mid}" in appendix
        assert "positive integer $Q$ is the shot count" in appendix

    # Retire synonyms that previously collided with input symbols, quantum
    # operators, or the separately reported shot budget.
    for legacy in (
        "Y(h)", r"\widehat Y", "K_d(s)", r"\ell_d", "u_d(s)",
        "a_d=", r"\widetilde\rho", "Q_{d,r}", r"Q^{\rm mid}",
        r"\epsilon", r"\varepsilon_{\rm tot}",
    ):
        assert legacy not in appendix
    assert re.search(r"(?<![A-Za-z\\])p_s\b", appendix) is None
