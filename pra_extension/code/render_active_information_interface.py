#!/usr/bin/env python3
"""Render the current Figure 1 without changing frozen historical renderers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import active_paper_palette as active_colors
from figure1_timeseries import provenance as timeseries_provenance
from figure1_timeseries import output_provenance as timeseries_output_provenance
from render_information_interface_figures import (
    DEFAULT_DARK_SUMMARY, DEFAULT_OUTPUT, REPO_ROOT, _load_dark_summary, _sha256,
)

def active_figure1_output_provenance() -> dict[str, Any]:
    """Describe current output colors without coupling the toy to input samples."""
    output = timeseries_output_provenance()
    output["styles"] = {
        "target": "orange QRC thin line with dense filled markers",
        "ideal_window": "blue QELM thin line with open markers",
    }
    for case in output["cases"].values():
        case["preservation_scope"] = case["preservation_scope"].replace(
            "charcoal target", "orange QRC target"
        )
    output["display_headroom"] = output["display_headroom"].replace(
        "charcoal targets", "orange QRC targets"
    )
    return output



def active_figure1_provenance(source: Path) -> dict[str, Any]:
    """Apply current color semantics to the unchanged historical sample audit."""
    payload = timeseries_provenance(source)
    payload["palette_name"] = "Semantic input/QRC/QELM"
    payload["diagram_encoding"] = dict(active_colors.DIAGRAM)
    payload["legacy_tikz_color_tokens"] = dict(active_colors.TIKZ)
    payload["past_state_cue"]["root_color"] = active_colors.QRC
    payload["output_comparisons"] = active_figure1_output_provenance()
    payload["palette_sources"] = {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in (
            Path(__file__).with_name("active_paper_palette.py"),
            Path(__file__).with_name("paper_palette.py"),
        )
    }
    return payload



def render_figure_1(dark_summary_path: Path, output_dir: Path) -> tuple[Path, ...]:
    """Compile the native-size TikZ explanatory graphic without rasterizing text."""

    _load_dark_summary(dark_summary_path)
    pdf = output_dir / "fig01_information_interface.pdf"
    png = output_dir / "fig01_information_interface.png"
    # Preserve the old provenance byte-for-byte: the prospective N5 seed audit
    # froze it before the current drawing revision. New renders have a separate
    # provenance file, never an updated historical audit snapshot.
    source = output_dir / "fig01_information_interface_render_source.json"
    historical_source = DEFAULT_OUTPUT / "fig01_information_interface_source.json"
    historical_sha256 = _sha256(historical_source)
    if historical_sha256 != (
        "d5a76f449661e3f4fabf909a6271208f838907f076db1c02652b771665d23965"
    ):
        raise ValueError("The frozen Figure 1 input-audit snapshot changed")
    tikz_source = DEFAULT_OUTPUT / "fig01_information_interface.tex"
    input_timeseries = active_figure1_provenance(tikz_source)
    with tempfile.TemporaryDirectory(prefix="qrc-tikz-") as build:
        subprocess.run(
            ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
             "-halt-on-error", f"-output-directory={build}", str(tikz_source)],
            check=True, capture_output=True, text=True,
        )
        shutil.copyfile(Path(build) / pdf.name, pdf)
        subprocess.run(
            ["pdftoppm", "-singlefile", "-r", "320", "-png", str(pdf),
             str(png.with_suffix(""))],
            check=True, capture_output=True, text=True,
        )
    source_payload = {
        "historical_input_audit": {
            "path": str(historical_source.relative_to(REPO_ROOT)),
            "sha256": historical_sha256,
            "status": "historical snapshot retained for the prospective N5 seed audit; "
                      "not the source description of the current drawing",
        },
        "input_timeseries": input_timeseries,
        "selected_design": "variant_a_architectures",
        "removed_history_error_panel": input_timeseries["removed_history_error_panel"],
        "canvas_aspect_ratio": 17.9197 / 9.60,
        "canvas_bounds_cm": [0.0, 3.60, 17.9197, 13.2],
        "figure_question": (
            "when can explicit finite context reproduce recurrent input-output behavior"
        ),
        "panels": {
            "architecture": "recurrent and reset-window architectures with unchanged illustrative input and causal access",
            "fading": "upper output frame: this old bit is fully forgotten by t=40; the ideal prediction remains exact at fixed d=40",
            "persistent": "lower output frame: gradual revelation of a persistent old bit as time advances at the same fixed d=40",
        },
        "panel_letters": [],
        "theorem_context": {
            "status": "manuscript interpretation, not a plotted curve or measured property of these toy traces",
            "common_suffix_diameter": "supremal declared-output distance between histories sharing the last d inputs",
            "scalar_information_limit": "D_d/2 is the optimal worst-case absolute scalar error due to missing history",
            "emulation_statement": (
                "a growing exact-expectation reset-QELM family universal on each finite "
                "window emulates uniformly to arbitrary accuracy exactly when D_d tends to zero"
            ),
            "regimes": {
                "fading": {
                    "condition": "D_d tends to zero; r_vis < 1 under the spectral theorem assumptions",
                    "consequence": "arbitrary accuracy as d grows, without a guarantee of exact emulation at a fixed finite d",
                },
                "finite": {
                    "condition": "D_d = 0 for some finite d",
                    "consequence": "output exactly determined by finite context",
                },
                "persistent": {
                    "condition": "D_d >= Delta > 0 for every d; r_vis = 1",
                    "consequence": "scalar worst-case error is at least Delta/2",
                },
            },
            "memory_relevance": (
                "past inputs must create the state difference and present or future "
                "declared outputs must reveal it; full-state forgetting is not required"
            ),
            "scope": (
                "finite-dimensional QRC with input-continuous channels and fixed finite "
                "observable output; the iff statement additionally assumes compact inputs "
                "with freely concatenable suffixes and per-window reset-QELM density"
            ),
            "illustration_boundary": (
                "The displayed time-dependent two-history gaps are not generic D_d/2 curves. "
                "Decay with elapsed time at fixed d does not establish uniform convergence as d grows."
            ),
        },
        "schematic_curves": True,
        "output_comparisons": input_timeseries["output_comparisons"],
        "scope": "Two analytic balanced-bit illustrations with known case-specific baselines, ideal window conditional means, "
                 "fixed d and evolving t; not simulated, fitted or measured QRC/QELM performance.",
        "upstream_dark_sector_summary_sha256": _sha256(dark_summary_path),
        "renderer_source": str(Path(__file__).relative_to(REPO_ROOT)),
        "renderer_source_sha256": _sha256(Path(__file__)),
        "tikz_source_sha256": _sha256(tikz_source),
        "typography": "native 11 pt labels and one shared 10 pt legend at 7.055 inch width; no panel letters",
        "history_access": "the interior point at t=180 supplies the current scalar; "
                          "older inputs act through the carried-state loop; the exact causal "
                          "window is 141--180 and grey future samples 181--200 are display-only",
        "past_input_cue": "blue!16 behind the older-input interval, ending where the green!24 window starts; "
                          "a 0.4-pt solid green!65!ink boundary marks index 140.5, distinct from the dashed current-time guide; "
                          "orange brace labeled past carried in / QRC state rho_(t-1); "
                          "not a second input wire, history replay, or claim of lossless history storage",
        "lane_spacing": "the 4.15 cm square input and all architecture coordinates retain their positions and native labels; "
                        "both dynamics groups retain their 0.35 cm rightward offset and observable banks their 0.15 cm offset; "
                        "the orthogonal scalar connector enters the QRC top center vertically downward; "
                        "the reset lane is 0.30 cm lower to separate the model rows; "
                        "a blue causal-window brace feeds the reset encoder from below without crossing quantum wires; "
                        "two 4.15-by-3.2 cm output frames occupy y=8.58..11.78 and y=4.99..8.19 with matching axes; "
                        "external rounded labels explain window sufficiency in each case; a single shared legend overlaps their facing borders; "
                        "the removed lower history-error panel permits a 17.9197-by-9.60 cm canvas",
        "architecture_motifs": "three-wire general QRC channel block and explicit reset encoding plus fixed gates; "
                               "matching yellow observable banks with softly rounded cards and connectors; "
                               "faint downstream lanes denote omitted intermediate observables, not extra quantum wires; "
                               "compact rounded unlabeled readout triangles; both outputs feed the paired memory-case comparisons",
        "readout_scope": "fixed target readout (identity for the declared observable vector); "
                         "trained reset-QELM emulator readout; operations remain defined by the caption and model section "
                         "while endpoints are labeled y_t and its hatted counterpart, with no trained-readout text",
        "observable_notation": "O_1,...,O_m label target observables and P_1,...,P_B label QELM features; "
                               "no identical-operator or equal-feature-count claim",
        "annotation_scope": "no standalone zoom or schematic tags; one unchanged input amplitude scale; "
                            "external Fading memory / Window sufficient and Persistent memory / Window insufficient boxes; "
                            "green!24 before d and blue!16 after d inside both output frames encode the old bit's window access; "
                            "a single one-row reference-style key overlaps the facing borders between both frames "
                            "and identifies QRC and QELM, with no independent third truth trajectory; "
                            "time tick labels appear only on the lower frame; input-step and time axis titles are omitted; "
                            "no panel letters, epsilon or D_d/2 appear in the artwork; "
                            "caption identifies the ideal toy reference, illustrative status and future exclusion",
    }
    source.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pdf, png, source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = render_figure_1(DEFAULT_DARK_SUMMARY, args.output_dir)
    ledger_path = args.output_dir / "figure_sha256.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    ledger.update({path.name: _sha256(path) for path in artifacts})
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print("\n".join(str(path) for path in artifacts))
