#!/usr/bin/env python3
"""Build the selected compact TikZ overview and its two retained alternatives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from figure1_timeseries import provenance as timeseries_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "paper/figures/figure1_variants"
VARIANTS = (
    "variant_a_architectures",
    "variant_b_same_window",
    "variant_c_relevant_memory",
)
SELECTED_VARIANT = "variant_a_architectures"
CANONICAL_SOURCE = (
    REPO_ROOT / "paper/figures/information_interface/fig01_information_interface.tex"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_all(output_dir: Path = OUTPUT) -> dict:
    selected_source = output_dir / f"{SELECTED_VARIANT}.tex"
    if selected_source.read_bytes() != CANONICAL_SOURCE.read_bytes():
        raise ValueError("selected Figure 1 and canonical TikZ sources differ")
    manifest = {
        "schema": "figure1_design_candidates_v2",
        "status": "variant_a_selected_for_main_paper",
        "selected_variant": SELECTED_VARIANT,
        "canonical_source": str(CANONICAL_SOURCE.relative_to(REPO_ROOT)),
        "canonical_source_sha256": sha256(CANONICAL_SOURCE),
        "input_timeseries_sha256": timeseries_provenance(CANONICAL_SOURCE)["float64_le_sha256"],
        "scope": "schematic layouts only; not additional numerical evidence",
        "native_width_inches": 7.055,
        "design_notes_sha256": sha256(output_dir / "DESIGN_NOTES.md"),
        "variants": {},
    }
    for name in VARIANTS:
        source = output_dir / f"{name}.tex"
        if not source.is_file():
            raise FileNotFoundError(source)
        pdf = output_dir / f"{name}.pdf"
        with tempfile.TemporaryDirectory(prefix="qrc-figure1-candidate-") as build:
            subprocess.run(
                [
                    "pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
                    "-halt-on-error", f"-output-directory={build}", str(source),
                ],
                check=True, capture_output=True, text=True,
            )
            shutil.copyfile(Path(build) / pdf.name, pdf)
        info = subprocess.run(
            ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True,
        ).stdout
        match = re.search(r"Page size:\s*([\d.]+) x ([\d.]+) pts", info)
        if not match or not re.search(r"Pages:\s+1\b", info):
            raise ValueError(f"{name}: expected one measurable PDF page")
        width, height = map(float, match.groups())
        expected_aspect = 17.9197 / 13.0 if name == SELECTED_VARIANT else 1.0
        if abs(width / height - expected_aspect) > 0.001:
            raise ValueError(f"{name}: unexpected canvas {width} x {height}")
        subprocess.run(
            ["pdftoppm", "-singlefile", "-scale-to", "1600", "-png",
             str(pdf), str(output_dir / name)],
            check=True, capture_output=True, text=True,
        )
        manifest["variants"][name] = {
            "canvas_aspect_ratio": width / height,
            "pdf_size_points": [width, height],
            "artifacts": {
                f"{name}.{extension}": sha256(output_dir / f"{name}.{extension}")
                for extension in ("tex", "pdf", "png")
            },
        }
    (output_dir / "design_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(render_all(), indent=2, sort_keys=True))
