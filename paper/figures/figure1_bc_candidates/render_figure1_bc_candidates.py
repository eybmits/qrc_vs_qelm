#!/usr/bin/env python3
"""Render ten editorial TikZ alternatives without touching manuscript artifacts.

Run from the repository with its virtual environment. Requires pdflatex,
pdfinfo, and pdftoppm; no simulations, remote services, or data files are used.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DESIGNS = ROOT / "paper/figures/figure1_bc_candidates"
NAMES = tuple(f"{panel}{index}" for panel in "bc" for index in range(1, 6))
NATIVE_CM = (8.60, 5.60)
PROTECTED = (
    "paper/canonical_emulation/main.pdf",
    "paper/icaart2027/submission.pdf",
    "paper/figures/information_interface/fig01_information_interface.tex",
    "paper/figures/information_interface/fig01_information_interface.pdf",
    "paper/figures/information_interface/fig01_information_interface_source.json",
    "paper/figures/figure1_variants/variant_a_architectures.tex",
    "pra_extension/PRA_EXTENSION_RELEASE.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_snapshot() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in PROTECTED}


def pdf_dimensions(path: Path, pages: int = 1) -> tuple[float, float]:
    info = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True,
    ).stdout
    if not re.search(rf"Pages:\s+{pages}\b", info):
        raise ValueError(f"Expected {pages} page(s): {path.name}")
    match = re.search(r"Page size:\s*([\d.]+) x ([\d.]+) pts", info)
    if not match:
        raise ValueError(f"Cannot read canvas: {path.name}")
    return tuple(map(float, match.groups()))


def compile_tex(source: Path, destination: Path, passes: int = 1) -> None:
    with tempfile.TemporaryDirectory(prefix="qrc-bc-tikz-") as build:
        for _ in range(passes):
            result = subprocess.run(
                [
                    "pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
                    "-halt-on-error", f"-output-directory={build}", source.name,
                ],
                cwd=source.parent, capture_output=True, text=True,
            )
            if result.returncode:
                raise RuntimeError(f"{source.name}\n{result.stdout[-6000:]}")
            if re.search(r"Overfull \\[hv]box", result.stdout):
                raise RuntimeError(f"Overflow in {source.name}\n{result.stdout[-6000:]}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(build) / source.with_suffix(".pdf").name, destination)


def load_descriptions(directory: Path = DESIGNS) -> list[dict[str, str]]:
    descriptions = []
    for panel in "bc":
        entries = json.loads((directory / f"{panel}_descriptions.json").read_text())
        if not isinstance(entries, list):
            raise ValueError(f"{panel}_descriptions.json must contain a list")
        descriptions.extend(entries)
    if [entry.get("id", "").lower() for entry in descriptions] != list(NAMES):
        raise ValueError("Need exactly B1-B5 followed by C1-C5")
    for entry in descriptions:
        if not all(entry.get(field, "").strip() for field in (
            "id", "title_de", "caption_de", "scope_en",
        )):
            raise ValueError(f"Incomplete selection description: {entry}")
    return descriptions


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "–": "--", "—": "---", "−": "-", "→": r"$\to$",
        "×": r"$\times$", "≤": r"$\leq$", "≥": r"$\geq$",
    }
    return "".join(replacements.get(character, character) for character in text)


def selection_tex(descriptions: list[dict[str, str]], directory: Path = DESIGNS) -> str:
    """Two A4 contact sheets; all candidate PDFs retain their native scale."""
    captions = json.loads((directory / "selection_guide.json").read_text())
    lines = [
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}",
        r"\usepackage{helvet,graphicx,tikz,amsmath}",
        r"\usepackage[a4paper,margin=0pt]{geometry}",
        r"\renewcommand{\familydefault}{\sfdefault}\pagestyle{empty}",
        r"\pdfinfoomitdate=1\pdftrailerid{}",
        r"\definecolor{muted}{HTML}{747474}",
        r"\definecolor{rule}{HTML}{D9D9D9}",
        "% Native-scale selection sheets.",
        r"\begin{document}",
    ]
    page_titles = (
        "B / Wie viel Vergangenheit fehlt?",
        "C / Welche Zustandsunterschiede zählen?",
    )
    for page, panel in enumerate("bc"):
        if page:
            lines.append(r"\newpage")
        lines.extend([
            r"\null\begin{tikzpicture}[remember picture,overlay,x=1cm,y=-1cm]",
            r"\begin{scope}[shift={(current page.north west)}]",
            r"\node[anchor=north west,inner sep=0,font=\bfseries\fontsize{15}{18}\selectfont] at (1.6,.85) {"
            + tex_escape(page_titles[page]) + "};",
            r"\node[anchor=north west,inner sep=0,font=\fontsize{10}{12}\selectfont,text=muted] at (1.6,1.60)"
            r" {Wähle eine B- und eine C-Variante. Die Nummern müssen nicht übereinstimmen.};",
        ])
        for index, entry in enumerate(descriptions[page * 5:page * 5 + 5]):
            column, row = index % 2, index // 2
            x, y = 1.6 + 9.2 * column, 2.30 + 8.0 * row
            heading = tex_escape(entry["id"] + "  /  " + entry["title_de"])
            lines.append(
                r"\node[anchor=north west,inner sep=0,font=\bfseries\fontsize{10.5}{12}\selectfont] at "
                f"({x:.2f},{y:.2f}) {{{heading}}};"
            )
            lines.append(
                r"\node[anchor=north west,inner sep=0] at "
                f"({x:.2f},{y + .48:.2f}) {{"
                r"\includegraphics[width=8.60cm]{" + entry["id"].lower() + ".pdf}};"
            )
            lines.append(
                r"\node[anchor=north west,inner sep=0,text width=8.60cm,align=left,"
                r"font=\fontsize{10}{12}\selectfont] at "
                f"({x:.2f},{y + 6.35:.2f}) {{{tex_escape(captions[entry['id']])}}};"
            )
        note = (
            "Alle Varianten erklären die Grenze durch fehlende Vergangenheit. "
            "Ein kleines Geschichtsproblem allein garantiert noch keine ausreichend "
            "leistungsfähige konkrete QELM."
            if panel == "b" else
            "Gleiche Ausgabe jetzt genügt nicht. Unsichtbar bedeutet hier: "
            "keine erlaubte gemeinsame Fortsetzung kann den Unterschied in den "
            "festgelegten Ausgaben sichtbar machen."
        )
        lines.extend([
            r"\draw[rule] (10.8,19.0)--(19.4,19.0);",
            r"\node[anchor=north west,inner sep=0,text width=8.60cm,align=left,"
            r"font=\fontsize{11}{14}\selectfont] at (10.8,19.4) {"
            + tex_escape(note) + "};",
            r"\node[anchor=north west,inner sep=0,text width=8.60cm,align=left,"
            r"font=\fontsize{10}{12}\selectfont,text=muted] at (10.8,22.2)"
            r" {Alle Zeichnungen sind konzeptionelle Erklärungen, keine neuen Messdaten. "
            r"Das aktuelle Paper bleibt bis zur Auswahl unverändert.};",
            r"\draw[rule] (1.6,27.0)--(19.4,27.0);",
            r"\node[anchor=north west,inner sep=0,text width=17.8cm,align=left,"
            r"font=\fontsize{9}{11}\selectfont,text=muted] at (1.6,27.3)"
            r" {Originalgröße je Entwurf: 8,6 $\times$ 5,6 cm; 10-pt-Beschriftung. "
            r"Gestalterische Referenz: Fig. 1 in Freiheit und Campaioli, "
            r"\textit{Superextensive learning in quantum reservoirs at the onset of information scrambling}. "
            r"Die Diagramme hier erklären ausschließlich den Inhalt unseres Papers.};",
            r"\end{scope}\end{tikzpicture}",
        ])
    return "\n".join(lines + [r"\end{document}", ""])


def render_candidate(name: str, directory: Path) -> dict:
    compile_tex(directory / f"{name}.tex", directory / f"{name}.pdf")
    size = pdf_dimensions(directory / f"{name}.pdf")
    expected = tuple(cm * 72 / 2.54 for cm in NATIVE_CM)
    if any(abs(measured - target) > .1 for measured, target in zip(size, expected)):
        raise ValueError(f"{name}: content escapes the native canvas: {size} != {expected}")
    subprocess.run(
        ["pdftoppm", "-singlefile", "-scale-to-x", "1400", "-scale-to-y", "-1",
         "-png", str(directory / f"{name}.pdf"), str(directory / name)],
        check=True, capture_output=True, text=True,
    )
    return {"pdf_size_points": size, "conceptual_only": True}


def render_all(directory: Path = DESIGNS) -> dict:
    before = protected_snapshot()
    source_names = [f"{name}.tex" for name in NAMES] + [
        "style.tex", "DESIGN_BRIEF.md", "README.md", "b_descriptions.json",
        "c_descriptions.json", "selection_guide.json",
    ]
    source_before = {name: sha256(directory / name) for name in source_names}
    descriptions = load_descriptions(directory)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rendered = list(pool.map(lambda name: render_candidate(name, directory), NAMES))
    gallery = directory / "selection.tex"
    gallery.write_text(selection_tex(descriptions, directory), encoding="utf-8")
    # Absolute page anchors require a second LaTeX pass.
    compile_tex(gallery, directory / "selection.pdf", passes=2)
    pdf_dimensions(directory / "selection.pdf", pages=2)
    subprocess.run(
        ["pdftoppm", "-r", "180", "-png", str(directory / "selection.pdf"),
         str(directory / "selection")],
        check=True, capture_output=True, text=True,
    )
    for page, panel in enumerate("bc", start=1):
        (directory / f"selection-{page}.png").replace(directory / f"overview_{panel}.png")
    after = protected_snapshot()
    if before != after:
        raise RuntimeError("A protected manuscript/figure/release changed during rendering")
    if source_before != {name: sha256(directory / name) for name in source_names}:
        raise RuntimeError("Candidate sources changed during rendering; rebuild from stable sources")
    filenames = ["style.tex", "DESIGN_BRIEF.md", "README.md", "b_descriptions.json", "c_descriptions.json",
                 "selection_guide.json", "render_figure1_bc_candidates.py", "test_figure1_bc_candidates.py"]
    filenames += [f"{name}.{extension}" for name in NAMES for extension in ("tex", "pdf", "png")]
    filenames += ["selection.tex", "selection.pdf", "overview_b.png", "overview_c.png"]
    manifest = {
        "schema": "figure1_bc_selection_v1",
        "status": "awaiting_user_selection_no_manuscript_replacement",
        "selected_b": None, "selected_c": None,
        "native_size_cm": NATIVE_CM, "label_size_pt": 10,
        "reference": {
            "title": "Superextensive learning in quantum reservoirs at the onset of information scrambling",
            "supplied_filename": "2608.25511v1 (5).pdf", "page": 2, "figure": 1,
            "use": "presentation reference only; no reference-paper data reused",
        },
        "scope": "Ten conceptual illustrations; not simulations, uncertainty bands, or new evidence.",
        "variants": {
            name: {**entry, **description}
            for name, entry, description in zip(NAMES, rendered, descriptions)
        },
        "unchanged_during_build": {"before": before, "after": after},
        "renderer_sha256": sha256(Path(__file__)),
        "artifacts": {name: sha256(directory / name) for name in filenames},
    }
    (directory / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    manifest = render_all()
    print(json.dumps({"status": manifest["status"], "variants": list(manifest["variants"]),
                      "selection_pdf": str(DESIGNS / "selection.pdf")}, indent=2))
