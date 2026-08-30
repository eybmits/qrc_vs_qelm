import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_paper_figures.py"

spec = importlib.util.spec_from_file_location("render_paper_figures", SCRIPT)
render_paper_figures = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = render_paper_figures
spec.loader.exec_module(render_paper_figures)


def test_committed_phase_artifacts_render(tmp_path):
    cells = render_paper_figures.load_phase_cells(
        ROOT / "paper" / "figures" / "stage_gate_summary.csv"
    )
    uninformative = render_paper_figures.load_uninformative_cells(
        ROOT / "paper" / "figures" / "REPORT.md"
    )

    assert len(cells) == 27
    assert len(uninformative) == 12
    assert {cell.label for cell in cells} == {"QRC", "QELM", "tie"}

    output = tmp_path / "phase.pdf"
    render_paper_figures.render_phase_diagram(cells, uninformative, output)
    assert output.stat().st_size > 10_000
    assert output.with_suffix(".png").stat().st_size > 10_000
