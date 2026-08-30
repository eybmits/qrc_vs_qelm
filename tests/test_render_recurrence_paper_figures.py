import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_recurrence_paper_figures.py"

spec = importlib.util.spec_from_file_location("render_recurrence_paper_figures", SCRIPT)
render = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = render
spec.loader.exec_module(render)


def test_heldout_paper_summaries_and_figures(tmp_path):
    results = ROOT / "results_recurrence_fingerprint_heldout"
    blocks = render.load_blocks(ROOT / "data" / "frozen" / "recurrence_fingerprint_v1.json")
    baselines, classification = render.fingerprint_source_tables(results, blocks)
    task_rows = render.task_source_table(results, blocks)

    assert len(baselines) == 24
    assert sum(row["below_registered_floor"] for row in baselines) == 8
    assert {row["dissipator"] for row in baselines if row["below_registered_floor"]} == {"dephasing"}
    assert sum(row["wrong"] for row in classification) == 0
    assert sum(row["correct"] for row in classification) == 895
    assert sum(row["abstain"] for row in classification) == 65
    assert len(task_rows) == 120

    fingerprint = tmp_path / "fingerprint.pdf"
    crossover = tmp_path / "crossover.pdf"
    schematic = tmp_path / "schematic.pdf"
    render.render_fingerprint_visibility(baselines, classification, fingerprint)
    render.render_task_crossover(task_rows, results, blocks, crossover)
    render.render_assay_schematic(schematic)
    for path in (
        fingerprint,
        fingerprint.with_suffix(".png"),
        crossover,
        crossover.with_suffix(".png"),
        schematic,
        schematic.with_suffix(".png"),
    ):
        assert path.stat().st_size > 10_000
