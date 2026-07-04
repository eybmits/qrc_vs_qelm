import importlib.util
import subprocess
import sys
from pathlib import Path

from qrc_vs_qelm.gates import gates_markdown_table


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_prereg.py"
PREREG = ROOT / "PREREGISTRATION.md"

spec = importlib.util.spec_from_file_location("render_prereg", SCRIPT)
render_prereg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = render_prereg
spec.loader.exec_module(render_prereg)


def test_render_text_replaces_only_gate_block():
    text = "before\n<!-- GATES:BEGIN -->\nstale\n<!-- GATES:END -->\nafter\n"

    rendered = render_prereg.render_text(text)

    assert rendered.startswith("before\n")
    assert rendered.endswith("\nafter\n")
    assert gates_markdown_table() in rendered
    assert "stale" not in rendered


def test_prereg_gate_table_is_in_sync():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    text = PREREG.read_text(encoding="utf-8")
    assert gates_markdown_table() in text
