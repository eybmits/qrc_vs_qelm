#!/usr/bin/env python
"""Render generated sections in PREREGISTRATION.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrc_vs_qelm.gates import gates_markdown_table


BEGIN = "<!-- GATES:BEGIN -->"
END = "<!-- GATES:END -->"


def gate_block() -> str:
    """Return the complete synchronized gate block."""

    return f"{BEGIN}\n## Gates\n\n{gates_markdown_table()}\n{END}"


def render_text(text: str) -> str:
    """Replace the marked gate block in ``text``."""

    start = text.find(BEGIN)
    end = text.find(END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"could not find ordered {BEGIN!r}/{END!r} markers")
    end += len(END)
    return text[:start] + gate_block() + text[end:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=ROOT / "PREREGISTRATION.md",
        help="Markdown file to update or check.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the rendered file is out of sync.",
    )
    args = parser.parse_args(argv)

    text = args.path.read_text(encoding="utf-8")
    rendered = render_text(text)
    if args.check:
        if rendered != text:
            print(f"{args.path} is out of sync; run scripts/render_prereg.py", file=sys.stderr)
            return 1
        return 0

    if rendered != text:
        args.path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
