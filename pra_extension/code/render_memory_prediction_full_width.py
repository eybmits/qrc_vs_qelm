"""Full-width, taller layout draft of Figure 3; scientific sources are unchanged."""
from copy import deepcopy
import json
from pathlib import Path

import matplotlib.pyplot as plt

import render_memory_prediction as base

SIZE = (6.221, 4.69)
WIDTH = 1.51
LEFT = {"a": .47, "b": 2.53, "c": 4.59}
TOP_BOTTOM, MIDDLE_BOTTOM, LOWER_BOTTOM = 3.53, 2.52, .87
UPPER_HEIGHT, LOWER_HEIGHT = .78, 1.00
TITLE_Y, LOWER_TITLE_Y, FOOTER_Y = 4.51, 1.97, .42
OUTPUT = base.ROOT / "paper/figures/memory_prediction_full_width"
STEM = "fig04_memory_prediction_full_width"


def position(ax, column, bottom, height):
    ax.set_position([LEFT[column] / SIZE[0], bottom / SIZE[1],
                     WIDTH / SIZE[0], height / SIZE[1]])


def text_position(text, x, y, fig):
    text.set_transform(fig.transFigure)
    text.set_position((x / SIZE[0], y / SIZE[1]))


def make_figure(*sources):
    fig, p = base.make_figure(*sources)
    fig.set_size_inches(*SIZE, forward=True)
    rows = ((p["laser"][0], p["history_inset"], p["narma"]["qrc"]),
            (p["laser"][1], p["memory"], p["narma"]["qelm"]),
            (p["laser"][2], p["exact"], p["narma"]["history"]))
    for axes, bottom, height in zip(rows,
            (TOP_BOTTOM, MIDDLE_BOTTOM, LOWER_BOTTOM),
            (UPPER_HEIGHT, UPPER_HEIGHT, LOWER_HEIGHT), strict=True):
        for column, ax in zip(LEFT, axes, strict=True):
            position(ax, column, bottom, height)

    main_texts = fig.texts[:6]
    for column, letter, title in zip(LEFT, main_texts[::2], main_texts[1::2], strict=True):
        text_position(letter, LEFT[column], TITLE_Y, fig)
        text_position(title, LEFT[column] + .15, TITLE_Y, fig)
    for key, column, y in (("Fast loss", "a", 3.38),
                            ("Slow loss", "a", LOWER_TITLE_Y),
                            ("exact", "b", LOWER_TITLE_Y),
                            ("history", "c", LOWER_TITLE_Y)):
        text_position(p["section_headings"][key], LEFT[column], y, fig)
    p["section_headings"]["history"].set_text("How much history is needed?")
    for ax, y in zip(p["laser"][1:], (3.38, LOWER_TITLE_Y), strict=True):
        text_position(ax.texts[0], LEFT["a"] + WIDTH, y, fig)

    for model, bottom in (("qrc", TOP_BOTTOM), ("qelm", MIDDLE_BOTTOM)):
        text_position(p["narma"][model].texts[0], LEFT["c"] + WIDTH,
                      bottom + UPPER_HEIGHT + .025, fig)
    p["narma_target_legend"].set_bbox_to_anchor(
        (LEFT["c"] / SIZE[0], (TOP_BOTTOM + UPPER_HEIGHT + .025) / SIZE[1]),
        transform=fig.transFigure)

    for key, column in (("model_legend", "a"), ("exact_legend", "b"), ("narma_legend", "c")):
        p[key].set_bbox_to_anchor((LEFT[column] / SIZE[0], FOOTER_Y / SIZE[1]),
                                 transform=fig.transFigure)
    memory = p["memory"]
    memory.get_legend().remove()
    p["memory_legend"] = memory.legend(loc="upper right", ncol=2,
        frameon=False, fontsize=7.5, handlelength=.9, handletextpad=.3,
        columnspacing=.8, borderpad=.25)
    memory.texts[0].set_text("Same window from input 4")
    memory.texts[0].set_y(.30)
    for label, words in zip(p["history_inset"].texts,
            ("Different earlier\ninputs", "Same inputs\nfrom here on"), strict=True):
        label.set_text(words)
    return fig, p


def render():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parent_path = base.OUTPUT / f"{base.STEM}_source.json"
    parent = json.loads(parent_path.read_text())
    for path, expected in parent["sources"].items():
        if base.digest(base.ROOT / path) != expected:
            raise ValueError(f"Changed parent dependency: {path}")
    with plt.rc_context():
        fig, _ = make_figure(*base.load_sources())
        fig.savefig(OUTPUT / f"{STEM}.pdf", metadata={
            "Title": "Memory emulation and prediction: full-width layout draft",
            "Creator": "qrc_vs_qelm", "CreationDate": None, "ModDate": None})
        fig.savefig(OUTPUT / f"{STEM}.png", dpi=300, metadata={"Software": "qrc_vs_qelm"})
        plt.close(fig)
    ledger = deepcopy(parent)
    ledger["schema"] = "three_panel_memory_prediction_full_width_layout_v1"
    ledger["figure_size_inches"] = list(SIZE)
    ledger["layout"] = {
        "column_left_inches": LEFT, "plot_width_inches": WIDTH,
        "upper_row_bottom_height_inches": [TOP_BOTTOM, UPPER_HEIGHT],
        "middle_row_bottom_height_inches": [MIDDLE_BOTTOM, UPPER_HEIGHT],
        "lower_row_bottom_height_inches": [LOWER_BOTTOM, LOWER_HEIGHT],
        "panel_title_y_inches": TITLE_Y, "lower_title_y_inches": LOWER_TITLE_Y,
        "footer_legend_top_inches": FOOTER_Y,
        "purpose": "Full text width with physically taller plots in every row; local layout draft."}
    ledger["parents"]["prior_layout"] = str(parent_path.relative_to(base.ROOT))
    ledger["sources"].update({str(path.relative_to(base.ROOT)): base.digest(path)
                              for path in (Path(__file__), parent_path)})
    ledger["artifacts"] = {f"{STEM}.{ext}": base.digest(OUTPUT / f"{STEM}.{ext}")
                           for ext in ("pdf", "png")}
    (OUTPUT / f"{STEM}_source.json").write_text(json.dumps(ledger, indent=2) + "\n")
    return ledger


if __name__ == "__main__":
    render()
