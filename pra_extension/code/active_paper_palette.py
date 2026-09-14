"""Semantic colors for current ICAART artwork; historical palettes stay frozen."""

from paper_palette import *  # noqa: F403

INPUT = TARGET = INK
QRC = "#D97632"
QELM = COBALT
QRC_LIGHT = "#F7E4D6"
QELM_LIGHT = COBALT_LIGHT
QELM_FOR_TARGET_SIZES = {2: "#3868D9", 3: "#2552AD", 4: "#173772"}
QELM_FEATURE_BUDGETS = {18: "#83A6E8", 36: "#3868D9", 72: "#173772"}
QELM_SELECTED = "#0C234B"

DIAGRAM = {
    "earlier_input": INPUT,
    "carried_state": QRC,
    "explicit_window": QELM,
    "classical_input_injection": INPUT,
    "measured_quantities_and_readouts": FEATURE_FILL,
    "target_output": QRC,
    "ideal_window_output": QELM,
}

# Retain the existing TikZ identifiers so geometry and drawing order do not move.
TIKZ = {
    **TIKZ,
    "blue": QRC,
    "green": QELM,
    "orange": QRC,
    "purple": INPUT,
}
