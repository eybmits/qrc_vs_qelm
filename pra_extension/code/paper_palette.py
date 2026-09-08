"""Author-selected Cobalt palette for the active ICAART paper.

Three plot roots, charcoal and gray, plus the author's restored yellow
observable/readout accent. No scientific parameters live here.
"""

COBALT = "#3868D9"
CORAL = "#D66A56"
VIOLET = "#7C5A9F"
INK = "#20252B"
MUTED = "#8B9198"
GRID = "#DEE0E3"
NEUTRAL = "#F1F2F3"
WHITE = "#FFFFFF"
COBALT_LIGHT = "#E7EDFA"
CORAL_LIGHT = "#FAECE9"
VIOLET_LIGHT = "#EAE5F0"
FEATURE_FILL = "#FFC943"
FEATURE_BACKGROUND = "#FFDD92"

# Match the chosen B preview, retaining the existing categorical marker mapping.
QRC_SIZES = {2: COBALT, 3: VIOLET, 4: CORAL}
FEATURE_BUDGETS = {18: COBALT, 36: CORAL, 72: VIOLET}
DIAGRAM = {
    "earlier_input": INK,
    "carried_state": COBALT,
    "explicit_window": CORAL,
    "classical_input_injection": VIOLET,
    "measured_quantities_and_readouts": FEATURE_FILL,
    "target_output": INK,
    "ideal_window_output": CORAL,
}

# Legacy TikZ identifiers are retained to keep the existing geometry unchanged.
# 'green' now denotes coral; 'gold' retains the original yellow feature fill.
TIKZ = {
    "ink": INK, "blue": COBALT, "green": CORAL, "orange": CORAL,
    "gold": FEATURE_FILL, "purple": VIOLET, "muted": MUTED,
    "neutral": NEUTRAL, "palegold": FEATURE_BACKGROUND, "rule": GRID,
}
