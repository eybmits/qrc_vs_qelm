"""Reproducible Figure 1 input and a separate analytic memory-access toy."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from qrc_vs_qelm.core.tasks import mackey_glass_series


ROOT = Path(__file__).resolve().parents[2]
N_SAMPLES = 201
SELECTED_STEP = 180
WINDOW = 40
WINDOW_START = SELECTED_STEP - WINDOW + 1
SEED = 1
PARAMETERS = {
    "tau": 17, "beta": 0.2, "gamma": 0.1, "n_exp": 10,
    "dt": 1.0, "sample_every": 1, "x0": 1.2, "discard": 1000,
}
BEGIN = "% BEGIN REPRODUCIBLE INPUT SAMPLES"
END = "% END REPRODUCIBLE INPUT SAMPLES"
OUTPUT_BEGIN = "% BEGIN ILLUSTRATIVE OUTPUT COMPARISONS"
OUTPUT_END = "% END ILLUSTRATIVE OUTPUT COMPARISONS"
OUTPUT_N_SAMPLES = 101
OUTPUT_SEED = 2
OUTPUT_WINDOW = 40
OUTPUT_WRITE_STEP = 0
OUTPUT_MEMORY_AMPLITUDE = 0.25


def output_baseline() -> np.ndarray:
    """Return a known common drawing baseline, independent of the stored bit."""
    raw = mackey_glass_series(
        OUTPUT_N_SAMPLES, rng=np.random.default_rng(OUTPUT_SEED), **PARAMETERS,
    )
    return 0.4 + 0.65 * (raw - np.min(raw)) / (np.max(raw) - np.min(raw))


def output_samples() -> np.ndarray:
    """Return the +1-history target and ideal window conditional-mean toy."""
    k = np.arange(OUTPUT_N_SAMPLES, dtype=float)
    baseline = output_baseline()
    target = baseline + OUTPUT_MEMORY_AMPLITUDE
    prediction = baseline + OUTPUT_MEMORY_AMPLITUDE * (k < OUTPUT_WRITE_STEP + OUTPUT_WINDOW)
    return np.column_stack((target, prediction))


def tikz_output_sample_block() -> str:
    rows = ",\n".join(
        f"{k}/" + "/".join(f"{value:.17g}" for value in row)
        for k, row in enumerate(output_samples())
    )
    return f"{OUTPUT_BEGIN}\n\\def\\OutputSamples{{%\n{rows}\n}}\n{OUTPUT_END}"


def output_provenance() -> dict:
    values = output_samples()
    baseline = output_baseline()
    other_history = baseline - OUTPUT_MEMORY_AMPLITUDE
    return {
        "kind": "analytic balanced-bit memory-access toy with an ideal window conditional mean",
        "status": "schematic information-access example, not simulated, fitted, or measured QRC/QELM performance",
        "generator": "figure1_timeseries.output_samples",
        "sample_count": OUTPUT_N_SAMPLES,
        "sample_indices": list(range(OUTPUT_N_SAMPLES)),
        "column_order": ["target", "prediction"],
        "template_stream": {
            "function": "mackey_glass_series",
            "seed": OUTPUT_SEED,
            "parameters": PARAMETERS,
            "generated_count": OUTPUT_N_SAMPLES,
            "indices": [0, OUTPUT_N_SAMPLES - 1],
            "scope": "A separate Mackey-Glass stream supplies a predetermined common baseline shape. "
                     "Its drawing normalization is not fitted preprocessing, future-input access, or a simulated reservoir output.",
        },
        "formulas": {
            "baseline": "g[k]=0.4+0.65*(raw[k]-min(raw))/(max(raw)-min(raw))",
            "target": "g[k]+0.25 for the displayed bit b=+1",
            "other_history": "g[k]-0.25 for bit b=-1",
            "prediction": "g[k]+0.25 for k<40; g[k] for k>=40",
        },
        "values": values.tolist(),
        "baseline_values": baseline.tolist(),
        "other_history_values": other_history.tolist(),
        "float64_le_sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
        "units": "arbitrary time and output units",
        "relationship_to_input": "a separate auxiliary toy timeline, not predictions computed from the left input trace or its selected time 180",
        "memory_access": {
            "bit_values": [-1, 1],
            "bit_probabilities": [0.5, 0.5],
            "displayed_bit": 1,
            "write_step": OUTPUT_WRITE_STEP,
            "window_length": OUTPUT_WINDOW,
            "first_excluded_step": OUTPUT_WRITE_STEP + OUTPUT_WINDOW,
            "memory_amplitude": OUTPUT_MEMORY_AMPLITUDE,
            "shared_future": "Both histories have identical later inputs and the same known baseline g[k]; the bit is independent of time and later inputs.",
            "visible_before_expiry": "The window contains the bit for k<40, so the ideal prediction matches the selected history.",
            "invisible_after_expiry": "For k>=40 the two histories have identical windows; their common conditional mean is g[k].",
            "post_expiry_pair_gap": 2 * OUTPUT_MEMORY_AMPLITUDE,
            "post_expiry_pair_midpoint_error": OUTPUT_MEMORY_AMPLITUDE,
        },
        "comparison_scope": "The selected +1-history target is shown against the ideal window conditional mean of a balanced two-history toy; the -1 history is recorded but not drawn.",
        "optimality_scope": "After expiry g[k] is both the squared-loss Bayes predictor for the balanced bit and the minimax midpoint of the scalar pair. Other window predictors may fit the shown history but cannot fit both histories exactly.",
        "emulator_scope": "The ideal conditional mean isolates missing input information. It is not a fitted finite-feature QELM, a claim of its attainability by the tested emulator, or new quantum-advantage evidence.",
        "styles": {"target": "ink thin line with dense filled markers", "ideal_window": "green thin line with open markers"},
        "axis_count": 1,
        "axes_side_cm": 4.40,
        "axes_bounds_cm": [13.40, 6.60, 17.80, 11.00],
        "amplitude_limits": [0.3, 1.5],
        "amplitude_ticks": [0.4, 0.8, 1.2],
        "display_headroom": "The output axis reserves room above the traces for its compact legend.",
        "time_limits": [-3, 103],
        "time_ticks": [0, 40, 80],
        "time_tick_labels": ["0", "d", "2d"],
        "horizontal_padding_scope": "Axis-only margins keep endpoint markers off the frame; the 101 toy coordinates remain indexed 0..100.",
    }


def samples() -> np.ndarray:
    return mackey_glass_series(
        N_SAMPLES, rng=np.random.default_rng(SEED), **PARAMETERS,
    )


def tikz_sample_block() -> str:
    values = samples()
    pairs = ",\n".join(f"{k}/{value:.17g}" for k, value in enumerate(values))
    return f"{BEGIN}\n\\def\\MGsamples{{%\n{pairs}\n}}\n{END}"


def provenance(source: Path) -> dict:
    text = source.read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("Figure 1 must contain one reproducible input sample block")
    recorded = text[text.index(BEGIN):text.index(END) + len(END)]
    if recorded != tikz_sample_block():
        raise ValueError("Figure 1 input samples differ from the declared generator")
    if text.count(OUTPUT_BEGIN) != 1 or text.count(OUTPUT_END) != 1:
        raise ValueError("Figure 1 must contain one illustrative output comparison block")
    recorded_outputs = text[text.index(OUTPUT_BEGIN):text.index(OUTPUT_END) + len(OUTPUT_END)]
    if recorded_outputs != tikz_output_sample_block():
        raise ValueError("Figure 1 output coordinates differ from the declared schematic generator")
    values = samples()
    generator = ROOT / "src/qrc_vs_qelm/core/tasks.py"
    return {
        "kind": "simulated illustrative Mackey-Glass input; not evaluation data",
        "generator": str(generator.relative_to(ROOT)),
        "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "preparation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "function": "mackey_glass_series",
        "parameters": PARAMETERS,
        "seed": SEED,
        "segment_selection": "first 201 returned samples after the declared transient; the first 120 values remain unchanged",
        "sample_count": N_SAMPLES,
        "window_length": WINDOW,
        "selected_step": SELECTED_STEP,
        "selected_value": float(values[SELECTED_STEP]),
        "window_start": WINDOW_START,
        "window_end": SELECTED_STEP,
        "window_indices": list(range(WINDOW_START, SELECTED_STEP + 1)),
        "future_context_indices": list(range(SELECTED_STEP + 1, N_SAMPLES)),
        "sample_indices": list(range(N_SAMPLES)),
        "values": values.tolist(),
        "window_values": values[WINDOW_START:SELECTED_STEP + 1].tolist(),
        "future_context_values": values[SELECTED_STEP + 1:].tolist(),
        "float64_le_sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
        "legacy_first_120_float64_le_sha256": hashlib.sha256(values[:120].astype("<f8").tobytes()).hexdigest(),
        "rescaling": "none; one shared input axis preserves the raw amplitude scale",
        "causal_scope": "Only the selected scalar a_180 enters the QRC update; the reset-QELM receives exactly inputs 141..180. "
                        "Grey samples 181..200 are future display context only and are supplied to neither model.",
        "interpolation": "straight segments between samples; no spline or smoothing",
        "plot_geometry": {
            "canvas_cm": [17.9197, 13.0],
            "input_axis_count": 1,
            "sample_axes_side_cm": 4.40,
            "full_input_axes_bounds_cm": [0.92, 6.60, 5.32, 11.0],
            "window_view": "highlighted and bracketed within the shared input axis; no separate duplicate plot",
            "history_error_axes_cm": [6.40, 2.40],
            "output_axis_count": 1,
            "output_axes_side_cm": 4.40,
            "output_axes_bounds_cm": [13.40, 6.60, 17.80, 11.00],
            "amplitude_limits": [0.3, 1.3],
            "amplitude_ticks": [0.4, 0.8, 1.2],
            "full_input_indices": [0, 200],
            "window_indices": [141, 180],
            "full_input_index_limits": [-6, 206],
            "window_bracket_index_extent": [140.5, 180],
        },
        "history_error_curve": {
            "kind": "analytic conceptual exponential; not an input-data fit",
            "local_y_expression": "0.85+2.19*exp(-0.40707656282948684*(x-1.66))",
            "local_domain_cm": [1.66, 7.52],
            "panel_translation_cm": [4.0, 0.0],
            "render_samples": 120,
            "zero_error_canvas_y_cm": 0.85,
            "local_tolerance_crossing_cm": [4.76, 1.47],
            "canvas_tolerance_crossing_cm": [8.76, 1.47],
            "scope": "The analytic curve belongs only to schematic panel b; "
                     "the 201 displayed Mackey-Glass samples and exact 40-sample causal window "
                     "remain connected by straight segments without smoothing.",
        },
        "output_comparisons": output_provenance(),
        "diagram_encoding": {
            "earlier_input": "#000000",
            "carried_state": "#3B86FF",
            "explicit_window": "#00A14B",
            "classical_input_injection": "#A93BFF",
            "measured_quantities_and_readouts": "#FFC943",
            "persistent_gap": "#FC4D30",
        },
        "architecture_scope": "Three wires through a schematic general QRC channel block; "
                              "only the current scalar enters each update. Earlier "
                              "inputs act through carried quantum state. The reset "
                              "QELM receives the full explicit window through E(S_d), "
                              "then fixed gates. Both models measure observables "
                              "and apply a readout. The target readout is fixed "
                              "(identity for the declared observable vector); "
                              "only the emulator readout is trained. No specific "
                              "topology or gate count is asserted.",
        "figure_notation": "Both general architectures show small gold readout triangles after their "
                           "observable banks, without readout text labels. The output labels are Y_t "
                           "and widehat Y_t and feed one analytic toy comparison; readout trainability and the fixed target remain "
                           "specified in the caption and model text. O_1,...,O_m are "
                           "model-specific schematic observable labels, not identical "
                           "operators or an equal-budget claim. The numerical target readout remains "
                           "the identity; actual feature counts and P_j notation are unchanged.",
        "annotation_scope": "No standalone zoom or schematic tags in the artwork. "
                            "One larger shared input plot has an interior selected step and a bracketed causal window; "
                            "the caption identifies the history-error curves, tolerance crossing "
                            "and ideal window prediction as schematic.",
        "scope": "Illustrates current-input versus explicit-window access only; "
                 "no QRC or QELM predictions are generated from this trace.",
    }
