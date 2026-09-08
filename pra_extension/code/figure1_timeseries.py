"""Reproducible Figure 1 input and a separate analytic memory-access toy."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import paper_palette as colors

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
# The left schematic spans less physical time, without smoothing its values.
# Keep the separate output toy's original PARAMETERS and baseline unchanged.
INPUT_PARAMETERS = {**PARAMETERS, "dt": 0.5, "discard": 2000}
BEGIN = "% BEGIN REPRODUCIBLE INPUT SAMPLES"
END = "% END REPRODUCIBLE INPUT SAMPLES"
OUTPUT_BEGIN = "% BEGIN ILLUSTRATIVE OUTPUT COMPARISONS"
OUTPUT_END = "% END ILLUSTRATIVE OUTPUT COMPARISONS"
OUTPUT_N_SAMPLES = 101
OUTPUT_SEED = 2
OUTPUT_WINDOW = 40
OUTPUT_WRITE_STEP = 0
OUTPUT_MEMORY_AMPLITUDE = 0.25
OUTPUT_FADING_TIMESCALE = 30.0
OUTPUT_REVEAL_START = 32
OUTPUT_REVEAL_DURATION = 64
OUTPUT_CASES = ("fading", "persistent")
OUTPUT_AXES_BOUNDS = {
    "fading": [13.525, 8.58, 17.675, 11.78],
    "persistent": [13.525, 4.99, 17.675, 8.19],
}


def output_baseline(case: str = "fading") -> np.ndarray:
    """Return the case's known drawing baseline, independent of the stored bit."""
    raw = mackey_glass_series(
        OUTPUT_N_SAMPLES, rng=np.random.default_rng(OUTPUT_SEED), **PARAMETERS,
    )
    baseline = 0.4 + 0.65 * (raw - np.min(raw)) / (np.max(raw) - np.min(raw))
    if case == "persistent":
        # A fixed common-input schedule reveals the latent bit. Adjust only
        # its bit-independent baseline to preserve the displayed +1 target.
        return baseline + OUTPUT_MEMORY_AMPLITUDE - _output_memory_coefficients(case)
    if case != "fading":
        raise ValueError(f"Unknown Figure 1 output case: {case!r}")
    return baseline


def _output_memory_coefficients(case: str) -> np.ndarray:
    """Return the old bit's physical trace, independently of window access."""
    k = np.arange(OUTPUT_N_SAMPLES, dtype=float)
    if case == "persistent":
        visible = np.clip((k - OUTPUT_REVEAL_START) / OUTPUT_REVEAL_DURATION, 0, 1)
        return OUTPUT_MEMORY_AMPLITUDE * visible**2 * (3 - 2 * visible)
    if case == "fading":
        return OUTPUT_MEMORY_AMPLITUDE * np.exp(-k / OUTPUT_FADING_TIMESCALE)
    raise ValueError(f"Unknown Figure 1 output case: {case!r}")


def output_samples(case: str = "persistent") -> np.ndarray:
    """Return the +1-history target and its ideal window conditional mean.

    The persistent bit is gradually revealed by a common future-control
    schedule independent of window length. The fading trace is unchanged.
    Both predictors lose access to the bit when it leaves their window.
    """
    k = np.arange(OUTPUT_N_SAMPLES, dtype=float)
    coefficients = _output_memory_coefficients(case)
    baseline = output_baseline(case)
    if case == "persistent":
        target = output_baseline() + OUTPUT_MEMORY_AMPLITUDE
        prediction = np.where(k < OUTPUT_WRITE_STEP + OUTPUT_WINDOW, target, baseline)
    else:
        target = baseline + coefficients
        prediction = baseline + coefficients * (k < OUTPUT_WRITE_STEP + OUTPUT_WINDOW)
    return np.column_stack((target, prediction))


def tikz_output_sample_block() -> str:
    blocks = []
    for case in ("persistent", "fading"):
        rows = ",\n".join(
            f"{k}/" + "/".join(f"{value:.17g}" for value in row)
            for k, row in enumerate(output_samples(case))
        )
        blocks.append(f"\\def\\OutputSamples{case.title()}{{%\n{rows}\n}}")
    return OUTPUT_BEGIN + "\n" + "\n".join(blocks) + "\n" + OUTPUT_END


def output_provenance() -> dict:
    baseline = output_baseline()
    expiry = OUTPUT_WRITE_STEP + OUTPUT_WINDOW
    cases = {}
    for case in OUTPUT_CASES:
        values = output_samples(case)
        coefficients = _output_memory_coefficients(case)
        case_baseline = output_baseline(case)
        cases[case] = {
            "title": f"{case.title()} memory",
            "formulas": {
                "memory_coefficient": "c[t]=0.25*exp(-t/30)" if case == "fading" else "c[t]=0.25*s[t]^2*(3-2*s[t]); s[t]=clip((t-32)/64,0,1)",
                "baseline": "g[t]=template[t]" if case == "fading" else "g[t]=template[t]+0.25-c[t]",
                "target": "g[t]+c[t] for the displayed bit b=+1",
                "other_history": "g[t]-c[t] for bit b=-1",
                "prediction": "g[t]+c[t] for t<40; g[t] for t>=40",
            },
            "values": values.tolist(),
            "baseline_values": case_baseline.tolist(),
            "other_history_values": (case_baseline - coefficients).tolist(),
            "memory_coefficients": coefficients.tolist(),
            "float64_le_sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
            "post_expiry_pair_gap_values": (2 * coefficients[expiry:]).tolist(),
            "post_expiry_pair_midpoint_error_values": coefficients[expiry:].tolist(),
            "axes_bounds_cm": OUTPUT_AXES_BOUNDS[case],
            "title_anchor_cm": None,
            "time_tick_labels_visible": case == "persistent",
        }
    cases["fading"]["decay_timescale_steps"] = OUTPUT_FADING_TIMESCALE
    cases["fading"]["decay_scope"] = (
        "The bit coefficient decays from t=0, before and after window expiry. "
        "This is one old input's influence at fixed d, not the uniform D_d limit as d grows."
    )
    cases["persistent"]["preservation_scope"] = "All 101 charcoal target samples equal the original persistent toy byte-for-byte; its bit-independent baseline and ideal prediction now reflect gradual revelation."
    cases["persistent"]["reveal_schedule"] = {
        "start_step": OUTPUT_REVEAL_START,
        "duration_steps": OUTPUT_REVEAL_DURATION,
        "fully_visible_step": OUTPUT_REVEAL_START + OUTPUT_REVEAL_DURATION,
        "scope": "The bit remains stored while common future controls progressively reveal it in the output. "
                 "The schedule does not depend on window length or access expiry, and the coefficient remains 0.25 afterward. "
                 "This is a chosen analytic illustration, not a claim that persistent memory must grow in every reservoir.",
    }
    return {
        "kind": "analytic fading and persistent balanced-bit memory-access toys with ideal window conditional means",
        "status": "schematic information-access example, not simulated, fitted, or measured QRC/QELM performance",
        "generator": "figure1_timeseries.output_samples",
        "sample_count": OUTPUT_N_SAMPLES,
        "sample_indices": list(range(OUTPUT_N_SAMPLES)),
        "column_order": ["target", "prediction"],
        "case_order": list(OUTPUT_CASES),
        "default_case": "persistent",
        "cases": cases,
        "template_stream": {
            "function": "mackey_glass_series",
            "seed": OUTPUT_SEED,
            "parameters": PARAMETERS,
            "generated_count": OUTPUT_N_SAMPLES,
            "indices": [0, OUTPUT_N_SAMPLES - 1],
            "scope": "A separate Mackey-Glass stream supplies a predetermined common baseline shape. "
                     "Its drawing normalization is not fitted preprocessing, future-input access, or a simulated reservoir output.",
        },
        "baseline_formula": "template[t]=0.4+0.65*(raw[t]-min(raw))/(max(raw)-min(raw)); see each case for its bit-independent baseline g[t]",
        "baseline_values": baseline.tolist(),
        "units": "arbitrary time and output units",
        "timeline_scope": "time t advances over 0..100 at the same fixed d=40 in both cases",
        "relationship_to_input": "a separate auxiliary toy timeline, not predictions computed from the left input trace or its selected time 180",
        "memory_access": {
            "bit_values": [-1, 1],
            "bit_probabilities": [0.5, 0.5],
            "displayed_bit": 1,
            "write_step": OUTPUT_WRITE_STEP,
            "window_length": OUTPUT_WINDOW,
            "first_excluded_step": expiry,
            "maximum_memory_amplitude": OUTPUT_MEMORY_AMPLITUDE,
            "shared_future": "Within each case both histories have identical later inputs and the same known baseline g[t]; the bit is independent of time and later inputs. Baselines differ between cases.",
            "visible_before_expiry": "The window contains the bit for t<40, so the ideal prediction matches the selected history.",
            "invisible_after_expiry": "For t>=40 the two histories have identical windows; their common conditional mean is g[t].",
            "expiry_discontinuity": "The ideal predictor loses access to the bit abruptly at t=40; its trace is not smoothed across expiry. The lower case has only a small jump because its stored bit is still weakly visible then.",
        },
        "comparison_scope": "Each case shows the selected +1-history target against the ideal window conditional mean of a balanced two-history toy; the -1 history is recorded but not drawn.",
        "curve_gap_scope": "After expiry each displayed discrepancy equals the midpoint error of that case's specific balanced pair. It is not labeled or identified as the generic supremal D_d/2.",
        "target_definition": "Truth means the target QRC output to be emulated, not an independent third forecasting target. No third trajectory is introduced.",
        "legend_labels": ["QRC", "QELM"],
        "legend_scope": "One shared 10-pt row between the output frames, overlapping their facing borders; "
                        "a white rounded charcoal-border box with a quiet shadow identifies both cases. "
                        "The caption defines QRC as the target and QELM as the ideal window predictor.",
        "legend_geometry": {
            "bounds_cm": [14.00, 8.11, 17.27, 8.64],
            "row_y_cm": 8.375,
            "font_pt": 10,
            "shadow_offset_cm": [0.035, -0.035],
            "style_reference": "user-supplied output reference; position and marks only, not its training/testing or uncertainty claims",
        },
        "prediction_marker": {"inner_sep_pt": 1.45, "every_nth_sample": 2,
                              "target_dot_layer": "drawn after the open ring so agreement stays visible"},
        "optimality_scope": "After expiry g[t] is both the squared-loss Bayes predictor for the balanced bit and the minimax midpoint of the scalar pair. Other window predictors may fit the shown history but cannot fit both histories exactly.",
        "emulator_scope": "The ideal conditional mean isolates missing input information. It is not a fitted finite-feature QELM, a claim of its attainability by the tested emulator, or new quantum-advantage evidence.",
        "styles": {"target": "ink thin line with dense filled markers", "ideal_window": "coral thin line with open markers"},
        "background_access_cue": {
            "before_expiry": {"fill": "green!24", "time_extent": [-3, 40],
                              "meaning": "the old input is still in the explicit window"},
            "after_expiry": {"fill": "blue!16", "time_extent": [40, 103],
                             "meaning": "the old input is outside the window but still affects the QRC"},
            "split_time": expiry,
            "split_x_local_cm": 4.15 * 43 / 106,
            "layer": "behind grid and both pairs of trajectories; full frame height",
            "non_color_cue": "dashed guide and tick d separate the two access regimes",
            "removed_labels": ["output", "old input leaves window", "QRC retains its influence", "Fading memory", "Persistent memory"],
            "scope": "The same old input leaves each fixed window; expiry is not the onset of QRC memory or of fading. "
                     "Caption explains the two access regimes. Hues and tint strengths match the input plot.",
        },
        "axis_count": 2,
        "axes_size_cm": [4.15, 3.2],
        "axes_bounds_cm": OUTPUT_AXES_BOUNDS,
        "amplitude_limits": [0.45, 1.35],
        "amplitude_ticks": [0.6, 0.8, 1.0, 1.2],
        "display_headroom": "Both cases share the same linear y-axis [0.45,1.35]; all target and prediction samples remain inside the frames. "
                            "The entire fading example and the persistent target are unchanged; only the lower ideal prediction changes.",
        "time_limits": [-3, 103],
        "time_ticks": [0, 40, 80],
        "time_tick_labels": ["0", "d", "2d"],
        "time_tick_label_scope": "numeric/symbolic time tick labels appear on the lower persistent-memory frame only",
        "shared_time_label_anchor_cm": None,
        "case_label_style": "no case labels in the artwork; upper and lower cases are explained in the caption",
        "horizontal_padding_scope": "Axis-only margins keep endpoint markers off the frame; the 101 toy coordinates remain indexed 0..100.",
    }


def samples() -> np.ndarray:
    return mackey_glass_series(
        N_SAMPLES, rng=np.random.default_rng(SEED), **INPUT_PARAMETERS,
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
        "parameters": INPUT_PARAMETERS,
        "seed": SEED,
        "segment_selection": "first 201 returned samples after the declared transient; dt=0.5 spans less physical time for a calmer schematic, with the same 1000-unit burn-in",
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
        "legacy_first_120_float64_le_sha256": hashlib.sha256(
            mackey_glass_series(120, rng=np.random.default_rng(SEED), **PARAMETERS)
            .astype("<f8").tobytes()
        ).hexdigest(),
        "legacy_sample_scope": "Checksum of the earlier dt=1 input retained for provenance only; these are not the current displayed samples.",
        "rescaling": "none; one shared input axis preserves the raw amplitude scale",
        "causal_scope": "Only the selected scalar a_180 enters the QRC update; the reset-QELM receives exactly inputs 141..180. "
                        "Grey samples 181..200 are future display context only and are supplied to neither model.",
        "interpolation": "straight segments between samples; no spline or smoothing",
        "plot_grid": {
            "color": colors.GRID,
            "line_width_pt": 0.35,
            "dash_pattern_pt": [2.0, 2.0],
            "directions": ["horizontal", "vertical"],
            "input_vertical_steps": [0, 50, 100, 150, 200],
            "input_horizontal_values": [0.4, 0.8, 1.2],
            "output_vertical_steps": [0, 20, 40, 60, 80, 100],
            "output_horizontal_values": [0.6, 0.8, 1.0, 1.2],
            "scope": "Guides only, above background fills and below input samples, output curves and window-expiry guides. "
                     "The two output frames share the same grid values. Architecture wiring is not gridded.",
        },
        "past_state_cue": {
            "fill": "blue!16",
            "root_color": colors.COBALT,
            "window_fill": "green!24",
            "window_start_boundary": {
                "index": 140.5,
                "style": "green!65!ink, solid, 0.4 pt",
                "scope": "Non-color onset cue; current-time guide remains dashed.",
            },
            "index_extent": [-6, 140.5],
            "data_indices": [0, 140],
            "layer": "behind the grid and charcoal samples",
            "label": "past carried in / QRC state rho_(t-1)",
            "label_center_cm": [2.55, 11.72],
            "scope": "Older inputs act through carried state; no lossless-storage claim. "
                     "Window membership and grey future indices remain unchanged. "
                     "Stronger washes and a solid onset boundary improve separation; "
                     "the output plot uses the same hues and tint strengths.",
        },
        "plot_geometry": {
            "canvas_cm": [17.9197, 9.60],
            "canvas_bounds_cm": [0.0, 3.60, 17.9197, 13.2],
            "input_axis_count": 1,
            "sample_axes_side_cm": 4.15,
            "full_input_axes_bounds_cm": [1.045, 6.725, 5.195, 10.875],
            "window_view": "highlighted and bracketed within the shared input axis; no separate duplicate plot",
            "output_axis_count": 2,
            "output_axes_size_cm": [4.15, 3.2],
            "output_axes_bounds_cm": OUTPUT_AXES_BOUNDS,
            "shared_time_label_anchor_cm": None,
            "removed_axis_titles": ["input step k", "time t"],
            "amplitude_limits": [0.3, 1.3],
            "amplitude_ticks": [0.4, 0.8, 1.2],
            "full_input_indices": [0, 200],
            "window_indices": [141, 180],
            "full_input_index_limits": [-6, 206],
            "window_bracket_index_extent": [140.5, 180],
        },
        "architecture_geometry": {
            "dynamics_shift_right_cm": 0.35,
            "observable_banks_shift_right_cm": 0.15,
            "qrc_channel_bounds_cm": [6.60, 10.15, 7.95, 12.05],
            "reset_encoding_bounds_cm": [6.47, 5.54, 7.53, 7.44],
            "reset_state_ket_anchor_x_cm": 6.35,
            "current_input_node_cm": [5.80, 12.38],
            "current_input_control_vertices_cm": [[7.275, 12.38], [7.275, 12.05]],
            "current_input_route": "orthogonal segments; final vertical arrow down into the QRC top center",
            "preserved": "input axis, architecture coordinates, native fonts, gate sizes and readout triangles; "
                         "the output comparison is now two stacked frames and the former lower history-error panel is removed",
        },
        "removed_history_error_panel": {
            "status": "removed from the artwork",
            "replacement": "two stacked fading/persistent output comparisons at one fixed window length",
            "theory_scope": "The manuscript retains D_d/2 as the scalar worst-case missing-history optimum. "
                            "It is not a label for the gap between generic QRC and QELM curves.",
        },
        "output_comparisons": output_provenance(),
        "diagram_encoding": dict(colors.DIAGRAM),
        "palette_name": "Cobalt",
        "legacy_tikz_color_tokens": dict(colors.TIKZ),
        "architecture_scope": "Three wires through a schematic general QRC channel block; "
                              "only the current scalar enters each update. Earlier "
                              "inputs act through carried quantum state. The reset "
                              "QELM receives the full explicit window through E(S_d), "
                              "then fixed gates. Both models measure observables "
                              "and apply a readout. The target readout is fixed "
                              "(identity for the declared observable vector); "
                              "only the emulator readout is trained. No specific "
                              "topology or gate count is asserted.",
        "observable_readout_style": {
            "readout_corner_radius_pt": 1.8,
            "readout_vertex_span_cm": [0.38, 0.50],
            "black_connector_corner_radius_pt": 2.0,
            "feature_card_corner_radius_pt": 2.0,
            "omitted_observable_lanes_per_bank": 3,
            "omitted_lane_span_cm": [10.80, 11.04],
            "omitted_lane_style": "muted!70, 0.55 pt, 1 pt rounded bends",
            "scope": "Faint lanes represent omitted intermediate observables behind the ellipsis; "
                     "they start at the bank's output edge, not at the quantum circuit. "
                     "Their illustrative count is not a feature budget or added quantum wires.",
        },
        "figure_notation": "Both general architectures show small yellow readout triangles with rounded corners after their "
                           "observable banks, without readout text labels. The output labels are y_t "
                           "and widehat y_t and feed two analytic memory-access comparisons; readout trainability and the fixed target remain "
                           "specified in the caption and model text. O_1,...,O_m label target observables; "
                           "P_1,...,P_B label measured QELM features, including checkpoints. These are not identical "
                           "operators or an equal-budget claim. The numerical target readout remains "
                           "the identity; m and B denote output dimension and measured-feature count.",
        "annotation_scope": "No standalone zoom or schematic tags in the artwork. "
                            "The initial-state kets and fixed-gate blocks remain, without fresh state or fixed gates text labels. "
                            "One larger shared input plot has an interior selected step and a bracketed causal window; "
                            "the two output frames have no internal case labels and share one QRC/QELM legend between them. "
                            "There are no panel letters, epsilon or D_d/2 artwork labels; the caption identifies the ideal window predictions as schematic.",
        "scope": "Illustrates current-input versus explicit-window access only; "
                 "no QRC or QELM predictions are generated from this trace.",
    }
