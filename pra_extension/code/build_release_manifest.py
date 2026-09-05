#!/usr/bin/env python3
"""Build a deterministic, fail-closed manifest for the PRA revision package.

The manifest deliberately keeps two evidence layers separate:

* the previously frozen canonical confirmatory material identity; and
* the post-confirmatory exploratory extension artifacts.

No timestamp, hostname, or absolute path is recorded, so identical inputs produce
identical manifest content on different machines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_FINAL_FIGURES_DIR = "pra_extension/figures_pra_final_v6"
REQUIRED_FINAL_FIGURES = {
    "fig06_observable_memory.pdf",
    "fig06_observable_memory.png",
    "fig07_finite_shot_frontier.pdf",
    "fig07_finite_shot_frontier.png",
    "fig08_spectral_visibility.pdf",
    "fig08_spectral_visibility.png",
    "fig09_continuous_robustness.pdf",
    "fig09_continuous_robustness.png",
    "fig10_n4_shot_robustness.pdf",
    "fig10_n4_shot_robustness.png",
    "fig11_student_architecture_robustness.pdf",
    "fig11_student_architecture_robustness.png",
    "fig12_noisy_training_frontier.pdf",
    "fig12_noisy_training_frontier.png",
}

CANONICAL_PROTOCOL_FILES = (
    "CANONICAL_EMULATION_LITERATURE_POSITIONING.md",
    "CANONICAL_EMULATION_PREREGISTRATION.md",
    "CANONICAL_EMULATION_THEORY.md",
    "EMULATION_MATCHED_PROTOCOL.md",
    "PREREGISTRATION.md",
    "PROTOCOL.md",
)
CANONICAL_SCRIPTS = (
    "scripts/evaluate_canonical_gates.py",
    "scripts/render_canonical_emulation_figures.py",
)

# These two non-executable notes are living post-confirmatory interpretations.
# Their pre-run hashes remain sealed inside canonical_manifest.json; revisions
# are permitted only when they are disclosed as documentation drift.  Frozen
# configurations, preregistration, code, and tests must still match byte-for-byte.
POST_FREEZE_MUTABLE_DOCUMENTATION = frozenset(
    {
        "protocol:CANONICAL_EMULATION_LITERATURE_POSITIONING.md",
        "protocol:CANONICAL_EMULATION_THEORY.md",
    }
)

CANONICAL_RESULT_FILES: dict[str, str] = {
    f"results_canonical_confirmatory_v1/{name}": "frozen_canonical_result"
    for name in (
        "canonical_block_contrasts.csv",
        "canonical_gate_report.json",
        "canonical_manifest.json",
        "canonical_summary.json",
        "computational_resources.json",
        "stream_metrics.csv",
        "suffix_decomposition.csv",
        "system_diagnostics.csv",
    )
}
CANONICAL_FIGURE_OUTPUTS = tuple(
    f"fig{index:02d}_{stem}{extension}"
    for index, stem in (
        (1, "history_frontier"),
        (2, "qelm_error_surface"),
        (3, "exact_mse_decomposition"),
        (4, "regime_map"),
        (5, "matched_controls"),
        (6, "predicted_vs_observed_window"),
        (7, "dissipation_window_scaling"),
    )
    for extension in (".pdf", ".png", "_source.csv")
)
CANONICAL_FIGURE_FILES: dict[str, str] = {
    (
        f"paper/figures/canonical_emulation/{name}"
    ): "canonical_paper_figure_artifact"
    for name in CANONICAL_FIGURE_OUTPUTS
}
CANONICAL_FIGURE_FILES[
    "paper/figures/canonical_emulation/figure_manifest.json"
] = "canonical_figure_hash_ledger"

DENSE_SECONDARY_FILES: dict[str, str] = {
    "DENSE_PHASE_SECONDARY_PROTOCOL.md": "frozen_dense_secondary_protocol",
    "configs/canonical_emulation_dense_secondary.yaml": (
        "frozen_dense_secondary_configuration"
    ),
    "configs/canonical_emulation_dense_secondary_freeze.json": (
        "dense_secondary_freeze_record"
    ),
    "scripts/dense_phase_secondary_common.py": "dense_secondary_audit_code",
    "scripts/run_dense_phase_secondary.py": "dense_secondary_runner",
    "scripts/analyze_dense_phase_secondary.py": "dense_secondary_analysis_code",
    "results_dense_phase_secondary_v1/canonical_manifest.json": (
        "dense_secondary_run_manifest"
    ),
    "results_dense_phase_secondary_v1/canonical_summary.json": (
        "dense_secondary_run_summary"
    ),
    "results_dense_phase_secondary_v1/computational_resources.json": (
        "dense_secondary_resource_record"
    ),
    "results_dense_phase_secondary_v1/dense_secondary_preflight_audit.json": (
        "dense_secondary_preflight_audit"
    ),
    "results_dense_phase_secondary_v1/stream_metrics.csv": (
        "dense_secondary_stream_metrics"
    ),
    "results_dense_phase_secondary_v1/suffix_decomposition.csv": (
        "dense_secondary_suffix_decomposition"
    ),
    "results_dense_phase_secondary_v1/system_diagnostics.csv": (
        "dense_secondary_system_diagnostics"
    ),
    "results_dense_phase_secondary_v1_analysis/dense_phase_cells.csv": (
        "dense_secondary_figure_source"
    ),
    "results_dense_phase_secondary_v1_analysis/dense_phase_root_scaling.csv": (
        "dense_secondary_root_scaling"
    ),
    "results_dense_phase_secondary_v1_analysis/dense_phase_summary.json": (
        "dense_secondary_decision_summary"
    ),
    "results_dense_phase_secondary_v1_analysis/dense_phase_analysis_manifest.json": (
        "dense_secondary_analysis_hash_ledger"
    ),
}

DENSE_N4_FILES: dict[str, str] = {
    "DENSE_PHASE_N4_PROTOCOL.md": "frozen_dense_n4_protocol",
    "configs/canonical_emulation_dense_n4.yaml": "frozen_dense_n4_configuration",
    "configs/canonical_emulation_dense_n4_freeze.json": "dense_n4_freeze_record",
    "scripts/dense_phase_n4_common.py": "dense_n4_audit_code",
    "scripts/run_dense_phase_n4.py": "dense_n4_runner",
    "scripts/analyze_dense_phase_n4.py": "dense_n4_analysis_code",
    "results_dense_phase_n4_v1/canonical_manifest.json": "dense_n4_run_manifest",
    "results_dense_phase_n4_v1/canonical_summary.json": "dense_n4_run_summary",
    "results_dense_phase_n4_v1/computational_resources.json": (
        "dense_n4_resource_record"
    ),
    "results_dense_phase_n4_v1/dense_n4_preflight_audit.json": (
        "dense_n4_preflight_audit"
    ),
    "results_dense_phase_n4_v1/stream_metrics.csv": "dense_n4_stream_metrics",
    "results_dense_phase_n4_v1/suffix_decomposition.csv": (
        "dense_n4_suffix_decomposition"
    ),
    "results_dense_phase_n4_v1/system_diagnostics.csv": (
        "dense_n4_system_diagnostics"
    ),
    "results_dense_phase_n4_v1_analysis/dense_n4_phase_cells.csv": (
        "dense_n4_figure_source"
    ),
    "results_dense_phase_n4_v1_analysis/dense_n4_root_scaling.csv": (
        "dense_n4_root_scaling"
    ),
    "results_dense_phase_n4_v1_analysis/dense_n4_summary.json": (
        "dense_n4_decision_summary"
    ),
    "results_dense_phase_n4_v1_analysis/dense_n4_analysis_manifest.json": (
        "dense_n4_analysis_hash_ledger"
    ),
}

# These are the authoritative source tables and machine-readable summaries used
# by the revised paper.  Earlier pilot, smoke, v1, and v2 directories are
# intentionally excluded.
AUTHORITATIVE_EXTENSION_FILES: dict[str, str] = {
    (
        "pra_extension/outputs/finite_k_history_calibration_v1_strict/"
        "finite_k_history_calibration.json"
    ): "strict_finite_k_history_calibration_sensitivity",
    (
        "pra_extension/outputs/observable_memory_full_exploratory_v3_"
        "rigorous_12roots/observable_memory_results.csv"
    ): "raw_observable_memory_table",
    (
        "pra_extension/outputs/observable_memory_full_exploratory_v3_"
        "rigorous_12roots/observable_memory_summary.json"
    ): "observable_memory_run_summary",
    (
        "pra_extension/outputs/finite_shots_full_exploratory_v4b_"
        "refit_design_12roots/finite_shot_results.csv"
    ): "raw_finite_shot_table_n2_n3",
    (
        "pra_extension/outputs/finite_shots_full_exploratory_v4b_"
        "refit_design_12roots/finite_shot_summary.json"
    ): "finite_shot_run_summary_n2_n3",
    (
        "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
        "pra_extension_analysis.json"
    ): "strict_analysis_observable_and_shots_n2_n3",
    (
        "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
        "artifact_sha256.json"
    ): "observable_and_shots_analysis_hash_ledger",
    (
        "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
        "finite_shot_root_effects.csv"
    ): "root_block_effects_finite_shots_n2_n3",
    (
        "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
        "observable_root_effects.csv"
    ): "root_block_effects_observable_memory",
    (
        "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
        "observable_visibility_rows.csv"
    ): "observable_visibility_diagnostic_rows",
    (
        "pra_extension/outputs/channel_spectrum_full_exploratory_v1_12roots/"
        "channel_spectrum.csv"
    ): "raw_representative_channel_spectrum_table",
    (
        "pra_extension/outputs/channel_spectrum_full_exploratory_v1_12roots/"
        "channel_spectrum_summary.json"
    ): "strict_analysis_representative_channel_spectrum",
    (
        "pra_extension/outputs/observable_memory_continuous_exploratory_v1_"
        "12roots/observable_memory_results.csv"
    ): "raw_continuous_input_observable_memory_table",
    (
        "pra_extension/outputs/observable_memory_continuous_exploratory_v1_"
        "12roots/observable_memory_summary.json"
    ): "continuous_input_run_summary",
    (
        "pra_extension/outputs/continuous_input_analysis_v1_strict/"
        "continuous_analysis.json"
    ): "strict_analysis_continuous_inputs",
    (
        "pra_extension/outputs/continuous_input_analysis_v1_strict/"
        "continuous_root_effects.csv"
    ): "root_block_effects_continuous_inputs",
    (
        "pra_extension/outputs/finite_shots_n4_exploratory_v2b_refit_design_12roots/"
        "finite_shot_results.csv"
    ): "raw_finite_shot_table_n4",
    (
        "pra_extension/outputs/finite_shots_n4_exploratory_v2b_refit_design_12roots/"
        "finite_shot_summary.json"
    ): "finite_shot_run_summary_n4",
    (
        "pra_extension/outputs/finite_shots_n4_analysis_v2_refit_design_strict/"
        "n4_shot_analysis.json"
    ): "strict_analysis_finite_shots_n4",
    (
        "pra_extension/outputs/finite_shots_eiv_n4_exploratory_v2_final_helpers_"
        "12roots/eiv_finite_shot_results.csv"
    ): "raw_noisy_feature_table_n4",
    (
        "pra_extension/outputs/finite_shots_eiv_n4_exploratory_v2_final_helpers_"
        "12roots/eiv_finite_shot_summary.json"
    ): "noisy_feature_run_summary_n4",
    (
        "pra_extension/outputs/finite_shots_eiv_n4_analysis_v6_final_helpers_"
        "strict/eiv_analysis.json"
    ): "strict_analysis_noisy_features_n4",
    (
        "pra_extension/outputs/finite_shots_eiv_n4_analysis_v6_final_helpers_"
        "strict/eiv_root_effects.csv"
    ): "root_block_effects_noisy_features_n4",
    (
        "pra_extension/outputs/finite_shots_eiv_n4_analysis_v6_final_helpers_"
        "strict/eiv_validation_selected_rows.csv"
    ): "validation_selected_rows_noisy_features_n4",
    (
        "pra_extension/outputs/student_architecture_full_exploratory_v1_12roots/"
        "student_architecture_results.csv"
    ): "raw_student_architecture_table",
    (
        "pra_extension/outputs/student_architecture_full_exploratory_v1_12roots/"
        "student_architecture_summary.json"
    ): "student_architecture_run_summary",
    (
        "pra_extension/outputs/student_architecture_analysis_v1_strict/"
        "student_architecture_analysis.json"
    ): "strict_analysis_student_architectures",
    (
        "pra_extension/outputs/student_architecture_analysis_v1_strict/"
        "student_architecture_selected_effects.csv"
    ): "validation_selected_effects_student_architectures",
    (
        "pra_extension/outputs/student_architecture_analysis_v1_strict/"
        "artifact_sha256.json"
    ): "student_architecture_analysis_hash_ledger",
    (
        "pra_extension/outputs/bernstein_qelm_validation_v1/"
        "bernstein_convergence.csv"
    ): "constructive_bernstein_convergence_table",
    (
        "pra_extension/outputs/bernstein_qelm_validation_v1/"
        "qrc_history_convergence.csv"
    ): "constructive_qrc_history_convergence_table",
    (
        "pra_extension/outputs/bernstein_qelm_validation_v1/"
        "bernstein_validation_summary.json"
    ): "constructive_qrc_to_qelm_validation_summary",
    (
        "pra_extension/outputs/bernstein_qelm_validation_v1/"
        "artifact_sha256.json"
    ): "constructive_qrc_to_qelm_hash_ledger",
}

DOCUMENTATION_FILES: dict[str, str] = {
    "README.md": "repository_reproduction_guide",
    "PRA_REVISION_REPORT.md": "referee_response_and_claim_boundary",
    "CANONICAL_EMULATION_LITERATURE_POSITIONING.md": (
        "post_confirmatory_literature_positioning"
    ),
    "CANONICAL_EMULATION_PREREGISTRATION.md": "frozen_canonical_preregistration",
    "CANONICAL_EMULATION_THEORY.md": "post_confirmatory_theory_exposition",
    "EMULATION_MATCHED_PROTOCOL.md": "frozen_matched_emulation_protocol",
    "PREREGISTRATION.md": "frozen_project_preregistration",
    "PROTOCOL.md": "frozen_project_protocol",
    "pra_extension/PROTOCOL.md": "extension_protocol_record",
    "pra_extension/theory_and_literature.md": "theory_and_prior_art_audit",
    (
        "paper/figures/information_interface/CHART_MAP.md"
    ): "main_figure_question_takeaway_map",
}

REQUIRED_PAPER_SOURCES = (
    "paper/canonical_emulation/main.tex",
    "paper/canonical_emulation/references.bib",
    "paper/canonical_emulation/sections/abstract.tex",
    "paper/canonical_emulation/sections/introduction.tex",
    "paper/canonical_emulation/sections/related-work.tex",
    "paper/canonical_emulation/sections/background.tex",
    "paper/canonical_emulation/sections/evaluation.tex",
    "paper/canonical_emulation/sections/conclusion.tex",
    "paper/canonical_emulation/sections/methodology.tex",
    "paper/canonical_emulation/sections/appendix.tex",
)

REQUIRED_PAPER_ARTIFACTS: dict[str, str] = {}

DARK_SECTOR_FILES: dict[str, str] = {
    "configs/dark_sector_validation.json": "frozen_dark_sector_configuration",
    "configs/DARK_SECTOR_VALIDATION_PROTOCOL.md": "frozen_dark_sector_protocol",
    (
        "pra_extension/outputs/dark_sector_validation/"
        "dark_sector_trajectories.csv"
    ): "dark_sector_trajectory_table",
    (
        "pra_extension/outputs/dark_sector_validation/"
        "dark_sector_audits.csv"
    ): "dark_sector_algebra_audit_table",
    (
        "pra_extension/outputs/dark_sector_validation/"
        "dark_sector_leakage.csv"
    ): "dark_sector_falsification_control_table",
    (
        "pra_extension/outputs/dark_sector_validation/"
        "dark_sector_summary.json"
    ): "dark_sector_validation_summary",
}

COMPACT_PRA_FIGURE_FILES: dict[str, str] = {
    (
        "paper/figures/compact_pra/fig01_protected_excited_visible.pdf"
    ): "protected_memory_figure_pdf",
    (
        "paper/figures/compact_pra/fig01_protected_excited_visible.png"
    ): "protected_memory_figure_png",
    (
        "paper/figures/compact_pra/fig01_protected_excited_visible_source.csv"
    ): "protected_memory_figure_source",
    (
        "paper/figures/compact_pra/fig01_protected_excited_visible_leakage_source.csv"
    ): "protected_memory_leakage_figure_source",
    (
        "paper/figures/compact_pra/fig02_phase_map_compact.pdf"
    ): "compact_phase_map_figure_pdf",
    (
        "paper/figures/compact_pra/fig02_phase_map_compact.png"
    ): "compact_phase_map_figure_png",
    (
        "paper/figures/compact_pra/fig02_phase_map_compact_source.csv"
    ): "compact_phase_map_figure_source",
    (
        "paper/figures/compact_pra/fig03_dissipation_window_scaling_compact.pdf"
    ): "compact_dissipation_window_scaling_pdf",
    (
        "paper/figures/compact_pra/fig03_dissipation_window_scaling_compact.png"
    ): "compact_dissipation_window_scaling_png",
    (
        "paper/figures/compact_pra/fig03_dissipation_window_scaling_compact_source.csv"
    ): "compact_dissipation_window_scaling_source",
    (
        "paper/figures/compact_pra/dark_sector_figure_sha256.json"
    ): "protected_memory_figure_hash_ledger",
    (
        "paper/figures/information_interface/fig01_information_interface.tex"
    ): "information_interface_tikz_source",
    (
        "paper/figures/information_interface/fig01_information_interface.pdf"
    ): "information_interface_figure_pdf",
    (
        "paper/figures/information_interface/fig01_information_interface.png"
    ): "information_interface_figure_png",
    (
        "paper/figures/information_interface/fig01_information_interface_source.json"
    ): "information_interface_figure_source",
    (
        "paper/figures/information_interface/fig02_information_frontier.pdf"
    ): "information_frontier_figure_pdf",
    (
        "paper/figures/information_interface/fig02_information_frontier.png"
    ): "information_frontier_figure_png",
    (
        "paper/figures/information_interface/fig02_information_frontier_source.csv"
    ): "information_frontier_figure_source",
    (
        "paper/figures/information_interface/fig03_resource_diagnosis.pdf"
    ): "resource_diagnosis_figure_pdf",
    (
        "paper/figures/information_interface/fig03_resource_diagnosis.png"
    ): "resource_diagnosis_figure_png",
    (
        "paper/figures/information_interface/fig03_resource_diagnosis_source.csv"
    ): "resource_diagnosis_figure_source",
    (
        "paper/figures/information_interface/figure_sha256.json"
    ): "information_interface_figure_hash_ledger",
    (
        "paper/figures/information_interface/phase_threshold_sensitivity.json"
    ): "post_hoc_phase_threshold_sensitivity",
}

THRESHOLD_SENSITIVITY_FILES: dict[str, str] = {
    "scripts/analyze_phase_threshold_sensitivity.py": (
        "post_hoc_phase_threshold_analysis_code"
    ),
}

# These unchanged historical artifacts support schema, mutation-rejection,
# and renderer regression tests only. They are not current paper evidence.
HISTORICAL_REGRESSION_FILES: dict[str, str] = {
    (
        "pra_extension/outputs/observable_memory_full_exploratory_v2_12roots/"
        "observable_memory_results.csv"
    ): "historical_regression_fixture_only",
    (
        "pra_extension/outputs/finite_shots_full_exploratory_v2_12roots/"
        "finite_shot_results.csv"
    ): "historical_regression_fixture_only",
    (
        "pra_extension/outputs/finite_shots_n4_exploratory_v1_12roots/"
        "finite_shot_results.csv"
    ): "historical_regression_fixture_only",
    (
        "results_recurrence_fingerprint_heldout/"
        "fingerprint_predictions_unblinded.csv"
    ): "historical_regression_fixture_only",
    (
        "results_recurrence_fingerprint_heldout/task_link_scores.csv"
    ): "historical_regression_fixture_only",
    (
        "results_recurrence_fingerprint_heldout/task_link_systems.csv"
    ): "historical_regression_fixture_only",
    (
        "results_recurrence_fingerprint_heldout/recurrence_fingerprint_study.json"
    ): "historical_regression_fixture_only",
}

# Verified against each historical run's existing artifact_sha256 ledger.
PINNED_HISTORICAL_REGRESSION_HASHES = {
    (
        "pra_extension/outputs/observable_memory_full_exploratory_v2_12roots/"
        "observable_memory_results.csv"
    ): "f0f89e26e75fa033d53de66afd68ce7fd09955bc16f351fca4e128e0868029a5",
    (
        "pra_extension/outputs/finite_shots_full_exploratory_v2_12roots/"
        "finite_shot_results.csv"
    ): "c32b1603945fa6a8639c14c2b556052b0529613f278e1a02f95d08d4db99d60d",
    (
        "pra_extension/outputs/finite_shots_n4_exploratory_v1_12roots/"
        "finite_shot_results.csv"
    ): "8853e7f4def056e62de1c505794eab1b7496f8518ac8837764e5cf0371e1cc10",
    (
        "results_recurrence_fingerprint_heldout/"
        "fingerprint_predictions_unblinded.csv"
    ): "51942d1f67eec7c626785ce591c2a13199e7d72214e27944321ee5c163e2710c",
    (
        "results_recurrence_fingerprint_heldout/task_link_scores.csv"
    ): "696c28723ecf0893182092c69879e95ccda70ebe07dd95bbace83e4c9ec5b281",
    (
        "results_recurrence_fingerprint_heldout/task_link_systems.csv"
    ): "24462981a9ba3c13388699dbf2defe8b9b42ba6585467209e70ac3119bbc5ebe",
    (
        "results_recurrence_fingerprint_heldout/recurrence_fingerprint_study.json"
    ): "67afb800c2996950d34ebb2b7741678826c4f650424cdb12191157f85892f1e8",
}

# These audited post-confirmatory artifacts are pinned so superseded shot
# protocols or stale analyses cannot silently enter the release.
PINNED_AUTHORITATIVE_HASHES = {
    (
        "pra_extension/outputs/finite_k_history_calibration_v1_strict/"
        "finite_k_history_calibration.json"
    ): "d6987822c249fe3ea61adb65161771a5097af4efee41f2b96f1020d46664e8e5",
    (
        "pra_extension/outputs/finite_shots_full_exploratory_v4b_"
        "refit_design_12roots/finite_shot_results.csv"
    ): "6554574812348bb0a83efbe67dfba22bb51dce8e6155332ca2ca81a1db1170a2",
    (
        "pra_extension/outputs/finite_shots_n4_exploratory_v2b_refit_design_12roots/"
        "finite_shot_results.csv"
    ): "5be7496382a3b36ce3d02685caa1436fce0b5d9a00316c1d166dd620310e5d9e",
    (
        "pra_extension/outputs/finite_shots_n4_exploratory_v2b_refit_design_12roots/"
        "finite_shot_summary.json"
    ): "72ed69dea3da5044c8b07e362da5834936887e902c4e2fc6973fd6dea1608aef",
    (
        "pra_extension/outputs/finite_shots_n4_analysis_v2_refit_design_strict/"
        "n4_shot_analysis.json"
    ): "d50d47d52dd248fa124cdad1d39c217c008f1096e22781db258b4cc973014e5a",
    (
        "pra_extension/outputs/finite_shots_eiv_n4_exploratory_v2_final_helpers_"
        "12roots/eiv_finite_shot_results.csv"
    ): "7a3d174d5b6aeedfc76283f9254c69a4729b4fb0e66c548eb82a6c01cf244e32",
    (
        "pra_extension/outputs/finite_shots_eiv_n4_analysis_v6_final_helpers_"
        "strict/eiv_analysis.json"
    ): "c663b22b7568ae69c5b41e6a90a2d83832018a466f24198ec6862532974736bf",
    (
        "pra_extension/outputs/student_architecture_full_exploratory_v1_12roots/"
        "student_architecture_results.csv"
    ): "c7b52b3d144c234c2b4da52bd72265178cf08bebd811346c71cc7b0cfffc9742",
    (
        "pra_extension/outputs/student_architecture_analysis_v1_strict/"
        "student_architecture_analysis.json"
    ): "f1becacd8a8ffa4ce94794110c59d81fcf2e6c3d65d8b26233afb103dec1c14d",
}

EXPLORATORY_ANALYSES = (
    "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
    "pra_extension_analysis.json",
    "pra_extension/outputs/channel_spectrum_full_exploratory_v1_12roots/"
    "channel_spectrum_summary.json",
    "pra_extension/outputs/continuous_input_analysis_v1_strict/"
    "continuous_analysis.json",
    "pra_extension/outputs/finite_shots_n4_analysis_v2_refit_design_strict/"
    "n4_shot_analysis.json",
    "pra_extension/outputs/finite_shots_eiv_n4_analysis_v6_final_helpers_"
    "strict/eiv_analysis.json",
    "pra_extension/outputs/student_architecture_analysis_v1_strict/"
    "student_architecture_analysis.json",
)

# (JSON file, object key containing recorded hashes, recorded key, source file)
SOURCE_HASH_LINKS = (
    (
        "paper/figures/information_interface/fig01_information_interface_source.json",
        "",
        "tikz_source_sha256",
        "paper/figures/information_interface/fig01_information_interface.tex",
    ),
    *tuple(
        (
            "pra_extension/outputs/dark_sector_validation/dark_sector_summary.json",
            "artifact_sha256",
            name,
            f"pra_extension/outputs/dark_sector_validation/{name}",
        )
        for name in (
            "dark_sector_audits.csv",
            "dark_sector_leakage.csv",
            "dark_sector_trajectories.csv",
        )
    ),
    (
        "pra_extension/outputs/dark_sector_validation/dark_sector_summary.json",
        "",
        "config_sha256",
        "configs/dark_sector_validation.json",
    ),
    *tuple(
        (
            "paper/figures/compact_pra/dark_sector_figure_sha256.json",
            "",
            name,
            f"paper/figures/compact_pra/{name}",
        )
        for name in (
            "fig01_protected_excited_visible.pdf",
            "fig01_protected_excited_visible.png",
            "fig01_protected_excited_visible_source.csv",
            "fig01_protected_excited_visible_leakage_source.csv",
            "fig02_phase_map_compact.pdf",
            "fig02_phase_map_compact.png",
            "fig02_phase_map_compact_source.csv",
            "fig03_dissipation_window_scaling_compact.pdf",
            "fig03_dissipation_window_scaling_compact.png",
            "fig03_dissipation_window_scaling_compact_source.csv",
        )
    ),
    *tuple(
        (
            "paper/figures/information_interface/figure_sha256.json",
            "",
            name,
            f"paper/figures/information_interface/{name}",
        )
        for name in (
            "fig01_information_interface.pdf",
            "fig01_information_interface.png",
            "fig01_information_interface_source.json",
            "fig02_information_frontier.pdf",
            "fig02_information_frontier.png",
            "fig02_information_frontier_source.csv",
            "fig03_resource_diagnosis.pdf",
            "fig03_resource_diagnosis.png",
            "fig03_resource_diagnosis_source.csv",
        )
    ),
    (
        EXPLORATORY_ANALYSES[0],
        "source_artifacts",
        (
            "pra_extension/outputs/observable_memory_full_exploratory_v3_"
            "rigorous_12roots/observable_memory_results.csv"
        ),
        (
            "pra_extension/outputs/observable_memory_full_exploratory_v3_"
            "rigorous_12roots/observable_memory_results.csv"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[0],
        "source_artifacts",
        (
            "pra_extension/outputs/finite_shots_full_exploratory_v4b_"
            "refit_design_12roots/finite_shot_results.csv"
        ),
        (
            "pra_extension/outputs/finite_shots_full_exploratory_v4b_"
            "refit_design_12roots/finite_shot_results.csv"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[1],
        "artifacts",
        "channel_spectrum.csv",
        (
            "pra_extension/outputs/channel_spectrum_full_exploratory_v1_"
            "12roots/channel_spectrum.csv"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[2],
        "source_artifacts",
        (
            "pra_extension/outputs/observable_memory_continuous_exploratory_"
            "v1_12roots/observable_memory_results.csv"
        ),
        (
            "pra_extension/outputs/observable_memory_continuous_exploratory_"
            "v1_12roots/observable_memory_results.csv"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[2],
        "source_artifacts",
        (
            "pra_extension/outputs/observable_memory_continuous_exploratory_"
            "v1_12roots/observable_memory_summary.json"
        ),
        (
            "pra_extension/outputs/observable_memory_continuous_exploratory_"
            "v1_12roots/observable_memory_summary.json"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[2],
        "artifacts",
        "continuous_root_effects.csv",
        (
            "pra_extension/outputs/continuous_input_analysis_v1_strict/"
            "continuous_root_effects.csv"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[3],
        "source_artifacts",
        (
            "pra_extension/outputs/finite_shots_n4_exploratory_v2b_refit_design_12roots/"
            "finite_shot_results.csv"
        ),
        (
            "pra_extension/outputs/finite_shots_n4_exploratory_v2b_refit_design_12roots/"
            "finite_shot_results.csv"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[3],
        "source_artifacts",
        (
            "pra_extension/outputs/finite_shots_n4_exploratory_v2b_refit_design_12roots/"
            "finite_shot_summary.json"
        ),
        (
            "pra_extension/outputs/finite_shots_n4_exploratory_v2b_refit_design_12roots/"
            "finite_shot_summary.json"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[4],
        "source_artifacts",
        (
            "pra_extension/outputs/finite_shots_eiv_n4_exploratory_v2_"
            "final_helpers_12roots/eiv_finite_shot_results.csv"
        ),
        (
            "pra_extension/outputs/finite_shots_eiv_n4_exploratory_v2_"
            "final_helpers_12roots/eiv_finite_shot_results.csv"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[4],
        "source_artifacts",
        (
            "pra_extension/outputs/finite_shots_eiv_n4_exploratory_v2_"
            "final_helpers_12roots/eiv_finite_shot_summary.json"
        ),
        (
            "pra_extension/outputs/finite_shots_eiv_n4_exploratory_v2_"
            "final_helpers_12roots/eiv_finite_shot_summary.json"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[4],
        "artifacts",
        "eiv_root_effects.csv",
        (
            "pra_extension/outputs/finite_shots_eiv_n4_analysis_v6_final_helpers_"
            "strict/eiv_root_effects.csv"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[4],
        "artifacts",
        "eiv_validation_selected_rows.csv",
        (
            "pra_extension/outputs/finite_shots_eiv_n4_analysis_v6_final_helpers_"
            "strict/eiv_validation_selected_rows.csv"
        ),
    ),
    (
        EXPLORATORY_ANALYSES[5],
        "",
        "source_results_sha256",
        (
            "pra_extension/outputs/student_architecture_full_exploratory_v1_12roots/"
            "student_architecture_results.csv"
        ),
    ),
    (
        "pra_extension/outputs/student_architecture_analysis_v1_strict/"
        "artifact_sha256.json",
        "",
        "student_architecture_analysis.json",
        (
            "pra_extension/outputs/student_architecture_analysis_v1_strict/"
            "student_architecture_analysis.json"
        ),
    ),
    (
        "pra_extension/outputs/student_architecture_analysis_v1_strict/"
        "artifact_sha256.json",
        "",
        "student_architecture_selected_effects.csv",
        (
            "pra_extension/outputs/student_architecture_analysis_v1_strict/"
            "student_architecture_selected_effects.csv"
        ),
    ),
    (
        "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
        "artifact_sha256.json",
        "",
        "pra_extension_analysis.json",
        (
            "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
            "pra_extension_analysis.json"
        ),
    ),
    *tuple(
        (
            "pra_extension/outputs/bernstein_qelm_validation_v1/"
            "artifact_sha256.json",
            "",
            name,
            f"pra_extension/outputs/bernstein_qelm_validation_v1/{name}",
        )
        for name in (
            "bernstein_convergence.csv",
            "qrc_history_convergence.csv",
            "bernstein_validation_summary.json",
        )
    ),
    (
        "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
        "artifact_sha256.json",
        "",
        "finite_shot_root_effects.csv",
        (
            "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
            "finite_shot_root_effects.csv"
        ),
    ),
    (
        "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
        "artifact_sha256.json",
        "",
        "observable_root_effects.csv",
        (
            "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
            "observable_root_effects.csv"
        ),
    ),
    (
        "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
        "artifact_sha256.json",
        "",
        "observable_visibility_rows.csv",
        (
            "pra_extension/outputs/pra_extension_analysis_v5_refit_common_support/"
            "observable_visibility_rows.csv"
        ),
    ),
    *tuple(
        (
            "paper/figures/canonical_emulation/figure_manifest.json",
            "input_artifacts",
            Path(relative).name,
            relative,
        )
        for relative in CANONICAL_RESULT_FILES
    ),
    *tuple(
        (
            "paper/figures/canonical_emulation/figure_manifest.json",
            "outputs",
            name,
            f"paper/figures/canonical_emulation/{name}",
        )
        for name in CANONICAL_FIGURE_OUTPUTS
    ),
    (
        "paper/figures/canonical_emulation/figure_manifest.json",
        "",
        "config_sha256",
        "configs/canonical_emulation.yaml",
    ),
    (
        "paper/figures/canonical_emulation/figure_manifest.json",
        "",
        "renderer_sha256",
        "scripts/render_canonical_emulation_figures.py",
    ),
)


class ManifestError(RuntimeError):
    """Raised when the release inputs are incomplete or internally inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ManifestError(f"invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"JSON release artifact is not an object: {path}")
    return value


def _resolve_inside(repo_root: Path, value: str | Path, *, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ManifestError(f"{label} must be inside repository root: {resolved}") from exc
    return resolved


def _require_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ManifestError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise ManifestError(f"missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ManifestError(f"empty {label}: {path}")


def _canonical_material_manifest(repo_root: Path) -> dict[str, Any]:
    labelled: dict[str, Path] = {
        "configuration": repo_root / "configs/canonical_emulation.yaml",
        "project_metadata": repo_root / "pyproject.toml",
        "runner": repo_root / "scripts/run_canonical_emulation.py",
    }
    labelled.update(
        {
            f"protocol:{name}": repo_root / name
            for name in CANONICAL_PROTOCOL_FILES
        }
    )
    labelled.update(
        {f"script:{name}": repo_root / name for name in CANONICAL_SCRIPTS}
    )
    source_root = repo_root / "src/qrc_vs_qelm"
    test_root = repo_root / "tests"
    if not source_root.is_dir() or not test_root.is_dir():
        raise ManifestError("canonical source and test directories must exist")
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(repo_root).as_posix()
        labelled[f"source:{relative}"] = path
    for path in sorted(test_root.rglob("*.py")):
        relative = path.relative_to(repo_root).as_posix()
        labelled[f"test:{relative}"] = path

    hashes: dict[str, str] = {}
    for label, path in sorted(labelled.items()):
        _require_regular_file(path, label=f"canonical material file {label}")
        hashes[label] = sha256_file(path)
    return {
        "file_count": len(hashes),
        "identity_sha256": sha256_json(hashes),
        "hashes": hashes,
    }


def _captured_canonical_material_manifest(repo_root: Path) -> dict[str, Any]:
    """Validate the immutable pre-run material ledger stored with the results."""

    manifest_path = repo_root / "results_canonical_confirmatory_v1/canonical_manifest.json"
    _require_regular_file(manifest_path, label="captured canonical manifest")
    manifest = load_strict_json_object(manifest_path)
    reproducibility = manifest.get("reproducibility")
    if not isinstance(reproducibility, dict):
        raise ManifestError("captured canonical manifest lacks reproducibility object")
    material = reproducibility.get("material")
    if not isinstance(material, dict):
        raise ManifestError("captured canonical manifest lacks material object")
    files = material.get("files")
    if not isinstance(files, dict) or not files:
        raise ManifestError("captured canonical material ledger is empty")

    hashes: dict[str, str] = {}
    for label, record in sorted(files.items()):
        if not isinstance(label, str) or not label or not isinstance(record, dict):
            raise ManifestError("malformed captured canonical material entry")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise ManifestError(f"captured material {label!r} lacks a path")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ManifestError(f"unsafe captured material path: {relative!r}")
        digest = str(record.get("sha256", ""))
        if not SHA256_RE.fullmatch(digest):
            raise ManifestError(f"captured material {label!r} has invalid SHA-256")
        size = record.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ManifestError(f"captured material {label!r} has invalid size")
        hashes[label] = digest

    file_count = material.get("file_count")
    identity = str(material.get("identity_sha256", ""))
    if file_count != len(hashes):
        raise ManifestError("captured canonical material file count is inconsistent")
    if not SHA256_RE.fullmatch(identity) or sha256_json(hashes) != identity:
        raise ManifestError("captured canonical material identity is inconsistent")
    return {
        "file_count": len(hashes),
        "identity_sha256": identity,
        "hashes": hashes,
    }


def _verify_frozen_identity(repo_root: Path) -> dict[str, Any]:
    freeze_path = repo_root / "configs/canonical_emulation_freeze.json"
    _require_regular_file(freeze_path, label="canonical freeze record")
    freeze = load_strict_json_object(freeze_path)
    required_values = {
        "schema_version": 1,
        "status": "FROZEN_CONFIRMATORY",
        "study": "canonical_qrc_to_parallel_qelm_emulation",
        "profile": "confirmatory",
        "confirmatory_roots_previously_opened": False,
    }
    for key, expected in required_values.items():
        if freeze.get(key) != expected:
            raise ManifestError(
                f"canonical freeze field {key!r} is {freeze.get(key)!r}, "
                f"expected {expected!r}"
            )
    for key in ("config_sha256", "material_identity_sha256"):
        if not SHA256_RE.fullmatch(str(freeze.get(key, ""))):
            raise ManifestError(f"invalid canonical freeze SHA-256 field: {key}")
    if not str(freeze.get("frozen_at_utc", "")).endswith("Z"):
        raise ManifestError("canonical freeze timestamp is not explicitly UTC")

    config_path = repo_root / "configs/canonical_emulation.yaml"
    _require_regular_file(config_path, label="canonical configuration")
    actual_config_hash = sha256_file(config_path)
    if actual_config_hash != freeze["config_sha256"]:
        raise ManifestError(
            "canonical configuration no longer matches the frozen config SHA-256"
        )

    captured = _captured_canonical_material_manifest(repo_root)
    if captured["identity_sha256"] != freeze["material_identity_sha256"]:
        raise ManifestError(
            "captured canonical material does not match the frozen identity"
        )

    live = _canonical_material_manifest(repo_root)
    if set(live["hashes"]) != set(captured["hashes"]):
        raise ManifestError("live canonical material labels differ from the frozen ledger")
    drift = []
    forbidden_drift = []
    for label in sorted(captured["hashes"]):
        before = captured["hashes"][label]
        after = live["hashes"][label]
        if before == after:
            continue
        entry = {
            "label": label,
            "frozen_sha256": before,
            "live_sha256": after,
        }
        drift.append(entry)
        if label not in POST_FREEZE_MUTABLE_DOCUMENTATION:
            forbidden_drift.append(entry)
    if forbidden_drift:
        raise ManifestError(
            "frozen executable or protocol material changed after confirmation: "
            f"{[entry['label'] for entry in forbidden_drift]}"
        )
    return {
        "freeze_record_path": freeze_path.relative_to(repo_root).as_posix(),
        "freeze_record_sha256": sha256_file(freeze_path),
        "frozen_at_utc": freeze["frozen_at_utc"],
        "study": freeze["study"],
        "profile": freeze["profile"],
        "config_sha256": actual_config_hash,
        "material_file_count": captured["file_count"],
        "material_identity_sha256": captured["identity_sha256"],
        "live_material_identity_sha256": live["identity_sha256"],
        "live_material_matches_frozen_snapshot": not drift,
        "frozen_executable_and_preregistered_material_match": not forbidden_drift,
        "post_freeze_documentation_revisions": drift,
        "confirmatory_roots_previously_opened": False,
    }


def _artifact_entry(path: Path, repo_root: Path, role: str) -> dict[str, Any]:
    _require_regular_file(path, label=role)
    return {
        "role": role,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _collect_python_files(repo_root: Path) -> dict[str, str]:
    collected: dict[str, str] = {}
    for directory, role in (
        (repo_root / "pra_extension/code", "extension_code"),
        (repo_root / "pra_extension/tests", "extension_test"),
    ):
        if not directory.is_dir():
            raise ManifestError(f"missing extension directory: {directory}")
        paths = sorted(directory.rglob("*.py"))
        if not paths:
            raise ManifestError(f"no Python files found in {directory}")
        for path in paths:
            _require_regular_file(path, label=role)
            collected[path.relative_to(repo_root).as_posix()] = role
    return collected


def _collect_paper_sources(repo_root: Path) -> dict[str, str]:
    """Collect exactly the LaTeX sources used to build the single paper PDF."""

    collected: dict[str, str] = {}
    for relative in REQUIRED_PAPER_SOURCES:
        path = repo_root / relative
        _require_regular_file(path, label="paper source")
        collected[relative] = "paper_source"
    return collected


def _verify_embedded_source_hashes(repo_root: Path) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    cache: dict[str, dict[str, Any]] = {}
    for analysis_rel, mapping_key, recorded_key, source_rel in SOURCE_HASH_LINKS:
        analysis = cache.setdefault(
            analysis_rel, load_strict_json_object(repo_root / analysis_rel)
        )
        mapping = analysis if mapping_key == "" else analysis.get(mapping_key)
        if not isinstance(mapping, dict):
            raise ManifestError(
                f"{analysis_rel} lacks the required {mapping_key!r} hash object"
            )
        recorded = str(mapping.get(recorded_key, ""))
        if not SHA256_RE.fullmatch(recorded):
            raise ManifestError(
                f"{analysis_rel} lacks a valid recorded hash for {recorded_key}"
            )
        actual = sha256_file(repo_root / source_rel)
        if actual != recorded:
            raise ManifestError(
                f"embedded source hash mismatch: {analysis_rel} records {recorded} "
                f"for {source_rel}, but the file hashes to {actual}"
            )
        checks.append(
            {
                "analysis": analysis_rel,
                "source": source_rel,
                "sha256": actual,
            }
        )
    return sorted(checks, key=lambda item: (item["analysis"], item["source"]))


def _verify_analysis_boundaries(repo_root: Path) -> None:
    for relative in EXPLORATORY_ANALYSES:
        analysis = load_strict_json_object(repo_root / relative)
        status = str(analysis.get("status", ""))
        if not status.startswith("EXPLORATORY"):
            raise ManifestError(
                f"analysis is not marked exploratory: {relative} ({status!r})"
            )


def _verify_pinned_hashes(repo_root: Path) -> None:
    inventories = (
        ("pinned authoritative", PINNED_AUTHORITATIVE_HASHES),
        ("pinned historical regression", PINNED_HISTORICAL_REGRESSION_HASHES),
    )
    for label, hashes in inventories:
        for relative, expected in sorted(hashes.items()):
            path = repo_root / relative
            _require_regular_file(path, label=f"{label} artifact")
            actual = sha256_file(path)
            if actual != expected:
                raise ManifestError(
                    f"{label} hash mismatch for {relative}: "
                    f"expected {expected}, actual {actual}"
                )


def _collect_and_verify_figures(
    repo_root: Path, figures_dir: Path
) -> dict[str, str]:
    if figures_dir.is_symlink() or not figures_dir.is_dir():
        raise ManifestError(f"missing or symlinked final figures directory: {figures_dir}")
    paths = sorted(path for path in figures_dir.rglob("*") if path.is_file())
    if not paths:
        raise ManifestError(f"final figures directory is empty: {figures_dir}")
    for path in paths:
        _require_regular_file(path, label="final figure artifact")
    suffixes = {path.suffix.lower() for path in paths}
    if ".pdf" not in suffixes or ".png" not in suffixes:
        raise ManifestError("final figures directory must contain PDF and PNG figures")

    figure_manifest_path = figures_dir / "figure_sha256.json"
    _require_regular_file(figure_manifest_path, label="figure SHA-256 manifest")
    recorded = load_strict_json_object(figure_manifest_path)
    actual_figure_paths = {
        path.relative_to(figures_dir).as_posix()
        for path in paths
        if path != figure_manifest_path
    }
    missing_required = sorted(REQUIRED_FINAL_FIGURES - actual_figure_paths)
    if missing_required:
        raise ManifestError(
            f"final figures directory lacks required robustness figure files: "
            f"{missing_required}"
        )
    if set(recorded) != actual_figure_paths:
        missing = sorted(actual_figure_paths - set(recorded))
        stale = sorted(set(recorded) - actual_figure_paths)
        raise ManifestError(
            "figure hash manifest does not exactly cover the final directory; "
            f"unrecorded={missing}, stale={stale}"
        )
    for relative, expected in sorted(recorded.items()):
        if not SHA256_RE.fullmatch(str(expected)):
            raise ManifestError(f"invalid figure SHA-256 for {relative}")
        actual = sha256_file(figures_dir / relative)
        if actual != expected:
            raise ManifestError(
                f"figure hash mismatch for {relative}: recorded {expected}, actual {actual}"
            )
    return {
        path.relative_to(repo_root).as_posix(): "final_figure_artifact"
        for path in paths
    }


def build_manifest(
    *,
    repo_root: Path,
    figures_dir: str | Path = DEFAULT_FINAL_FIGURES_DIR,
    paper_pdf: str | Path = "paper/canonical_emulation/main.pdf",
) -> dict[str, Any]:
    """Validate all release inputs and return deterministic manifest content."""

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ManifestError(f"repository root does not exist: {root}")
    figures = _resolve_inside(root, figures_dir, label="figures directory")
    paper = _resolve_inside(root, paper_pdf, label="paper PDF")
    _require_regular_file(paper, label="paper PDF")
    if not paper.read_bytes()[:5] == b"%PDF-":
        raise ManifestError(f"paper artifact does not have a PDF header: {paper}")

    frozen_identity = _verify_frozen_identity(root)
    _verify_analysis_boundaries(root)
    _verify_pinned_hashes(root)

    roles: dict[str, str] = dict(AUTHORITATIVE_EXTENSION_FILES)
    roles.update(DOCUMENTATION_FILES)
    roles.update(DARK_SECTOR_FILES)
    roles.update(COMPACT_PRA_FIGURE_FILES)
    roles.update(CANONICAL_RESULT_FILES)
    roles.update(CANONICAL_FIGURE_FILES)
    roles.update(DENSE_SECONDARY_FILES)
    roles.update(DENSE_N4_FILES)
    roles.update(THRESHOLD_SENSITIVITY_FILES)
    roles.update(HISTORICAL_REGRESSION_FILES)
    roles.update(REQUIRED_PAPER_ARTIFACTS)
    roles.update(_collect_python_files(root))
    roles.update(_collect_paper_sources(root))
    roles.update(_collect_and_verify_figures(root, figures))
    paper_relative = paper.relative_to(root).as_posix()
    if paper_relative in roles:
        raise ManifestError(f"paper PDF duplicates another release role: {paper_relative}")
    roles[paper_relative] = "compiled_paper_pdf"

    artifacts: dict[str, dict[str, Any]] = {}
    for relative, role in sorted(roles.items()):
        path = root / relative
        if path.suffix.lower() == ".json":
            load_strict_json_object(path)
        artifacts[relative] = _artifact_entry(path, root, role)

    source_hash_checks = _verify_embedded_source_hashes(root)
    artifact_hashes = {
        relative: entry["sha256"] for relative, entry in artifacts.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "POST_CONFIRMATORY_REPRODUCIBILITY_SNAPSHOT",
        "evidence_layers": {
            "canonical": "FROZEN_CONFIRMATORY",
            "dense_secondary": "FROZEN_PROSPECTIVE_SECONDARY",
            "dense_n4": "FROZEN_PROSPECTIVE_N4",
            "dark_sector": "FROZEN_VALIDATION",
            "pra_extension": "EXPLORATORY_POST_CONFIRMATORY",
        },
        "claim_boundary": (
            "A content-addressed handoff for the exact scalar common-suffix "
            "criterion, the if-and-only-if growing reset-QELM emulation theorem, "
            "the reachable-observable joint spectral radius and context law, a "
            "constructive contractive-QRC-to-growing-reset-QELM bound, an exact "
            "protected-memory obstruction, and the bounded phase-map study; not "
            "fixed-architecture equivalence, efficient learning, quantum advantage, "
            "or hardware validation."
        ),
        "canonical_frozen_identity": frozen_identity,
        "figures_directory": figures.relative_to(root).as_posix(),
        "paper_pdf": paper_relative,
        "source_hash_checks": source_hash_checks,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "artifact_set_sha256": sha256_json(artifact_hashes),
    }
    manifest["manifest_identity_sha256"] = sha256_json(manifest)
    return manifest


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output = output_path.resolve()
    if not output.parent.is_dir():
        raise ManifestError(f"manifest output parent does not exist: {output.parent}")
    encoded = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(output)


def _parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--figures-dir",
        default=DEFAULT_FINAL_FIGURES_DIR,
        help="Final figure directory, relative to the repository or absolute.",
    )
    parser.add_argument(
        "--paper-pdf",
        default="paper/canonical_emulation/main.pdf",
        help="Compiled paper PDF, relative to the repository or absolute.",
    )
    parser.add_argument(
        "--output",
        default="pra_extension/PRA_EXTENSION_RELEASE.json",
        help="Output JSON path, relative to the repository or absolute.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        output_arg = Path(args.output)
        output = (
            output_arg.resolve()
            if output_arg.is_absolute()
            else (root / output_arg).resolve()
        )
        manifest = build_manifest(
            repo_root=root,
            figures_dir=args.figures_dir,
            paper_pdf=args.paper_pdf,
        )
        artifact_paths = {(root / path).resolve() for path in manifest["artifacts"]}
        if output in artifact_paths:
            raise ManifestError("manifest output cannot also be a hashed input artifact")
        write_manifest(manifest, output)
    except ManifestError as exc:
        parser.exit(2, f"release manifest error: {exc}\n")
    try:
        output_display = output.relative_to(root).as_posix()
    except ValueError:
        output_display = str(output)
    print(
        f"wrote {output_display} with "
        f"{manifest['artifact_count']} artifacts; "
        f"identity={manifest['manifest_identity_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
