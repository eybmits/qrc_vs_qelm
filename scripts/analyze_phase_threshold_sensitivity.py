#!/usr/bin/env python
"""Post hoc threshold sensitivity of the frozen history frontiers.

This analysis changes only the numerical history-error tolerance and the
number of successful root blocks required to call a tested window sufficient.
It does not recompute trajectories, fit models, or reclassify the full phase
diagram.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SECONDARY_SOURCE = ROOT / "results_dense_phase_secondary_v1" / "suffix_decomposition.csv"
N4_SOURCE = ROOT / "results_dense_phase_n4_v1" / "suffix_decomposition.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "paper"
    / "figures"
    / "information_interface"
    / "phase_threshold_sensitivity.json"
)

EXPECTED_GAMMAS = (0.0, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
EXPECTED_WINDOWS = (1, 2, 3, 4, 5, 6)
EXPECTED_ROOTS_PER_SIZE = 12
EXPECTED_ROWS_PER_ROOT_CELL = 10
EXPECTED_SOURCE_ROWS = {"secondary": 15_840, "n4": 7_920}
EXPECTED_SIZES = {"secondary": (2, 3), "n4": (4,)}
TOLERANCES = (0.08, 0.10, 0.12)
ROOT_SUCCESS_RULES = (9, 10, 11)
PRIMARY_TOLERANCE = 0.10
PRIMARY_ROOT_SUCCESS_RULE = 10
HISTORY_MODEL = "rr_same_identity"
RIGHT_CENSORED_WINDOW = 7
HISTORY_SPREAD_ATOL = 1e-12

REQUIRED_COLUMNS = {
    "root_id",
    "teacher_n_qubits",
    "teacher_dissipator",
    "teacher_gamma",
    "window",
    "model",
    "history_nrmse",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gamma_key(gamma: float) -> str:
    return f"{gamma:g}"


def _canonical_gamma(raw: str, *, source: Path, line_number: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: invalid teacher_gamma {raw!r}") from exc
    for expected in EXPECTED_GAMMAS:
        if math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
            return expected
    raise ValueError(f"{source}:{line_number}: teacher_gamma is outside the frozen grid: {value}")


def _integer(raw: str, label: str, *, source: Path, line_number: int) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: invalid {label} {raw!r}") from exc
    return value


def _finite_nonnegative(raw: str, label: str, *, source: Path, line_number: int) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: invalid {label} {raw!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{source}:{line_number}: {label} must be finite and nonnegative")
    return value


def _load_frozen_source(
    path: Path,
    *,
    source_name: str,
) -> tuple[dict[tuple[int, str, float, int], float], dict[str, Any]]:
    """Load and validate one frozen suffix table.

    The history error is repeated on all ten model rows in a root/grid cell.
    We verify that equality and then retain the single reset-replay identity
    row, matching the registered dense-phase analyzers.
    """

    path = Path(path).resolve()
    if source_name not in EXPECTED_SIZES:
        raise ValueError(f"unknown source name: {source_name}")
    if not path.is_file():
        raise FileNotFoundError(path)

    all_history_values: dict[tuple[int, str, float, int], list[float]] = defaultdict(list)
    identity_values: dict[tuple[int, str, float, int], list[float]] = defaultdict(list)
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            n_qubits = _integer(
                row["teacher_n_qubits"],
                "teacher_n_qubits",
                source=path,
                line_number=line_number,
            )
            if n_qubits not in EXPECTED_SIZES[source_name]:
                raise ValueError(
                    f"{path}:{line_number}: unexpected teacher size {n_qubits} "
                    f"for {source_name} source"
                )
            root_id = str(row["root_id"])
            if not root_id:
                raise ValueError(f"{path}:{line_number}: empty root_id")
            gamma = _canonical_gamma(
                row["teacher_gamma"], source=path, line_number=line_number
            )
            window = _integer(
                row["window"], "window", source=path, line_number=line_number
            )
            if window not in EXPECTED_WINDOWS:
                raise ValueError(f"{path}:{line_number}: window is outside the frozen grid")
            if row["teacher_dissipator"] != "local_loss":
                raise ValueError(f"{path}:{line_number}: expected local_loss dissipator")
            history_nrmse = _finite_nonnegative(
                row["history_nrmse"],
                "history_nrmse",
                source=path,
                line_number=line_number,
            )
            key = (n_qubits, root_id, gamma, window)
            all_history_values[key].append(history_nrmse)
            if row["model"] == HISTORY_MODEL:
                identity_values[key].append(history_nrmse)

    if row_count != EXPECTED_SOURCE_ROWS[source_name]:
        raise ValueError(
            f"{source_name} source must contain {EXPECTED_SOURCE_ROWS[source_name]:,} rows; "
            f"found {row_count:,}"
        )

    expected_keys: set[tuple[int, str, float, int]] = set()
    roots_by_size: dict[int, tuple[str, ...]] = {}
    for n_qubits in EXPECTED_SIZES[source_name]:
        roots = tuple(
            sorted({key[1] for key in all_history_values if key[0] == n_qubits})
        )
        if len(roots) != EXPECTED_ROOTS_PER_SIZE:
            raise ValueError(
                f"N_T={n_qubits} must contain {EXPECTED_ROOTS_PER_SIZE} roots; "
                f"found {len(roots)}"
            )
        roots_by_size[n_qubits] = roots
        expected_keys.update(
            (n_qubits, root, gamma, window)
            for root in roots
            for gamma in EXPECTED_GAMMAS
            for window in EXPECTED_WINDOWS
        )

    if set(all_history_values) != expected_keys:
        missing_keys = expected_keys.difference(all_history_values)
        extra_keys = set(all_history_values).difference(expected_keys)
        raise ValueError(
            f"{source_name} history cells do not form the frozen Cartesian grid "
            f"(missing={len(missing_keys)}, extra={len(extra_keys)})"
        )
    if set(identity_values) != expected_keys:
        raise ValueError(f"{source_name} lacks one {HISTORY_MODEL} row in every root-grid cell")

    history: dict[tuple[int, str, float, int], float] = {}
    for key in sorted(expected_keys):
        repeated = all_history_values[key]
        if len(repeated) != EXPECTED_ROWS_PER_ROOT_CELL:
            raise ValueError(
                f"{source_name} root-grid cell {key} must contain "
                f"{EXPECTED_ROWS_PER_ROOT_CELL} model rows"
            )
        if max(repeated) - min(repeated) > HISTORY_SPREAD_ATOL:
            raise ValueError(f"history NRMSE changed across models in cell {key}")
        selected = identity_values[key]
        if len(selected) != 1:
            raise ValueError(f"{source_name} cell {key} must contain one {HISTORY_MODEL} row")
        history[key] = selected[0]

    audit = {
        "source_name": source_name,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "input_rows": row_count,
        "history_root_cells": len(history),
        "teacher_n_qubits": list(EXPECTED_SIZES[source_name]),
        "roots_by_teacher_n_qubits": {
            str(n_qubits): list(roots_by_size[n_qubits])
            for n_qubits in EXPECTED_SIZES[source_name]
        },
    }
    return history, audit


def _cell_counts(
    history: dict[tuple[int, str, float, int], float]
) -> tuple[list[dict[str, Any]], dict[tuple[int, float, float, int], int]]:
    records: list[dict[str, Any]] = []
    lookup: dict[tuple[int, float, float, int], int] = {}
    for n_qubits in (2, 3, 4):
        roots = sorted({key[1] for key in history if key[0] == n_qubits})
        if len(roots) != EXPECTED_ROOTS_PER_SIZE:
            raise ValueError(f"N_T={n_qubits}: expected 12 root blocks")
        for tolerance in TOLERANCES:
            for gamma in EXPECTED_GAMMAS:
                for window in EXPECTED_WINDOWS:
                    values = [
                        history[(n_qubits, root, gamma, window)] for root in roots
                    ]
                    successful_roots = sum(value <= tolerance for value in values)
                    lookup[(n_qubits, tolerance, gamma, window)] = successful_roots
                    records.append(
                        {
                            "teacher_n_qubits": n_qubits,
                            "tolerance_nrmse": tolerance,
                            "teacher_gamma": gamma,
                            "window": window,
                            "successful_roots": successful_roots,
                            "root_blocks": EXPECTED_ROOTS_PER_SIZE,
                        }
                    )
    return records, lookup


def _frontier_record(
    counts: dict[tuple[int, float, float, int], int],
    *,
    n_qubits: int,
    tolerance: float,
    minimum_successes: int,
) -> dict[str, Any]:
    frontier: list[dict[str, Any]] = []
    values: list[int] = []
    for gamma in EXPECTED_GAMMAS:
        qualifying = [
            window
            for window in EXPECTED_WINDOWS
            if counts[(n_qubits, tolerance, gamma, window)] >= minimum_successes
        ]
        window = min(qualifying) if qualifying else RIGHT_CENSORED_WINDOW
        values.append(window)
        frontier.append(
            {
                "teacher_gamma": gamma,
                "minimum_window_censored_at_7": window,
                "display": ">6" if window == RIGHT_CENSORED_WINDOW else str(window),
            }
        )

    differences = [
        right - left for left, right in zip(values[:-1], values[1:], strict=True)
    ]
    violations = [
        {
            "from_gamma": EXPECTED_GAMMAS[index],
            "to_gamma": EXPECTED_GAMMAS[index + 1],
            "from_window": values[index],
            "to_window": values[index + 1],
        }
        for index, difference in enumerate(differences)
        if difference > 0
    ]
    return {
        "teacher_n_qubits": n_qubits,
        "tolerance_nrmse": tolerance,
        "minimum_successful_roots": minimum_successes,
        "root_blocks": EXPECTED_ROOTS_PER_SIZE,
        "frontier": frontier,
        "non_increasing_with_loss": not violations,
        "strict_decrease_count": sum(difference < 0 for difference in differences),
        "finite_loss_rate_count": sum(value <= max(EXPECTED_WINDOWS) for value in values),
        "monotonicity_violations": violations,
    }


def _frontier_values(record: dict[str, Any]) -> tuple[int, ...]:
    return tuple(point["minimum_window_censored_at_7"] for point in record["frontier"])


def build_analysis(
    secondary_source: Path = SECONDARY_SOURCE,
    n4_source: Path = N4_SOURCE,
) -> dict[str, Any]:
    secondary, secondary_audit = _load_frozen_source(
        secondary_source, source_name="secondary"
    )
    n4, n4_audit = _load_frozen_source(n4_source, source_name="n4")
    overlap = set(secondary).intersection(n4)
    if overlap:
        raise ValueError(f"source history keys overlap unexpectedly: {len(overlap)}")
    history = {**secondary, **n4}
    expected_history_cells = 3 * EXPECTED_ROOTS_PER_SIZE * len(EXPECTED_GAMMAS) * len(EXPECTED_WINDOWS)
    if len(history) != expected_history_cells:
        raise ValueError("combined history rows do not form the three-size frozen grid")

    count_records, count_lookup = _cell_counts(history)
    frontiers = [
        _frontier_record(
            count_lookup,
            n_qubits=n_qubits,
            tolerance=tolerance,
            minimum_successes=minimum_successes,
        )
        for n_qubits in (2, 3, 4)
        for tolerance in TOLERANCES
        for minimum_successes in ROOT_SUCCESS_RULES
    ]
    if len(frontiers) != 27:
        raise AssertionError("threshold grid must produce 27 frontiers")
    primary_frontiers = [
        record
        for record in frontiers
        if record["tolerance_nrmse"] == PRIMARY_TOLERANCE
        and record["minimum_successful_roots"] == PRIMARY_ROOT_SUCCESS_RULE
    ]
    if len(primary_frontiers) != 3:
        raise AssertionError("primary settings must produce one frontier per target size")

    primary_by_size = {
        record["teacher_n_qubits"]: _frontier_values(record)
        for record in primary_frontiers
    }
    shift_counts = {"shorter": 0, "same": 0, "longer": 0}
    alternative_point_count = 0
    for record in frontiers:
        if (
            record["tolerance_nrmse"] == PRIMARY_TOLERANCE
            and record["minimum_successful_roots"] == PRIMARY_ROOT_SUCCESS_RULE
        ):
            continue
        primary_values = primary_by_size[record["teacher_n_qubits"]]
        for value, primary_value in zip(
            _frontier_values(record), primary_values, strict=True
        ):
            alternative_point_count += 1
            if value < primary_value:
                shift_counts["shorter"] += 1
            elif value > primary_value:
                shift_counts["longer"] += 1
            else:
                shift_counts["same"] += 1

    pointwise_ranges: list[dict[str, Any]] = []
    stable_point_count = 0
    for n_qubits in (2, 3, 4):
        size_frontiers = [record for record in frontiers if record["teacher_n_qubits"] == n_qubits]
        for gamma_index, gamma in enumerate(EXPECTED_GAMMAS):
            values = [
                _frontier_values(record)[gamma_index] for record in size_frontiers
            ]
            stable = min(values) == max(values)
            stable_point_count += int(stable)
            pointwise_ranges.append(
                {
                    "teacher_n_qubits": n_qubits,
                    "teacher_gamma": gamma,
                    "minimum_window_over_threshold_grid": min(values),
                    "maximum_window_over_threshold_grid": max(values),
                    "stable_across_all_nine_threshold_rules": stable,
                }
            )

    non_increasing_count = sum(
        record["non_increasing_with_loss"] for record in frontiers
    )
    return {
        "schema_version": 1,
        "analysis_type": "post_hoc_history_frontier_threshold_sensitivity",
        "data_scope": {
            "sources": [secondary_audit, n4_audit],
            "teacher_n_qubits": [2, 3, 4],
            "teacher_dissipator": "local_loss",
            "history_metric": "history_nrmse",
            "history_row_model": HISTORY_MODEL,
            "loss_rates": list(EXPECTED_GAMMAS),
            "window_lengths": list(EXPECTED_WINDOWS),
            "root_blocks_per_teacher_size": EXPECTED_ROOTS_PER_SIZE,
            "right_censored_window_value": RIGHT_CENSORED_WINDOW,
            "right_censored_window_meaning": "no tested window d<=6 reaches the rule",
        },
        "threshold_grid": {
            "nrmse_tolerances": list(TOLERANCES),
            "minimum_successful_roots": list(ROOT_SUCCESS_RULES),
            "primary_tolerance_nrmse": PRIMARY_TOLERANCE,
            "primary_minimum_successful_roots": PRIMARY_ROOT_SUCCESS_RULE,
            "frontiers_per_teacher_size": 9,
            "frontiers_total": 27,
        },
        "counts": {
            "input_rows": sum(
                source["input_rows"] for source in (secondary_audit, n4_audit)
            ),
            "history_root_cells": len(history),
            "history_root_cells_per_teacher_size": expected_history_cells // 3,
            "loss_window_cells": 3 * len(EXPECTED_GAMMAS) * len(EXPECTED_WINDOWS),
            "history_sufficiency_count_records": len(count_records),
            "frontiers": len(frontiers),
            "frontier_loss_points": len(frontiers) * len(EXPECTED_GAMMAS),
        },
        "history_sufficient_root_counts": count_records,
        "frontiers": frontiers,
        "primary_frontiers": primary_frontiers,
        "sensitivity_summary": {
            "frontiers_evaluated": len(frontiers),
            "non_increasing_frontiers": non_increasing_count,
            "all_27_measured_frontiers_non_increasing_with_loss": (
                non_increasing_count == len(frontiers)
            ),
            "alternative_frontier_points_compared_with_primary": alternative_point_count,
            "alternative_frontier_point_shifts_relative_to_primary": shift_counts,
            "teacher_size_loss_points": len(pointwise_ranges),
            "teacher_size_loss_points_stable_across_all_nine_threshold_rules": stable_point_count,
            "teacher_size_loss_points_changed_across_threshold_grid": (
                len(pointwise_ranges) - stable_point_count
            ),
            "pointwise_frontier_ranges": pointwise_ranges,
        },
        "supported_statement": (
            "Across the tested two-, three-, and four-qubit local-loss cohorts, all 27 "
            "post hoc measured history-sufficiency frontiers are non-increasing over the "
            "sampled loss grid when the NRMSE tolerance is varied over 0.08, 0.10, and "
            "0.12 and the root-success rule over 9, 10, and 11 of 12."
        ),
        "claim_boundary": (
            "This post hoc analysis varies only the history-sufficiency threshold and "
            "root-success rule. It does not establish invariance of individual cell labels, "
            "QELM emulation labels, representation-limited labels, unresolved labels, or "
            "measurement-limited labels. A censored value of 7 means only that no tested "
            "window d<=6 reaches the specified rule."
        ),
    }


def serialize_analysis(analysis: dict[str, Any]) -> str:
    return json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_analysis(
    output: Path = DEFAULT_OUTPUT,
    *,
    secondary_source: Path = SECONDARY_SOURCE,
    n4_source: Path = N4_SOURCE,
) -> dict[str, Any]:
    analysis = build_analysis(secondary_source, n4_source)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialize_analysis(analysis), encoding="utf-8")
    return analysis


def _primary_frontier_summary(records: Iterable[dict[str, Any]]) -> dict[str, list[int]]:
    return {
        str(record["teacher_n_qubits"]): list(_frontier_values(record))
        for record in records
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secondary-source", type=Path, default=SECONDARY_SOURCE)
    parser.add_argument("--n4-source", type=Path, default=N4_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    analysis = write_analysis(
        args.output,
        secondary_source=args.secondary_source,
        n4_source=args.n4_source,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "all_27_measured_frontiers_non_increasing_with_loss": analysis[
                    "sensitivity_summary"
                ]["all_27_measured_frontiers_non_increasing_with_loss"],
                "primary_frontiers_censored_at_7": _primary_frontier_summary(
                    analysis["primary_frontiers"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
