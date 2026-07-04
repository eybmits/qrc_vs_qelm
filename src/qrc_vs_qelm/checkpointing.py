"""Chunked checkpoint and resume support for stage runners."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def run_chunked(
    items: Iterable[Any],
    evaluate_chunk: Callable[[list[Any]], pd.DataFrame],
    *,
    out_dir: str | Path,
    stem: str,
    chunk_size: int,
    expected_cols: Sequence[str],
    dedup_cols: Sequence[str] | None = None,
    item_id_col: str | None = None,
    item_id_fn: Callable[[Any], Any] | None = None,
) -> pd.DataFrame:
    """Run chunked evaluation with deterministic checkpoint resume."""

    item_list = list(items)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[pd.DataFrame] = []
    total_chunks = int(math.ceil(len(item_list) / chunk_size)) if item_list else 0
    for chunk_idx, start in enumerate(range(0, len(item_list), chunk_size), start=1):
        stop = min(start + chunk_size, len(item_list))
        chunk_path = checkpoint_dir / f"{stem}_chunk_{chunk_idx:04d}.csv"
        expected_rows = stop - start
        expected_ids = (
            [str(item_id_fn(item)) for item in item_list[start:stop]]
            if item_id_fn is not None
            else None
        )
        if _chunk_complete(
            chunk_path,
            expected_rows,
            expected_cols,
            item_id_col=item_id_col,
            expected_ids=expected_ids,
        ):
            chunk = pd.read_csv(chunk_path, float_precision="round_trip")
        else:
            chunk = _as_frame(evaluate_chunk(item_list[start:stop]))
            chunk.to_csv(
                chunk_path,
                index=False,
                float_format="%.17g",
                lineterminator="\n",
            )
        chunks.append(chunk)
        partial = _concat_dedup(chunks, dedup_cols=dedup_cols)
        _write_csv(partial, out_path / f"{stem}_partial.csv")

    result = _concat_dedup(chunks, dedup_cols=dedup_cols)
    if total_chunks == 0:
        result = pd.DataFrame()
    _write_csv(result, out_path / f"{stem}.csv")
    return result


def _chunk_complete(
    path: Path,
    expected_rows: int,
    expected_cols: Sequence[str],
    *,
    item_id_col: str | None = None,
    expected_ids: Sequence[str] | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
    except Exception:
        return False
    if expected_ids is None:
        if len(frame) != int(expected_rows):
            return False
    else:
        if item_id_col is None or item_id_col not in frame.columns:
            return False
        observed = set(frame[item_id_col].astype(str))
        if observed != set(str(item_id) for item_id in expected_ids):
            return False
    for col in expected_cols:
        if col not in frame.columns or frame[col].isna().any():
            return False
    return True


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame(value)


def _concat_dedup(
    chunks: Sequence[pd.DataFrame],
    *,
    dedup_cols: Sequence[str] | None,
) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()
    frame = pd.concat(chunks, ignore_index=True)
    key = list(dedup_cols) if dedup_cols is not None else _infer_dedup_cols(frame)
    if key:
        frame = frame.drop_duplicates(subset=key, keep="last", ignore_index=True)
    return frame


def _infer_dedup_cols(frame: pd.DataFrame) -> list[str]:
    scoring_key = [
        "cell_id",
        "model",
        "target",
        "input_seed",
        "coupling_seed",
        "shot_seed",
    ]
    if all(col in frame.columns for col in scoring_key):
        return scoring_key
    for candidate in ("id", "item_id", "cell_id"):
        if candidate in frame.columns:
            return [candidate]
    return []


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


__all__ = ["run_chunked"]
