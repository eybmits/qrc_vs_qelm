from __future__ import annotations

import pandas as pd
import pytest

from qrc_vs_qelm.checkpointing import run_chunked


def _evaluate(items: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": items,
            "value": [item * item for item in items],
            "complete": [1 for _ in items],
        }
    )


def test_resume_byte_identical(tmp_path) -> None:
    items = list(range(6))
    interrupted_dir = tmp_path / "interrupted"
    single_pass_dir = tmp_path / "single"
    calls = {"n": 0}

    def flaky(items_chunk: list[int]) -> pd.DataFrame:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("stop after chunk 2")
        return _evaluate(items_chunk)

    with pytest.raises(RuntimeError, match="stop after chunk 2"):
        run_chunked(
            items,
            flaky,
            out_dir=interrupted_dir,
            stem="scores",
            chunk_size=2,
            expected_cols=("complete",),
        )

    resumed = run_chunked(
        items,
        _evaluate,
        out_dir=interrupted_dir,
        stem="scores",
        chunk_size=2,
        expected_cols=("complete",),
    )
    single = run_chunked(
        items,
        _evaluate,
        out_dir=single_pass_dir,
        stem="scores",
        chunk_size=2,
        expected_cols=("complete",),
    )

    assert resumed.equals(single)
    assert (interrupted_dir / "scores.csv").read_bytes() == (
        single_pass_dir / "scores.csv"
    ).read_bytes()


def test_incomplete_chunk_recomputed(tmp_path) -> None:
    out_dir = tmp_path / "run"
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    pd.DataFrame({"id": [0, 1], "value": [0, 1], "complete": [1, None]}).to_csv(
        checkpoint_dir / "scores_chunk_0001.csv",
        index=False,
    )
    calls = {"n": 0}

    def counted(items_chunk: list[int]) -> pd.DataFrame:
        calls["n"] += 1
        return _evaluate(items_chunk)

    frame = run_chunked(
        [0, 1],
        counted,
        out_dir=out_dir,
        stem="scores",
        chunk_size=2,
        expected_cols=("complete",),
    )

    assert calls["n"] == 1
    assert frame["complete"].notna().all()
