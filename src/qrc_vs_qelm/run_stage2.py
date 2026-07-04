"""Stage-2 runner for the pre-registered RQ2 shot-tier sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrc_vs_qelm import controls, manifests, pipeline, run_stage1, shots
from qrc_vs_qelm.checkpointing import run_chunked
from qrc_vs_qelm.manifests import sha256_file, sha256_frame, write_manifest
from qrc_vs_qelm.seeds import SEED_GRIDS, SeedCell, grid_manifest


ROOT = Path(__file__).resolve().parents[2]
STAGE2_SCORE_STEM = "stage2_scores"
QUANTUM_SHOT_NOISE_MODELS = frozenset({"qrc", "qelm"})
SHRINK_RULE_MESSAGE = (
    "Projected runtime exceeds the Stage-2 budget; apply the preregistered "
    "shrink rule: cut input seeds first, then the secondary product family. "
    "Do not change gates, phase-diagram cells, fairness rules, shot tiers, or "
    "win conditions."
)

STAGE2_SCORE_COLUMNS = [
    "cell_id",
    "input_seed",
    "coupling_seed",
    "shot_seed",
    "shots",
    "model",
    "target",
    "target_family",
    "family_cell",
    "confirmatory",
    "tau",
    "p",
    "capacity_test",
    "nrmse_test",
    "alpha",
    "effective_rank",
    "effective_rank_diag",
    "multiplex_gain",
    "D",
    "n_obs",
    "n_multiplex",
    "selected_hp",
    "gamma",
    "uninformative",
    "capacity_linear_delay",
    "capacity_poly_delay",
    "complete",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dirty_state = {
        "dirty_tree_allowed": bool(args.allow_dirty),
        **manifests.require_clean_tree(bool(args.allow_dirty)),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    stage1_out = Path(args.stage1_out)
    stage1_manifest = _load_stage1_manifest(stage1_out)
    config = _stage_config(stage1_manifest, str(args.profile))
    verified_cache = _verify_stage1_feature_cache(stage1_out, stage1_manifest)
    cells, skipped_cells, seed_grid_name = _stage2_seed_cells(
        config,
        verified_cache,
        profile=str(args.profile),
    )
    shot_tiers = _shot_tiers(config)
    items = _stage2_items(cells, shot_tiers)
    runtime_projection = _assert_runtime_budget(config, items)
    shot_noise_protocol = _shot_noise_protocol()
    _invalidate_stage2_checkpoints_if_stale(out_dir, shot_noise_protocol)

    manifest_state: dict[str, Any] = {
        "stage1_feature_cache": verified_cache,
        "skipped_seed_cells": skipped_cells,
        "errors": [],
    }

    def evaluate_items(chunk: list[dict[str, Any]]) -> pd.DataFrame:
        frames = [
            _score_shot_item(
                item,
                config=config,
                stage1_manifest=stage1_manifest,
                feature_cache=verified_cache,
            )
            for item in chunk
        ]
        if not frames:
            return pd.DataFrame(columns=[*STAGE2_SCORE_COLUMNS, "stage2_item_id"])
        return _order_stage2_scores(pd.concat(frames, ignore_index=True), keep_item_id=True)

    try:
        scores = run_chunked(
            items,
            evaluate_items,
            out_dir=out_dir,
            stem=STAGE2_SCORE_STEM,
            chunk_size=int(args.checkpoint_every),
            expected_cols=(
                "stage2_item_id",
                "cell_id",
                "model",
                "target",
                "shots",
                "capacity_test",
                "complete",
            ),
            dedup_cols=("stage2_item_id", "model", "target"),
            item_id_col="stage2_item_id",
            item_id_fn=lambda item: item["item_id"],
        )
    except Exception as exc:
        manifest_state["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        _write_stage2_manifest(
            out_dir,
            config=config,
            seed_grid_name=seed_grid_name,
            shot_tiers=shot_tiers,
            dirty_state=dirty_state,
            runtime_projection=runtime_projection,
            stage1_manifest=stage1_manifest,
            manifest_state=manifest_state,
            elapsed_s=float(time.perf_counter() - start),
            scores=None,
        )
        raise

    scores = _order_stage2_scores(scores, keep_item_id=False)
    _write_scores(scores, out_dir / "stage2_scores.csv")
    _write_stage2_manifest(
        out_dir,
        config=config,
        seed_grid_name=seed_grid_name,
        shot_tiers=shot_tiers,
        dirty_state=dirty_state,
        runtime_projection=runtime_projection,
        stage1_manifest=stage1_manifest,
        manifest_state=manifest_state,
        elapsed_s=float(time.perf_counter() - start),
        scores=scores,
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if int(args.checkpoint_every) < 1:
        raise ValueError("--checkpoint-every must be >= 1")
    return args


def _load_stage1_manifest(stage1_out: Path) -> dict[str, Any]:
    path = stage1_out / "stage1_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Stage-1 manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    payload = manifest.get("core", {}).get("payload", {})
    if int(payload.get("stage", -1)) != 1:
        raise ValueError(f"{path} is not a Stage-1 manifest")
    return manifest


def _stage_config(stage1_manifest: Mapping[str, Any], profile: str) -> dict[str, Any]:
    payload = stage1_manifest["core"]["payload"]
    stage1_profile = str(payload.get("profile"))
    if stage1_profile != str(profile):
        raise ValueError(
            f"Stage-1 profile mismatch: manifest={stage1_profile!r}, "
            f"requested={profile!r}"
        )
    config = dict(payload["config"])
    if str(config.get("profile")) != str(profile):
        raise ValueError(f"Stage-1 config profile mismatch: {config.get('profile')!r}")
    return config


def _verify_stage1_feature_cache(
    stage1_out: Path,
    stage1_manifest: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, str]]]:
    payload = stage1_manifest["core"]["payload"]
    feature_cache = payload.get("manifest_state", {}).get("feature_cache", {})
    if not feature_cache:
        raise ValueError("Stage-1 manifest contains no feature-cache hashes")

    verified: dict[str, dict[str, dict[str, str]]] = {}
    for cell_id, by_model in sorted(feature_cache.items()):
        verified[str(cell_id)] = {}
        for model in ("qrc", "qelm"):
            entry = by_model.get(model)
            if not isinstance(entry, Mapping):
                raise ValueError(f"missing Stage-1 {model} cache entry for {cell_id}")
            path = _resolve_feature_path(stage1_out, str(entry.get("path", "")), model, str(cell_id))
            expected = str(entry.get("sha256", "")).strip().lower()
            observed = sha256_file(path)
            if observed != expected:
                raise RuntimeError(
                    f"Stage-1 cached features changed for {cell_id}/{model}: "
                    f"expected {expected}, observed {observed}"
                )
            sidecar = path.with_suffix(path.suffix + ".sha256")
            if not sidecar.exists():
                raise FileNotFoundError(f"missing Stage-1 feature sidecar: {sidecar}")
            sidecar_hash = sidecar.read_text(encoding="utf-8").strip().lower()
            if sidecar_hash != observed:
                raise RuntimeError(
                    f"Stage-1 sidecar hash mismatch for {cell_id}/{model}: "
                    f"sidecar {sidecar_hash}, observed {observed}"
                )
            verified[str(cell_id)][model] = {
                "path": str(path),
                "sha256": observed,
            }
    return verified


def _resolve_feature_path(
    stage1_out: Path,
    recorded_path: str,
    model: str,
    cell_id: str,
) -> Path:
    candidates = []
    if recorded_path:
        candidates.append(Path(recorded_path))
    candidates.append(run_stage1._feature_cache_path(stage1_out, model, cell_id))
    if recorded_path:
        candidates.append(stage1_out / "features" / Path(recorded_path).name)

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"missing Stage-1 cached features for {cell_id}/{model}; checked "
        + ", ".join(str(path) for path in candidates)
    )


def _stage2_seed_cells(
    config: Mapping[str, Any],
    verified_cache: Mapping[str, Any],
    *,
    profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    seed_grid_name = str(
        config.get("seed_grids", {}).get("stage2", config.get("seed_grid", "stage2_shots"))
    )
    raw_cells = list(SEED_GRIDS[seed_grid_name])
    available = set(str(cell_id) for cell_id in verified_cache)
    cells: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cell in raw_cells:
        stage1_cell = SeedCell(
            input_seed=int(cell.input_seed),
            coupling_seed=int(cell.coupling_seed),
            shot_seed=0,
        )
        stage1_cell_id = run_stage1._cell_id(stage1_cell)
        record = {
            "seed_cell": cell,
            "cell_id": run_stage1._cell_id(cell),
            "stage1_cell_id": stage1_cell_id,
        }
        if stage1_cell_id in available:
            cells.append(record)
        else:
            skipped.append(record)

    if skipped and str(profile) != "smoke":
        missing = ", ".join(item["stage1_cell_id"] for item in skipped)
        raise FileNotFoundError(
            "Stage-2 requires matching Stage-1 cached features for every "
            f"frozen seed cell; missing: {missing}"
        )
    if not cells:
        raise FileNotFoundError("Stage-2 found no seed cells with Stage-1 cached features")
    return cells, [_jsonable_skipped(item) for item in skipped], seed_grid_name


def _jsonable_skipped(item: Mapping[str, Any]) -> dict[str, Any]:
    cell = item["seed_cell"]
    return {
        "cell_id": str(item["cell_id"]),
        "stage1_cell_id": str(item["stage1_cell_id"]),
        "seed_cell": {
            "input_seed": int(cell.input_seed),
            "coupling_seed": int(cell.coupling_seed),
            "shot_seed": int(cell.shot_seed),
        },
    }


def _shot_tiers(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    tiers = config.get("shot_tiers")
    if not tiers:
        raise ValueError("config must define shot_tiers")
    return [_parse_shot_tier(tier) for tier in tiers]


def _parse_shot_tier(value: Any) -> dict[str, Any]:
    infinity_tokens = {"infinity", "+infinity", "inf", "+inf", "∞", "+∞"}
    if isinstance(value, str) and value.strip().lower() in infinity_tokens:
        return {"label": "infinity", "shots": np.inf}
    shots_value = int(value)
    if shots_value < 1:
        raise ValueError(f"shot tier must be positive or infinity, got {value!r}")
    return {"label": str(shots_value), "shots": shots_value}


def _stage2_items(
    cells: Sequence[Mapping[str, Any]],
    shot_tiers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for cell in cells:
        for tier_index, tier in enumerate(shot_tiers):
            item_id = f"{cell['cell_id']}_shots{tier['label']}"
            items.append({**cell, **tier, "tier_index": tier_index, "item_id": item_id})
    return items


def _assert_runtime_budget(
    config: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    runtime_cfg = config.get("runtime_budget", {})
    budget_hours = float(runtime_cfg.get("stage_budget_hours", 10.0))
    projection = {
        "available": True,
        "n_items": int(len(items)),
        "total_s": 0.0,
        "total_hours": 0.0,
        "budget_hours": budget_hours,
        "note": "Stage-2 reuses cached Stage-1 features and reruns ridge only.",
    }
    if projection["total_hours"] > budget_hours:
        raise SystemExit(SHRINK_RULE_MESSAGE)
    return projection


def _score_shot_item(
    item: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    stage1_manifest: Mapping[str, Any],
    feature_cache: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> pd.DataFrame:
    cell: SeedCell = item["seed_cell"]
    input_seed = int(cell.input_seed)
    coupling_seed = int(cell.coupling_seed)
    shot_seed = int(cell.shot_seed)
    cell_id = str(item["cell_id"])
    stage1_cell_id = str(item["stage1_cell_id"])
    shot_label = str(item["label"])
    shot_value = item["shots"]
    tier_index = int(item["tier_index"])

    rng = np.random.default_rng(input_seed)
    inputs = rng.uniform(0.0, 1.0, size=int(config["T"]))
    targets, target_meta = run_stage1._build_targets(inputs, input_seed=input_seed, config=config)
    max_delay = run_stage1._max_required_delay(config, target_meta)
    splits = pipeline.make_splits(
        pipeline.valid_indices(
            int(config["T"]),
            washout=int(config["washout"]),
            max_delay=max_delay,
            targets=targets,
        )
    )
    alphas = tuple(float(alpha) for alpha in config["hp_grids"]["alpha"])

    qrc_bundle = run_stage1._load_bundle(Path(feature_cache[stage1_cell_id]["qrc"]["path"]))
    qelm_bundle = run_stage1._load_bundle(Path(feature_cache[stage1_cell_id]["qelm"]["path"]))
    if qrc_bundle.features.shape[1] != qelm_bundle.features.shape[1]:
        raise ValueError(f"Stage-1 cached bundle width mismatch for {stage1_cell_id}")

    D = int(qrc_bundle.features.shape[1])
    control_d = int(qelm_bundle.config["d"])
    selected = _stage1_selected(stage1_manifest, stage1_cell_id)
    model_matrices: dict[str, tuple[np.ndarray, int, dict[str, Any], int, int]] = {
        "qrc": (
            qrc_bundle.features,
            int(qrc_bundle.first_valid),
            selected.get("qrc", qrc_bundle.config),
            int(qrc_bundle.n_obs),
            int(qrc_bundle.n_multiplex),
        ),
        "qelm": (
            qelm_bundle.features,
            int(qelm_bundle.first_valid),
            selected.get("qelm", qelm_bundle.config),
            int(qelm_bundle.n_obs),
            int(qelm_bundle.n_multiplex),
        ),
        "linear_delay": (
            controls.linear_delay_features(inputs, d=control_d, D=D),
            int(qelm_bundle.first_valid),
            {"d": control_d, "shot_noise_exempt": True},
            int(qrc_bundle.n_obs),
            int(qrc_bundle.n_multiplex),
        ),
        "poly_delay": (
            controls.poly_delay_features(inputs, d=control_d, D=D),
            int(qelm_bundle.first_valid),
            {"d": control_d, "degree": 2, "shot_noise_exempt": True},
            int(qrc_bundle.n_obs),
            int(qrc_bundle.n_multiplex),
        ),
    }

    scored: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for model_name, (features, first_valid, hp, n_obs, n_multiplex) in model_matrices.items():
        if model_name in QUANTUM_SHOT_NOISE_MODELS:
            noisy = shots.add_shot_noise(
                features,
                shot_value,
                _shot_noise_rng(model_name, tier_index, shot_seed),
            )
        else:
            noisy = np.asarray(features, dtype=np.float64).copy()
        scores = pipeline.evaluate_feature_matrix(
            noisy,
            targets,
            splits,
            alphas=alphas,
            first_valid=first_valid,
        ).set_index("target")
        scored[model_name] = scores
        diag = run_stage1.effective_dimension(noisy[splits.train])
        mux = run_stage1._safe_multiplex_gain(noisy[splits.train], n_multiplex, n_obs)
        for target_name, score_row in scores.iterrows():
            meta = target_meta[target_name]
            rows.append(
                {
                    "stage2_item_id": str(item["item_id"]),
                    "cell_id": cell_id,
                    "input_seed": input_seed,
                    "coupling_seed": coupling_seed,
                    "shot_seed": shot_seed,
                    "shots": shot_label,
                    "model": model_name,
                    "target": target_name,
                    "target_family": meta["target_family"],
                    "family_cell": meta["family_cell"],
                    "confirmatory": bool(meta["confirmatory"]),
                    "tau": meta.get("tau"),
                    "p": meta.get("p"),
                    "capacity_test": float(score_row["capacity_test"]),
                    "nrmse_test": float(score_row["nrmse_test"]),
                    "alpha": float(score_row["alpha"]),
                    "effective_rank": float(score_row["effective_rank"]),
                    "effective_rank_diag": float(diag["eff_rank"]),
                    "multiplex_gain": float(mux["gain"]),
                    "D": D,
                    "n_obs": int(n_obs),
                    "n_multiplex": int(n_multiplex),
                    "selected_hp": json.dumps(run_stage1._jsonable(hp), sort_keys=True),
                    "gamma": float(qrc_bundle.config["gamma"]),
                    "uninformative": False,
                    "capacity_linear_delay": np.nan,
                    "capacity_poly_delay": np.nan,
                    "complete": 1,
                }
            )

    rows = run_stage1._attach_control_capacities(rows, scored)
    return _order_stage2_scores(pd.DataFrame(rows), keep_item_id=True)


def _stage1_selected(
    stage1_manifest: Mapping[str, Any],
    stage1_cell_id: str,
) -> dict[str, dict[str, Any]]:
    cells = stage1_manifest["core"]["payload"].get("manifest_state", {}).get("cells", {})
    selected = cells.get(stage1_cell_id, {}).get("selected", {})
    return {
        str(model): dict(value)
        for model, value in selected.items()
        if isinstance(value, Mapping)
    }


def _write_stage2_manifest(
    out_dir: Path,
    *,
    config: Mapping[str, Any],
    seed_grid_name: str,
    shot_tiers: Sequence[Mapping[str, Any]],
    dirty_state: Mapping[str, Any],
    runtime_projection: Mapping[str, Any],
    stage1_manifest: Mapping[str, Any],
    manifest_state: Mapping[str, Any],
    elapsed_s: float,
    scores: pd.DataFrame | None,
) -> Path:
    outputs: dict[str, Any] = {
        "scores": str(out_dir / "stage2_scores.csv"),
        "checkpoints": str(out_dir / "checkpoints"),
    }
    if scores is not None and (out_dir / "stage2_scores.csv").exists():
        outputs["stage2_scores_sha256"] = sha256_file(out_dir / "stage2_scores.csv")
        outputs["stage2_scores_frame_sha256"] = sha256_frame(scores)
        outputs["n_score_rows"] = int(len(scores))

    return write_manifest(
        out_dir,
        "stage2",
        {
            "repo_root": ROOT,
            "stage": 2,
            "profile": config["profile"],
            "config": run_stage1._jsonable(config),
            "seed_grid": grid_manifest(seed_grid_name),
            "shot_tiers": [tier["label"] for tier in shot_tiers],
            "shot_noise": _shot_noise_protocol(),
            "stage1": {
                "manifest_core_sha256": stage1_manifest.get("core_sha256"),
                "profile": stage1_manifest["core"]["payload"].get("profile"),
                "outputs": stage1_manifest["core"]["payload"].get("outputs", {}),
            },
            "runtime_projection": run_stage1._jsonable(runtime_projection),
            "runtime": {"elapsed_s": float(elapsed_s)},
            "dirty_state": dict(dirty_state),
            "manifest_state": run_stage1._jsonable(manifest_state),
            "outputs": outputs,
        },
    )


def _shot_noise_protocol() -> dict[str, Any]:
    return {
        "applies_to": sorted(QUANTUM_SHOT_NOISE_MODELS),
        "exempt_controls": ["linear_delay", "poly_delay"],
        "rng": "SeedSequence([shot_seed, tier_index, stable_model_token])",
    }


def _invalidate_stage2_checkpoints_if_stale(
    out_dir: Path,
    shot_noise_protocol: Mapping[str, Any],
) -> None:
    manifest_path = out_dir / "stage2_manifest.json"
    existing_protocol: Any = None
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_protocol = payload["core"]["payload"].get("shot_noise")
        except Exception:
            existing_protocol = None
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_files = (
        list(checkpoint_dir.glob(f"{STAGE2_SCORE_STEM}_chunk_*.csv"))
        if checkpoint_dir.exists()
        else []
    )
    if existing_protocol == dict(shot_noise_protocol):
        return
    for path in [
        out_dir / f"{STAGE2_SCORE_STEM}.csv",
        out_dir / f"{STAGE2_SCORE_STEM}_partial.csv",
        *checkpoint_files,
    ]:
        if path.exists():
            path.unlink()


def _shot_noise_rng(model_name: str, tier_index: int, shot_seed: int) -> np.random.Generator:
    token = _stable_model_token(model_name)
    seed = np.random.SeedSequence([int(shot_seed), int(tier_index), int(token)])
    return np.random.default_rng(seed)


def _stable_model_token(model_name: str) -> int:
    digest = hashlib.sha256(str(model_name).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _write_scores(scores: pd.DataFrame, path: Path) -> None:
    _order_stage2_scores(scores, keep_item_id=False).to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )


def _order_stage2_scores(
    scores: pd.DataFrame,
    *,
    keep_item_id: bool,
) -> pd.DataFrame:
    columns = [*STAGE2_SCORE_COLUMNS]
    if keep_item_id:
        columns.append("stage2_item_id")
    if scores.empty:
        return pd.DataFrame(columns=columns)
    out = scores.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    ordered = out.loc[:, columns].sort_values(
        ["cell_id", "shots", "target_family", "target", "model"],
        kind="mergesort",
    )
    return ordered.reset_index(drop=True)


def main_cli() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(main())
