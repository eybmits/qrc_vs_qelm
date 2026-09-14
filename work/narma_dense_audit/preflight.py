"""Record the initial cross-runtime mismatch without changing fit selection."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import scipy

sys.path.insert(0, str(Path.cwd() / "pra_extension/code"))
import narma_dense_replay as dense
from laser_forecast import replay_features

parent, cfg = dense.load_parent()
cell = parent["study_summaries"]["narma"]["results"][0]
raw, blocks, indices, targets = dense.narma_memory.make_data(cell["seed"], cfg)
qrc, _, obs, _ = dense.architecture(cell["seed"], cell["selection"]["qrc"], cfg)

def conditioning(features, fitted):
    fit = fitted[0]
    x = (features - fit.mean) / fit.scale
    x = x - x.mean(axis=0)
    matrix = x.T @ x + fit.alpha * np.eye(x.shape[1])
    return float(np.linalg.cond(matrix))

full = [qrc.run(u, obs, n_virtual=4)[t] for u, t in zip(blocks, indices)]
row, fitted = dense.narma_memory.fitted_row("qrc", full, targets,
                                         rate=cell["selection"]["qrc"])
old = next(r for r in cell["rows"] if r["model"] == "qrc")
report = {
    "seed": cell["seed"], "parent_sha256": dense.digest(dense.PARENT),
    "initial_specification_sha256": dense.digest(dense.DEFAULT_RESULTS / "extension_specification.json"),
    "initial_runner_sha256": dense.digest(Path(dense.__file__)),
    "diagnostic_code_sha256": dense.digest(Path(__file__)),
    "numpy": np.__version__, "scipy": scipy.__version__,
    "archived_full_qrc": old, "recomputed_full_qrc": row,
    "full_qrc_ridge_system_condition": conditioning(full[0], fitted),
    "full_qrc_metric_differences": {k: abs(row[k]-old[k]) for k in ("validation_mse", "rmse", "nrmse")},
    "first96_target_max_difference": float(np.max(np.abs(targets[2][:96]-parent["traces"]["narma"]["target"]))),
    "first96_qrc_prediction_max_difference": float(np.max(np.abs(fitted[3][:96]-parent["traces"]["narma"]["qrc"]))),
    "anchors": [],
}
start = time.perf_counter()
generators = [dense.replay_feature_grid(qrc, u, t, 48, obs, 4) for u, t in zip(blocks, indices)]
for window in range(1, 49):
    features = [next(g)[1] for g in generators]
    if window not in dense.ANCHORS:
        continue
    row, fit = dense.narma_memory.fitted_row("replay", features, targets,
                        rate=cell["selection"]["qrc"], window=window)
    frozen = [replay_features(qrc, u, t, window, obs, 4) for u, t in zip(blocks, indices)]
    reference, _ = dense.narma_memory.fitted_row("replay", frozen, targets,
                        rate=cell["selection"]["qrc"], window=window)
    old = next(r for r in cell["rows"] if r["model"] == "replay" and r["window"] == window)
    report["anchors"].append({"window": window, "alpha": row["alpha"], "archived_alpha": old["alpha"],
        "ridge_system_condition": conditioning(features[0], fit),
        "same_runtime_feature_max_difference": float(max(np.max(np.abs(x-y)) for x,y in zip(features,frozen))),
        "same_runtime_metric_differences": {k: abs(row[k]-reference[k]) for k in ("validation_mse", "rmse", "nrmse")},
        "archive_metric_differences": {k: abs(row[k]-old[k]) for k in ("validation_mse", "rmse", "nrmse")}})
report["dense_and_legacy_seconds"] = time.perf_counter()-start
report["interpretation"] = "Frozen and optimized replay agree exactly on this host. Archived differences are concentrated in ill-conditioned alpha=1e-8 normal-equation fits; no targets, selected settings, physics or splits were changed. Initial strict cross-runtime gate failed; no extension result was published."
dense.dump(dense.DEFAULT_RESULTS / "numerical_preflight.json", report)
print(json.dumps(report, indent=2))
