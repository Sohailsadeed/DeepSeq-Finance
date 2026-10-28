"""Batch multi-task training orchestrator across all 6 deep learning sequence architectures.
Saves PyTorch weights to artifacts/<asset>/ and exports benchmark metrics to Excel."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch

import config
import evaluate
import prepare_data
import train
from models import MODEL_REGISTRY

TASK_OBJECTIVE = {
    "binary": ("roc_auc", "max"),
    "direction3": ("f1_macro", "max"),
    "regression": ("mae_price", "min"),
}


def train_asset(asset: str, raw, tasks=config.TASKS, verbose=True, progress_cb=None):
    asset_dir = config.ARTIFACTS_DIR / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "asset": asset,
        "created": datetime.now().isoformat(timespec="seconds"),
        "window": config.WINDOW,
        "band": config.FLAT_BAND,
        "split_ratios": list(config.SPLIT_RATIOS),
        "seed": config.SEED,
        "interval": "1 day",
        "tasks": {},
    }

    for task in tasks:
        data = prepare_data.build_task(asset, raw, task)
        n_features = data["X_train"].shape[-1]
        entry = {
            "features": [str(f) for f in data["meta_features"]],
            "scaler_mean": data["meta_scaler_mean"].tolist(),
            "scaler_scale": data["meta_scaler_scale"].tolist(),
            "n_rows": int(data["meta_n_rows_total"]),
            "last_date": str(data["dates_test"][-1]),
            "splits": {
                s: {"range": [str(data[f"meta_{s}_range"][0]),
                              str(data[f"meta_{s}_range"][1])],
                    "start": str(data[f"dates_{s}"][0]),
                    "end": str(data[f"dates_{s}"][-1]),
                    "n": int(len(data[f"y_{s}"]))}
                for s in ("train", "val", "test")
            },
            "class_share_train": (data["meta_class_share_train"].tolist()
                                  if "meta_class_share_train" in data else None),
            "models": {},
        }

        out_dim = 3 if task == "direction3" else 1
        metric_name, direction = TASK_OBJECTIVE[task]
        best_name, best_value = None, None

        for arch, ctor in MODEL_REGISTRY.items():
            train.set_seed(config.SEED)
            model = ctor(n_features, out_dim=out_dim)
            t0 = time.time()
            model, history = train.train_model(
                model, data["X_train"], data["y_train"],
                data["X_val"], data["y_val"], task=task, verbose=False,
            )
            seconds = round(time.time() - t0, 1)

            val_out = train.to_probabilities(
                task, train.predict_outputs(model, data["X_val"]))
            test_out = train.to_probabilities(
                task, train.predict_outputs(model, data["X_test"]))
            m_val = evaluate.evaluate_task(task, data["y_val"], val_out,
                                           close=data["close_val"])
            m_test = evaluate.evaluate_task(task, data["y_test"], test_out,
                                            close=data["close_test"])

            model_dir = asset_dir / task
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / f"{arch}.pt")

            entry["models"][arch] = {
                "file": str((model_dir / f"{arch}.pt").relative_to(config.BASE_DIR)),
                "best_epoch": history["best_epoch"],
                "epochs": history["epochs_run"],
                "seconds": seconds,
                "val": {k: round(float(v), 6) for k, v in m_val.items()},
                "test": {k: round(float(v), 6) for k, v in m_test.items()},
            }

            score = m_val.get(metric_name)
            if score is not None:
                better = ((best_value is None) or
                          (direction == "max" and score > best_value) or
                          (direction == "min" and score < best_value))
                if better:
                    best_name, best_value = arch, score

            if verbose:
                print(f"  [{asset}/{task}] {arch}: val {metric_name}="
                      f"{m_val.get(metric_name):.4f} test="
                      f"{m_test.get(metric_name):.4f} ({seconds}s, "
                      f"best_epoch={history['best_epoch']})")
            if progress_cb:
                progress_cb(task, arch)

        entry["best_model"] = best_name
        entry["objective"] = metric_name
        meta["tasks"][task] = entry

    (asset_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def overview_rows(meta: dict) -> list:
    rows = []
    for task, entry in meta["tasks"].items():
        metric = entry["objective"]
        for arch, m in entry["models"].items():
            rows.append({
                "asset": meta["asset"], "task": task, "model": arch,
                "is_best": arch == entry["best_model"],
                "objective": metric,
                "val_objective": m["val"].get(metric),
                "test_objective": m["test"].get(metric),
                "best_epoch": m["best_epoch"], "epochs": m["epochs"],
                "seconds": m["seconds"],
            })
    return rows


def metrics_rows(meta: dict) -> list:
    rows = []
    for task, entry in meta["tasks"].items():
        for arch, m in entry["models"].items():
            for split in ("val", "test"):
                rows.append({"asset": meta["asset"], "task": task, "model": arch,
                             "split": split, **m[split]})
    return rows


def write_excel(metas: list, path=None):
    path = Path(path or config.RESULTS_DIR / "metrics_extended.xlsx")
    path.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame([r for m in metas for r in overview_rows(m)]).to_excel(
            xw, sheet_name="Overview", index=False)
        for task in config.TASKS:
            rows = [r for m in metas for r in metrics_rows(m) if r["task"] == task]
            if rows:
                pd.DataFrame(rows).round(4).to_excel(xw, sheet_name=task, index=False)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Train all tasks for an asset")
    p.add_argument("--csv", default=str(config.DATA_PATH))
    p.add_argument("--asset", default="BTC")
    p.add_argument("--tasks", nargs="+", default=list(config.TASKS),
                   choices=list(config.TASKS))
    args = p.parse_args(argv)

    print(f"Loading {args.csv} as '{args.asset}'")
    raw = prepare_data.load_csv(args.csv)
    print(f"{len(raw)} rows: {raw['Open time'].iloc[0].date()} -> "
          f"{raw['Open time'].iloc[-1].date()}")

    metas = []
    print(f"\n=== Training {args.asset} ===")
    metas.append(train_asset(args.asset, raw, args.tasks))

    xl = write_excel(metas)
    print(f"\nArtifacts: {config.ARTIFACTS_DIR / args.asset}")
    print(f"Metrics:   {xl}")
    return metas


if __name__ == "__main__":
    main()
