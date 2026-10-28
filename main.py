"""Part I benchmark orchestrator for baseline binary trend classification on daily Bitcoin.
Evaluates logistic regression and neural models, tuning decision thresholds on validation data."""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
import evaluate
import prepare_data
import train
from models import MODEL_REGISTRY

RESULTS = config.RESULTS_DIR
MODELS_DIR = RESULTS / "models"
HEADLINE = "LSTM"


def get_data():
    if config.PROCESSED_PATH.exists():
        return prepare_data.load()
    return prepare_data.build()


def run_baselines(data):
    y_tr = data["y_train"]
    results = {}

    p_val, majority = evaluate.majority_baseline(y_tr, len(data["y_val"]))
    p_test, _ = evaluate.majority_baseline(y_tr, len(data["y_test"]))
    results["Majority baseline"] = {"val": p_val, "test": p_test, "history": None,
                                    "majority_class": majority}

    clf = evaluate.fit_logistic_regression(data["X_train"], y_tr, seed=config.SEED)
    results["Logistic Regression"] = {
        "val": evaluate.logistic_proba(clf, data["X_val"]),
        "test": evaluate.logistic_proba(clf, data["X_test"]),
        "history": None,
        "coef_shape": clf.coef_.shape,
    }
    return results


def run_recurrent(data, verbose=True):
    results = {}
    n_features = data["X_train"].shape[-1]
    for name, ctor in MODEL_REGISTRY.items():
        if verbose:
            print(f"\n=== {name} ===")
        train.set_seed(config.SEED)
        model = ctor(n_features)
        t0 = time.time()
        model, history = train.train_model(
            model, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], verbose=verbose,
        )
        elapsed = time.time() - t0
        results[name] = {
            "val": train.predict_proba(model, data["X_val"]),
            "test": train.predict_proba(model, data["X_test"]),
            "history": history,
            "train_seconds": elapsed,
        }
        torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")
        if verbose:
            print(f"{name}: best_epoch={history['best_epoch']} "
                  f"epochs={history['epochs_run']} time={elapsed:.1f}s")
    return results


def collect_metrics(results, data):
    rows = []
    for model_name, r in results.items():
        thr = (0.5 if model_name == "Majority baseline"
               else evaluate.tune_threshold(data["y_val"], r["val"]))
        r["threshold"] = thr
        rows.append({"model": model_name, "split": "val",
                     **evaluate.binary_metrics(data["y_val"], r["val"], 0.5)})
        rows.append({"model": model_name, "split": "test",
                     **evaluate.binary_metrics(data["y_test"], r["test"], 0.5)})
        rows.append({"model": model_name, "split": "test_tuned",
                     **evaluate.binary_metrics(data["y_test"], r["test"], thr)})
    return pd.DataFrame(rows)


STATS = ["accuracy", "precision", "recall", "f1", "roc_auc", "threshold",
         "pred_up_rate", "tn", "fp", "fn", "tp"]
SPLITS = ["val", "test", "test_tuned"]
ORDER = [
    "Majority baseline",
    "Logistic Regression",
    "SimpleRNN",
    "LSTM",
    "GRU",
    "LSTM_Attention",
    "TCN",
    "TimeGAN",
]


def summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for model, g in metrics_df.groupby("model", sort=False):
        row = {}
        for split in SPLITS:
            sub = g[g["split"] == split]
            if sub.empty:
                continue
            r = sub.iloc[0]
            for s in STATS:
                row[f"{split}_{s}"] = r[s]
        rows[model] = row
    wide = pd.DataFrame(rows).T
    wide = wide.reindex([m for m in ORDER if m in wide.index])
    return wide.astype(float).round(4)



def predictions_frame(results, data):
    n = len(data["y_test"])
    df = pd.DataFrame({
        "Date": data["dates_test"],
        "Close": data["close_test"],
        "Actual": data["y_test"].astype(int),
    })
    df["Majority baseline"] = results["Majority baseline"]["test"].astype(int)
    for name in results:
        if name == "Majority baseline":
            continue
        probs = results[name]["test"]
        thr = results[name]["threshold"]
        df[f"{name} prob"] = np.round(probs.astype(float), 4)
        df[f"{name} pred (0.5)"] = (probs >= 0.5).astype(int)
        df[f"{name} pred (val thr {thr:.3f})"] = (probs >= thr).astype(int)
    assert len(df) == n
    return df


def plot_training_curves(results):
    names = [n for n in ORDER if n in results and results[n].get("history")]
    if not names:
        return
    cols = 3
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for idx, name in enumerate(names):
        ax = axes[idx]
        h = results[name]["history"]
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], label="Train Loss", lw=1.5)
        ax.plot(epochs, h["val_loss"], label="Val Loss", lw=1.5)
        ax.axvline(h["best_epoch"], color="grey", ls="--", lw=1, label=f"Best ({h['best_epoch']})")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.legend(fontsize=8)
        if idx % cols == 0:
            ax.set_ylabel("BCE Loss", fontsize=9)
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Training & Validation Loss Curves (Early Stopping on Validation Loss)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(metrics_df, data):
    order = [m for m in ORDER if m in metrics_df["model"].values]
    test = metrics_df[metrics_df["split"] == "test_tuned"].set_index("model")
    cols = 4
    rows = (len(order) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for idx, name in enumerate(order):
        ax = axes[idx]
        if name not in test.index:
            ax.axis("off")
            continue
        row = test.loc[name]
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["Pred DOWN", "Pred UP"])
        ax.set_yticks([0, 1], ["Actual DOWN", "Actual UP"])
        ax.set_title(f"{name}\nAcc={row['accuracy']:.3f}, Thr={row['threshold']:.2f}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Confusion Matrices Across All Models — Test Set (Threshold Tuned on Validation)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=150)
    plt.close(fig)



def plot_timeline(results, data):
    dates = pd.to_datetime(data["dates_test"])
    close = data["close_test"]
    actual = data["y_test"].astype(int)
    headline = "TimeGAN" if "TimeGAN" in results else ("LSTM_Attention" if "LSTM_Attention" in results else "LSTM")
    thr = results[headline]["threshold"]
    pred = (results[headline]["test"] >= thr).astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(dates, close, color="black", lw=1.2, label="Close (USDT)")
    up = pred == 1
    ax.scatter(dates[up], close[up], marker="^", s=18, color="tab:green",
               label=f"{headline} predicted UP", zorder=3)
    ax.scatter(dates[~up], close[~up], marker="v", s=18, color="tab:red",
               label=f"{headline} predicted DOWN", zorder=3)
    ax.set_ylabel("Close (USDT)")
    ax.set_title(f"Next-day trend classification — {headline} on the test set "
                 f"(threshold {thr:.2f}, tuned on validation)")
    ax.legend(fontsize=8, loc="upper left")

    ax2 = axes[1]
    correct = (pred == actual).astype(int)
    ax2.bar(dates, correct, color=["tab:green" if c else "tab:red" for c in correct], width=1.0)
    ax2.set_ylim(0, 1.6)
    ax2.set_yticks([0, 1], ["wrong", "correct"])
    ax2.set_ylabel("hit")
    ax2.set_xlabel("date")
    fig.tight_layout()
    fig.savefig(RESULTS / "test_timeline.png", dpi=150)
    plt.close(fig)


def write_excel(summary, metrics_df, preds, data, results):
    config_sheet = pd.DataFrame({
        "item": [
            "dataset", "n_rows_used", "window", "split train", "split val", "split test",
            "features", "target", "seed", "hidden size", "dropout", "learning rate",
            "batch size", "max epochs", "patience",             "baseline majority class",
            "train up-rate", "test up-rate",
            "thresholds (val-tuned)",
        ],
        "value": [
            config.DATA_PATH.name, int(data["meta_n_rows_total"]), config.WINDOW,
            " ".join(data["meta_train_range"]), " ".join(data["meta_val_range"]),
            " ".join(data["meta_test_range"]),
            ", ".join(data["meta_features"]),
            "1 = next day Close up, 0 = down", config.SEED, config.HIDDEN_SIZE,
            config.DROPOUT, config.LEARNING_RATE, config.BATCH_SIZE,
            config.MAX_EPOCHS, config.PATIENCE,
            results["Majority baseline"]["majority_class"],
            round(float(data["meta_base_rate_train"]), 4),
            round(float(data["meta_base_rate_test"]), 4),
            "; ".join(f"{k}: {v['threshold']:.3f}" for k, v in results.items()),
        ],
    })

    history_rows = []
    for name in ("SimpleRNN", "LSTM", "GRU"):
        h = results[name]["history"]
        for i in range(h["epochs_run"]):
            history_rows.append({
                "model": name, "epoch": i + 1,
                "train_loss": h["train_loss"][i], "val_loss": h["val_loss"][i],
                "val_acc": h["val_acc"][i],
            })
    history_df = pd.DataFrame(history_rows)

    with pd.ExcelWriter(RESULTS / "metrics.xlsx", engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary")
        metrics_df.round(4).to_excel(xw, sheet_name="All metrics", index=False)
        preds.to_excel(xw, sheet_name="Test predictions", index=False)
        config_sheet.to_excel(xw, sheet_name="Config", index=False)
        history_df.round(5).to_excel(xw, sheet_name="Training history", index=False)


def main(verbose=True):
    RESULTS.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    data = get_data()
    if verbose:
        for s in ("train", "val", "test"):
            print(f"{s:5s} {data[f'X_{s}'].shape} up_rate={data[f'y_{s}'].mean():.3f} "
                  f"{data[f'dates_{s}'][0]} -> {data[f'dates_{s}'][-1]}")

    results = run_baselines(data)
    results.update(run_recurrent(data, verbose=verbose))

    metrics_df = collect_metrics(results, data)
    summary = summary_table(metrics_df)
    preds = predictions_frame(results, data)

    if verbose:
        print("\n=== Summary ===")
        cols = [f"{s}_{m}" for s in SPLITS
                for m in ("accuracy", "precision", "recall", "f1", "roc_auc")]
        print(summary[[c for c in cols if c in summary.columns]].to_string())

    plot_training_curves(results)
    plot_confusion_matrices(metrics_df, data)
    plot_timeline(results, data)
    write_excel(summary, metrics_df, preds, data, results)

    metrics_df.to_csv(RESULTS / "metrics.csv", index=False)
    preds.to_csv(RESULTS / "test_predictions.csv", index=False)

    if verbose:
        print(f"\nOutputs written to {RESULTS}")
    return summary, metrics_df, preds


if __name__ == "__main__":
    main()
