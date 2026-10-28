"""Evaluation metrics suite for binary classification, multi-class regimes, and price regression.
Calculates accuracy, ROC-AUC, Balanced F1, MAE, MAPE, and Youden's J optimal threshold."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def binary_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": (roc_auc_score(y_true, y_prob)
                    if len(np.unique(y_true)) > 1 else float("nan")),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "pred_up_rate": float(y_pred.mean()),
        "threshold": float(threshold),
    }


def majority_baseline(y_train, n_samples):
    majority = int(np.mean(y_train) >= 0.5)
    return np.full(n_samples, float(majority)), majority


def tune_threshold(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = np.unique(np.clip(y_prob, 1e-4, 1 - 1e-4))
    best_j, best_t = -np.inf, 0.5
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        tn = float(((pred == 0) & (y_true == 0)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        if tp + fn == 0 or tn + fp == 0:
            continue
        j = tp / (tp + fn) - fp / (fp + tn)
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def fit_logistic_regression(X_train, y_train, seed=42):
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X_train.reshape(len(X_train), -1), y_train.astype(int))
    return clf


def logistic_proba(clf, X):
    return clf.predict_proba(X.reshape(len(X), -1))[:, 1]


def majority_baseline3(y_train) -> int:
    y = np.asarray(y_train).astype(int)
    return int(np.bincount(y, minlength=3).argmax())


def multiclass_metrics(y_true, y_pred, labels=(0, 1, 2)) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=list(labels),
                             zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted",
                                labels=list(labels), zero_division=0),
        "balanced_accuracy": float(np.mean([cm[k, k] / max(cm[k].sum(), 1)
                                            for k in range(len(labels))])),
        "pred_up_rate": float(np.mean(y_pred == labels[-1])),
    }
    names = ("down", "flat", "up") if len(labels) == 3 else tuple(map(str, labels))
    for k, name in enumerate(names):
        tp = cm[k, k]
        out[f"precision_{name}"] = float(tp / max(cm[:, k].sum(), 1))
        out[f"recall_{name}"] = float(tp / max(cm[k].sum(), 1))
        out[f"f1_{name}"] = float(2 * tp / max(cm[:, k].sum() + cm[k].sum(), 1))
    for i in range(len(labels)):
        for j in range(len(labels)):
            out[f"cm_{i}{j}"] = int(cm[i, j])
    return out


def regression_metrics(close, r_true, r_pred) -> dict:
    close = np.asarray(close, dtype=np.float64)
    r_true = np.asarray(r_true, dtype=np.float64)
    r_pred = np.asarray(r_pred, dtype=np.float64)
    price_true = close * (1.0 + r_true)
    price_pred = close * (1.0 + r_pred)
    err = price_pred - price_true
    return {
        "mae_price": float(np.mean(np.abs(err))),
        "rmse_price": float(np.sqrt(np.mean(err ** 2))),
        "mape_pct": float(np.mean(np.abs(err) / price_true) * 100.0),
        "mae_return_pct": float(np.mean(np.abs(r_pred - r_true)) * 100.0),
        "rmse_return_pct": float(np.sqrt(np.mean((r_pred - r_true) ** 2)) * 100.0),
        "directional_accuracy": float(np.mean(np.sign(r_pred) == np.sign(r_true))),
        "pred_mean_return_pct": float(r_pred.mean() * 100.0),
        "actual_mean_return_pct": float(r_true.mean() * 100.0),
    }


def evaluate_task(task, y_true, output, close=None) -> dict:
    if task == "binary":
        return binary_metrics(y_true, output)
    if task == "direction3":
        return multiclass_metrics(y_true, np.argmax(output, axis=1))
    if task == "regression":
        return regression_metrics(close, y_true, output)
    raise ValueError(f"Unknown task: {task}")
