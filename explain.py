"""Integrated Gradients XAI attribution engine for financial sequence models.
Computes signed feature drivers and temporal lookback saliency in PyTorch."""

import numpy as np
import torch
from torch import nn


def integrated_gradients(model: nn.Module, X: np.ndarray, target_class: int = None,
                         steps: int = 25, device: str = "cpu") -> np.ndarray:
    model = model.to(device)
    model.eval()

    if X.ndim == 2:
        X = X[None, ...]

    x_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    baseline = torch.zeros_like(x_tensor)

    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)
    grads = []

    for alpha in alphas:
        x_step = baseline + alpha * (x_tensor - baseline)
        x_step.requires_grad_(True)
        out = model(x_step)

        if out.ndim > 1 and out.shape[-1] > 1:
            cls_idx = target_class if target_class is not None else int(out.argmax(dim=-1).item())
            score = out[:, cls_idx]
        else:
            score = out.squeeze()

        score.backward()
        grads.append(x_step.grad.detach())

    avg_grads = torch.stack(grads).mean(dim=0)
    attributions = (x_tensor - baseline) * avg_grads
    return attributions.squeeze(0).cpu().numpy()


def compute_xai_summary(model: nn.Module, X: np.ndarray, feature_names: list,
                        target_class: int = None, steps: int = 25) -> dict:
    attr_2d = integrated_gradients(model, X, target_class=target_class, steps=steps)

    temporal_attr = np.sum(np.abs(attr_2d), axis=1)
    temporal_attr_pct = temporal_attr / (np.sum(temporal_attr) + 1e-8) * 100.0
    feat_attr = np.sum(attr_2d, axis=0)

    indices = np.argsort(np.abs(feat_attr))[::-1]
    ranked_features = []
    for idx in indices:
        f_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        val = float(feat_attr[idx])
        ranked_features.append({
            "feature": f_name,
            "attribution": val,
            "abs_impact": abs(val),
            "direction": "positive (+)" if val > 0 else "negative (-)"
        })

    return {
        "attributions_2d": attr_2d,
        "temporal_importance_pct": temporal_attr_pct,
        "ranked_features": ranked_features,
    }
