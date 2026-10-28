"""PyTorch training pipeline supporting binary, 3-class, and regression objectives.
Includes early stopping, learning rate scheduling, and adversarial TimeGAN training."""

import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config

if not torch.cuda.is_available():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def make_criterion(task, y_train, device):
    if task == "binary":
        pos = float(np.sum(y_train))
        neg = float(len(y_train) - pos)
        pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0], device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if task == "direction3":
        y = np.asarray(y_train)
        counts = np.array([(y == k).sum() for k in range(3)], dtype=float)
        counts[counts == 0] = 1.0
        weights = len(y) / (3.0 * counts)
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    if task == "regression":
        return nn.MSELoss()
    raise ValueError(f"Unknown task: {task}")


def _raw_outputs(model, X, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        for i in range(0, len(xb), 256):
            chunks.append(model(xb[i:i + 256]).cpu().numpy())
    return np.concatenate(chunks)


def _val_metrics(task, out, y_val):
    if task == "binary":
        p = 1.0 / (1.0 + np.exp(-np.clip(out, -50, 50)))
        val_loss = float(nn.functional.binary_cross_entropy(
            torch.from_numpy(p.astype(np.float32)),
            torch.from_numpy(np.asarray(y_val, dtype=np.float32)),
        ).item())
        val_acc = float(((p >= 0.5) == np.asarray(y_val)).mean())
        return val_loss, val_acc
    if task == "direction3":
        logits = torch.from_numpy(out.astype(np.float32))
        y = torch.from_numpy(np.asarray(y_val, dtype=np.int64))
        val_loss = float(nn.functional.cross_entropy(logits, y).item())
        val_acc = float((out.argmax(axis=1) == np.asarray(y_val)).mean())
        return val_loss, val_acc
    pred = out.reshape(-1)
    y = np.asarray(y_val, dtype=np.float64)
    val_loss = float(np.mean((pred - y) ** 2))
    val_acc = float(-np.mean(np.abs(pred - y)))
    return val_loss, val_acc


def train_model(model, X_train, y_train, X_val, y_val, task="binary",
                max_epochs=config.MAX_EPOCHS, patience=config.PATIENCE,
                min_delta=config.MIN_DELTA, lr=config.LEARNING_RATE,
                batch_size=config.BATCH_SIZE, device=None, verbose=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    is_gan = hasattr(model, "discriminator") and hasattr(model, "generator")
    criterion = make_criterion(task, y_train, device)

    if is_gan:
        opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr * 0.5)
        d_criterion = nn.BCEWithLogitsLoss()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    y_train_arr = np.asarray(y_train)
    train_loader = _loader(np.asarray(X_train), y_train_arr.astype(np.float32)
                           if task == "binary" else y_train_arr, batch_size, True)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            if is_gan:
                opt_d.zero_grad()
                gen_y = model.generator(xb).detach()
                real_target = yb if task == "regression" else (yb.float() if task == "binary" else nn.functional.one_hot(yb.long(), 3).float())
                fake_target = gen_y if task == "regression" else (torch.sigmoid(gen_y) if task == "binary" else torch.softmax(gen_y, dim=-1))

                d_real = model.discriminator(xb, real_target)
                d_fake = model.discriminator(xb, fake_target)
                loss_d = d_criterion(d_real, torch.ones_like(d_real)) + d_criterion(d_fake, torch.zeros_like(d_fake))
                loss_d.backward()
                opt_d.step()

                opt_g.zero_grad()
                pred_y = model.generator(xb)
                fake_for_g = pred_y if task == "regression" else (torch.sigmoid(pred_y) if task == "binary" else torch.softmax(pred_y, dim=-1))
                d_fake_for_g = model.discriminator(xb, fake_for_g)
                adv_loss = d_criterion(d_fake_for_g, torch.ones_like(d_fake_for_g))
                sup_loss = criterion(pred_y, yb)
                loss_g = sup_loss + 0.1 * adv_loss
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
                opt_g.step()

                loss = sup_loss
            else:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total += loss.item() * len(xb)
            count += len(xb)

        val_out = _raw_outputs(model, X_val, device)
        val_loss, val_acc = _val_metrics(task, val_out, y_val)

        history["train_loss"].append(total / count)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} train_loss={history['train_loss'][-1]:.4f} "
                  f"val_loss={val_loss:.4f} val_metric={val_acc:.4f}")

        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    history["best_epoch"] = best_epoch
    history["epochs_run"] = len(history["train_loss"])
    return model, history


def predict_outputs(model, X, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return _raw_outputs(model, X, device)


def predict_proba(model, X, device=None):
    out = predict_outputs(model, X, device)
    return 1.0 / (1.0 + np.exp(-np.clip(out, -50, 50)))


def to_probabilities(task, out):
    out = np.asarray(out, dtype=np.float64)
    if task == "binary":
        return 1.0 / (1.0 + np.exp(-np.clip(out, -50, 50)))
    if task == "direction3":
        e = np.exp(out - out.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return out.reshape(-1)


def predict_probs(model, X, task="binary", device=None):
    return to_probabilities(task, predict_outputs(model, X, device))
