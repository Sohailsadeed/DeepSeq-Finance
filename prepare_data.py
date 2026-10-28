"""Data ingestion, feature engineering, and chronological window generator.
Processes 22 financial microstructure features with zero-lookahead scaling."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config

COLUMN_ALIASES = {
    "open time": "Open time",
    "date": "Open time",
    "datetime": "Open time",
    "timestamp": "Open time",
    "time": "Open time",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Close",
    "adj. close": "Close",
    "volume": "Volume",
    "quote asset volume": "Quote asset volume",
    "quote volume": "Quote asset volume",
    "number of trades": "Number of trades",
    "trades": "Number of trades",
}


def load_csv(path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rename = {}
    for c in raw.columns:
        target = COLUMN_ALIASES.get(str(c).strip().lower())
        if target:
            rename[c] = target
    df = raw.rename(columns=rename)
    if "Volume" not in df.columns:
        vol_c = next((c for c in df.columns if "volume" in str(c).lower() and "quote" not in str(c).lower() and "usdt" not in str(c).lower()), None)
        if not vol_c:
            vol_c = next((c for c in df.columns if "volume" in str(c).lower() or "vol" in str(c).lower()), None)
        if vol_c:
            df = df.rename(columns={vol_c: "Volume"})

    if "Open time" not in df.columns:
        raise ValueError("No date/time column found. Expected one of: "
                         "Open time, Date, Datetime, timestamp, Time")
    missing = [f for f in config.REQUIRED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    base = [f for f in config.BASE_FEATURES if f in df.columns]
    out = df[["Open time"] + base].copy()
    out["Open time"] = pd.to_datetime(out["Open time"], errors="coerce")
    for c in base:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = (out.dropna()
              .sort_values("Open time")
              .drop_duplicates(subset="Open time")
              .reset_index(drop=True))
    if len(out) < 60:
        raise ValueError(f"Only {len(out)} usable rows; need at least 60")
    return out


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    return load_csv(path)


def feature_list(df: pd.DataFrame) -> list:
    base = [f for f in config.BASE_FEATURES if f in df.columns]
    return base + list(config.DERIVED_FEATURES)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    open_p = out["Open"]
    high = out["High"]
    low = out["Low"]
    vol = out["Volume"]

    out["Daily return"] = close.pct_change()
    out["High-low pct range"] = (high - low) / np.maximum(low, 1e-8) * 100.0
    out["MA 7"] = close.rolling(7).mean()
    out["MA 30"] = close.rolling(30).mean()
    out["Rolling vol 7"] = out["Daily return"].rolling(7).std()
    out["Volume change"] = vol.pct_change()

    prev_close = close.shift(1)
    out["Overnight return"] = (open_p - prev_close) / np.maximum(prev_close, 1e-8)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-8)
    out["RSI 14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0) / 100.0

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    band_width = np.maximum(upper - lower, 1e-8)
    out["Bollinger pct b"] = ((close - lower) / band_width).fillna(0.5)

    log_hl = np.log(np.maximum(high / np.maximum(low, 1e-8), 1e-8))
    log_co = np.log(np.maximum(close / np.maximum(open_p, 1e-8), 1e-8))
    gk = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    out["Garman-Klass vol"] = np.sqrt(np.maximum(gk, 0.0))

    dt = pd.to_datetime(out["Open time"])
    hours = dt.dt.hour
    dows = dt.dt.day_of_week
    out["Sin hour"] = np.sin(2.0 * np.pi * hours / 24.0)
    out["Cos hour"] = np.cos(2.0 * np.pi * hours / 24.0)
    out["Sin dayofweek"] = np.sin(2.0 * np.pi * dows / 7.0)
    out["Cos dayofweek"] = np.cos(2.0 * np.pi * dows / 7.0)
    out["Is weekend"] = dows.isin([5, 6]).astype(float)

    return out


def make_target(close: pd.Series, task: str) -> pd.Series:
    next_close = close.shift(-1)
    ret = next_close / close - 1.0
    if task == "binary":
        target = (next_close > close).astype(float)
    elif task == "direction3":
        band = config.FLAT_BAND
        target = pd.Series(np.where(ret > band, 2.0,
                                    np.where(ret < -band, 0.0, 1.0)),
                           index=close.index, dtype=float)
    elif task == "regression":
        target = ret.astype(float)
    else:
        raise ValueError(f"Unknown task: {task}")
    target[next_close.isna()] = np.nan
    return target


def split_indices(n: int, ratios=config.SPLIT_RATIOS):
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    return (0, n_train), (n_train, n_train + n_val), (n_train + n_val, n)


def make_windows(X, y, dates, close, window=config.WINDOW):
    n = len(X)
    if n <= window:
        raise ValueError("Split too small for the requested window")
    starts = range(window - 1, n)
    Xw = np.stack([X[i - window + 1 : i + 1] for i in starts])
    yw = y[window - 1 : n]
    dw = dates[window - 1 : n]
    cw = close[window - 1 : n]
    return Xw, yw, dw, cw


def _day(v):
    return str(pd.Timestamp(v).date())


def build_task(asset: str, raw: pd.DataFrame, task: str, path=None) -> dict:
    df = add_features(raw)
    features = feature_list(raw)
    df = df.dropna(subset=features).reset_index(drop=True)
    df[config.TARGET] = make_target(df["Close"], task)
    df = df.dropna(subset=[config.TARGET]).reset_index(drop=True)

    dates = df["Open time"].to_numpy()
    y = df[config.TARGET].to_numpy()
    y = y.astype(np.int64) if task in ("binary", "direction3") else y.astype(np.float32)
    X = df[features].to_numpy(dtype=np.float32)
    close = df["Close"].to_numpy(dtype=np.float32)

    (a, b), (c, d), (e, f) = split_indices(len(df))
    scaler = StandardScaler().fit(X[a:b])
    Xs = scaler.transform(X).astype(np.float32)

    splits = {
        "train": make_windows(Xs[a:b], y[a:b], dates[a:b], close[a:b]),
        "val": make_windows(Xs[c:d], y[c:d], dates[c:d], close[c:d]),
        "test": make_windows(Xs[e:f], y[e:f], dates[e:f], close[e:f]),
    }

    n_classes = 3 if task == "direction3" else (2 if task == "binary" else 0)
    meta = {
        "asset": np.array(asset),
        "task": np.array(task),
        "features": np.array(features),
        "train_range": np.array([_day(dates[a]), _day(dates[b - 1])]),
        "val_range": np.array([_day(dates[c]), _day(dates[d - 1])]),
        "test_range": np.array([_day(dates[e]), _day(dates[f - 1])]),
        "n_rows_total": np.array(len(df)),
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "band": np.array(config.FLAT_BAND),
        "n_classes": np.array(n_classes),
    }
    for name, arr in (("train", y[a:b]), ("val", y[c:d]), ("test", y[e:f])):
        meta[f"base_rate_{name}"] = np.array(float(np.mean(arr)))
        if n_classes:
            shares = [float(np.mean(arr == k)) for k in range(n_classes)]
            meta[f"class_share_{name}"] = np.array(shares)

    payload = {f"meta_{k}": v for k, v in meta.items()}
    for name, (Xw, yw, dw, cw) in splits.items():
        payload[f"X_{name}"] = Xw
        payload[f"y_{name}"] = yw
        payload[f"dates_{name}"] = dw.astype("datetime64[D]").astype(str)
        payload[f"close_{name}"] = cw

    path = Path(path or config.ARTIFACTS_DIR / asset / f"{task}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    payload["_path"] = path
    return payload


def build():
    return build_task("BTC", load_raw(), "binary", config.PROCESSED_PATH)


def load(path=None):
    return np.load(path or config.PROCESSED_PATH, allow_pickle=False)


def last_window(raw: pd.DataFrame, features, mean, scale,
                window=config.WINDOW) -> np.ndarray:
    df = add_features(raw)
    X = df[features].to_numpy(dtype=np.float32)
    X = X[-window:]
    if len(X) < window or np.isnan(X).any():
        raise ValueError("Not enough clean history to build a full window")
    X = (X - np.asarray(mean, dtype=np.float32)) / np.asarray(scale, dtype=np.float32)
    return X[None, ...].astype(np.float32)


def infer_interval(dates) -> str:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(idx) < 2:
        return "unknown"
    step = idx[-1] - idx[-2]
    secs = int(step.total_seconds())
    for s, label in ((86400, "1 day"), (3600, "1 hour"), (900, "15 minutes"),
                     (14400, "4 hours")):
        if secs == s:
            return label
    if secs % 86400 == 0:
        return f"{secs // 86400} days"
    if secs % 3600 == 0:
        return f"{secs // 3600} hours"
    return f"{secs} seconds"


if __name__ == "__main__":
    data = build()
    for name in ("train", "val", "test"):
        X, y = data[f"X_{name}"], data[f"y_{name}"]
        print(f"{name:5s} X={X.shape} y={y.shape} mean_y={y.mean():.3f} "
              f"{data[f'dates_{name}'][0]} -> {data[f'dates_{name}'][-1]}")
    print("features:", list(data["meta_features"]))
