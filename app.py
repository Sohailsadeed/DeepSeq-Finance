"""Interactive Streamlit dashboard for financial forecasting, XAI, and trade suggestions.
Features live inference, multi-interval switching, attention maps, and price trajectories."""

import hashlib
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
import explain
import prepare_data
import train
import train_all
from models import MODEL_REGISTRY

st.set_page_config(page_title="Trend & Price Predictor", layout="wide",
                   page_icon="📈")

BUNDLED = {
    "BTC daily (bundled)": ("BTC", str(config.DATA_PATH)),
    "BTC 4-hour (Crypto)": ("BTC_4H", str(config.BASE_DIR / "datasets" / "btc_4h_cdd.csv")),
    "ETH daily (Crypto)": ("ETH_1D", str(config.BASE_DIR / "datasets" / "eth_1d_cdd.csv")),
    "ETH 4-hour (Crypto)": ("ETH_4H", str(config.BASE_DIR / "datasets" / "eth_4h_cdd.csv")),
    "SPY daily (Stock / S&P 500)": ("SPY_1D", str(config.BASE_DIR / "datasets" / "spy_1d_data.csv")),
    "SPY weekly (Stock / S&P 500)": ("SPY_1W", str(config.BASE_DIR / "datasets" / "spy_1w_data.csv")),
    "AAPL daily (Stock / Apple)": ("AAPL_1D", str(config.BASE_DIR / "datasets" / "aapl_1d_data.csv")),
    "AAPL weekly (Stock / Apple)": ("AAPL_1W", str(config.BASE_DIR / "datasets" / "aapl_1w_data.csv")),
}
# Only keep entries that exist on disk
BUNDLED = {k: v for k, v in BUNDLED.items() if Path(v[1]).exists()}
UPLOAD_LABEL = "Upload a CSV"
UPLOAD_DIR = config.ARTIFACTS_DIR / "uploads"
TASKS = list(config.TASKS)

st.markdown(
    """
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- utilities
def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:40]


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:10]


@st.cache_resource(show_spinner=False)
def load_meta(asset: str) -> dict:
    path = config.ARTIFACTS_DIR / asset / "meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_models(asset: str) -> dict:
    """Returns {(task, arch): nn.Module} for every persisted model."""
    import torch
    meta = load_meta(asset)
    if meta is None:
        return {}
    out = {}
    for task, entry in meta["tasks"].items():
        out_dim = 3 if task == "direction3" else 1
        for arch in entry["models"]:
            model = MODEL_REGISTRY[arch](len(entry["features"]), out_dim=out_dim)
            state = torch.load(config.ARTIFACTS_DIR / asset / task / f"{arch}.pt",
                               map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            out[(task, arch)] = model
    return out


def resolve_arch(meta: dict, task: str, choice: str) -> str:
    entry = meta["tasks"][task]
    if choice != "Best (auto)" and choice in entry["models"]:
        return choice
    return entry["best_model"]


def ensure_trained(asset: str, csv_path: str) -> dict | None:
    meta = load_meta(asset)
    if meta is not None:
        return meta
    raw = prepare_data.load_csv(csv_path)
    if len(raw) < 120:
        st.error(f"Not enough data: {len(raw)} rows, need at least 120.")
        return None
    # CPU optimization: for large intraday datasets, train on the most recent 3,500 bars
    if len(raw) > 3500:
        raw = raw.tail(3500).reset_index(drop=True)
    progress = st.progress(0.0, text="Training models…")
    done = {"n": 0}
    total = len(TASKS) * len(MODEL_REGISTRY)

    def cb(task, arch):
        done["n"] += 1
        progress.progress(done["n"] / total,
                          text=f"Training {task} / {arch} "
                               f"({done['n']}/{total})")

    try:
        train_all.train_asset(asset, raw, TASKS, verbose=False, progress_cb=cb)
    except Exception as exc:  # pragma: no cover - surfaced in the UI
        progress.empty()
        st.error(f"Training failed: {exc}")
        return None
    progress.empty()
    load_meta.clear()
    load_models.clear()
    return load_meta(asset)


def predict(df: pd.DataFrame, meta: dict, models: dict, task: str, arch: str):
    entry = meta["tasks"][task]
    X = prepare_data.last_window(
        df, entry["features"], entry["scaler_mean"], entry["scaler_scale"],
        window=meta["window"])
    raw_out = train.predict_outputs(models[(task, arch)], X)
    return train.to_probabilities(task, raw_out)[0]


def suggest(dir_probs, ret_pred, p_thr, ret_thr):
    """Combined BUY/SELL/HOLD rule: class probability AND predicted return."""
    down, flat, up = (float(dir_probs[0]), float(dir_probs[1]),
                      float(dir_probs[2]))
    confidence = max(down, flat, up)
    if up >= p_thr and up >= max(down, flat) and ret_pred > ret_thr:
        return "BUY", confidence, (
            f"P(UP)={up:.2f} ≥ {p_thr:.2f} and predicted return "
            f"{ret_pred:+.2%} is positive")
    if down >= p_thr and down >= max(up, flat) and ret_pred < -ret_thr:
        return "SELL", confidence, (
            f"P(DOWN)={down:.2f} ≥ {p_thr:.2f} and predicted return "
            f"{ret_pred:+.2%} is negative")
    return "HOLD", confidence, (
        f"strongest class prob {confidence:.2f} < {p_thr:.2f} or predicted "
        f"return {ret_pred:+.2%} does not confirm the direction")


def banner(action: str, confidence: float, rationale: str):
    colors = {"BUY": "#1a7f37", "SELL": "#b42318", "HOLD": "#667085"}
    st.markdown(
        f"""<div style="background:{colors[action]};color:white;padding:16px 20px;
        border-radius:12px;text-align:center;margin:8px 0">
        <div style="font-size:13px;opacity:.9">Model suggestion
        (not financial advice)</div>
        <div style="font-size:38px;font-weight:700;letter-spacing:2px">{action}</div>
        <div style="font-size:13px">confidence {confidence:.2f} · {rationale}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def price_chart(df: pd.DataFrame, price_pred: float, title: str, horizon_label: str):
    tail = df.tail(180).copy()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(tail["Open time"], tail["Close"], lw=1.3, color="#1f77b4", label="Close")
    last_date = df["Open time"].iloc[-1]
    delta = pd.Timedelta({"1 day": "1D", "15 minutes": "15min",
                          "1 hour": "1h", "4 hours": "4h"}
                         .get(horizon_label, "1D"))
    ax.scatter([last_date + delta], [price_pred], marker="*", s=220,
               color="#d62728", zorder=5, label=f"predicted {horizon_label} close")
    ax.scatter([last_date], [df["Close"].iloc[-1]], marker="o", s=45,
               color="black", zorder=5, label="last close")
    ax.axvline(last_date, color="grey", ls="--", lw=1)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(title)
    ax.set_ylabel("price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)


def metrics_block(meta: dict, task: str, arch: str):
    m = meta["tasks"][task]["models"][arch]["test"]
    if task == "binary":
        cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        vals = [f"{m[c]:.4f}" for c in cols]
    elif task == "direction3":
        cols = ["accuracy", "f1_macro", "f1_weighted", "balanced_accuracy"]
        labels = ["Accuracy", "F1 (macro)", "F1 (weighted)", "Balanced acc."]
        vals = [f"{m[c]:.4f}" for c in cols]
    else:
        cols = ["mae_price", "mape_pct", "rmse_price", "directional_accuracy"]
        labels = ["MAE (price)", "MAPE %", "RMSE (price)", "Directional acc."]
        vals = [f"{m[c]:,.2f}" if "price" in c else f"{m[c]:.4f}" for c in cols]
    st.caption(f"Test-set metrics — {task} · {arch}")
    cols_ui = st.columns(len(labels))
    for c, lab, v in zip(cols_ui, labels, vals):
        c.metric(lab, v)


def recent_table(df, meta, models, task, arch, n=10):
    npz_path = config.ARTIFACTS_DIR / meta["asset"] / f"{task}.npz"
    if not Path(npz_path).exists():
        return
    data = np.load(npz_path, allow_pickle=False)
    outs = train.to_probabilities(
        task, train.predict_outputs(models[(task, arch)], data["X_test"]))
    dates = data["dates_test"][-n:]
    rows = []
    for i, d in enumerate(dates):
        j = len(outs) - n + i
        if task == "regression":
            p_close = float(data["close_test"][j] * (1.0 + outs[j]))
            rows.append({"date": d, "actual close": float(data["close_test"][j]),
                         "predicted close": round(p_close, 2)})
        elif task == "direction3":
            rows.append({"date": d,
                         "actual": config.DIRECTION3_LABELS[int(data["y_test"][j])],
                         "predicted": config.DIRECTION3_LABELS[int(np.argmax(outs[j]))],
                         "P(UP)": round(float(outs[j][2]), 3)})
        else:
            rows.append({"date": d,
                         "actual": "UP" if data["y_test"][j] == 1 else "DOWN",
                         "predicted": "UP" if outs[j] >= 0.5 else "DOWN",
                         "P(UP)": round(float(outs[j]), 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_prediction(df: pd.DataFrame, meta: dict, models: dict,
                      horizon_label: str, arch_choice: str,
                      p_thr: float, ret_thr: float):
    """Renders the full prediction panel for one tab."""
    d3_arch = resolve_arch(meta, "direction3", arch_choice)
    reg_arch = resolve_arch(meta, "regression", arch_choice)
    bin_arch = resolve_arch(meta, "binary", arch_choice)

    p3 = predict(df, meta, models, "direction3", d3_arch)
    ret_pred = float(predict(df, meta, models, "regression", reg_arch))
    p_up_bin = float(predict(df, meta, models, "binary", bin_arch))

    last_close = float(df["Close"].iloc[-1])
    price_pred = last_close * (1.0 + ret_pred)
    action, confidence, rationale = suggest(p3, ret_pred, p_thr, ret_thr)

    last_date = pd.Timestamp(df["Open time"].iloc[-1]).date()
    st.caption(f"Last bar: {last_date} · last close {last_close:,.2f} · "
               f"features {len(meta['tasks']['regression']['features'])} · "
               f"window {meta['window']} bars")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current price", f"{last_close:,.2f}")
    c2.metric(f"Predicted {horizon_label} price", f"{price_pred:,.2f}",
              delta=f"{ret_pred * 100:+.2f}%")
    c3.metric("Best class",
              config.DIRECTION3_LABELS[int(np.argmax(p3))],
              delta=f"p={float(np.max(p3)):.2f}")
    c4.metric("Binary P(UP)", f"{p_up_bin:.2f}")

    banner(action, confidence, rationale)

    st.progress(float(p3[2]), text=f"P(UP) = {p3[2]:.2f}")
    st.progress(float(p3[1]), text=f"P(FLAT ±{config.FLAT_BAND:.1%}) = {p3[1]:.2f}")
    st.progress(float(p3[0]), text=f"P(DOWN) = {p3[0]:.2f}")

    price_chart(df, price_pred, f"{meta['asset']} — next {horizon_label} prediction",
                horizon_label)

    # Attention Weights visualizer if LSTM_Attention was used
    chosen_models = [d3_arch, reg_arch, bin_arch]
    if any("Attention" in m for m in chosen_models):
        for t_name, m_name in [("Direction (3-class)", d3_arch), ("Regression", reg_arch), ("Binary", bin_arch)]:
            task_key = "direction3" if "3-class" in t_name else ("regression" if "Regression" in t_name else "binary")
            model_obj = models.get((task_key, m_name))
            if model_obj and hasattr(model_obj, "last_attn_weights") and model_obj.last_attn_weights is not None:
                attn_w = model_obj.last_attn_weights.squeeze().cpu().numpy()
                with st.expander(f"🔍 Temporal Attention Weights ({t_name})"):
                    fig_attn, ax_attn = plt.subplots(figsize=(10, 2.2))
                    ax_attn.bar(range(1, len(attn_w) + 1), attn_w, color="#6366f1", alpha=0.85)
                    ax_attn.set_xlabel(f"Lookback Bar (1 = oldest, {len(attn_w)} = latest)")
                    ax_attn.set_ylabel("Attention")
                    ax_attn.set_title(f"Dynamic Temporal Attention Distribution across {len(attn_w)}-step window")
                    st.pyplot(fig_attn, clear_figure=True)
                break

    # Explainable AI (XAI) section via Integrated Gradients
    with st.expander("🧠 Explainable AI (XAI) — Feature & Temporal Attribution", expanded=False):
        try:
            explain_task = "direction3" if action in ["BUY", "SELL"] else "regression"
            explain_arch = d3_arch if explain_task == "direction3" else reg_arch
            model_to_explain = models.get((explain_task, explain_arch))

            if model_to_explain is not None:
                feat_names = meta["tasks"][explain_task]["features"]
                X_explain = prepare_data.last_window(
                    df, feat_names, meta["tasks"][explain_task]["scaler_mean"],
                    meta["tasks"][explain_task]["scaler_scale"], window=meta["window"]
                )
                target_cls = int(np.argmax(p3)) if explain_task == "direction3" else None
                xai_res = explain.compute_xai_summary(model_to_explain, X_explain, feat_names,
                                                      target_class=target_cls, steps=20)

                col_x1, col_x2 = st.columns([1, 1.2])
                with col_x1:
                    st.markdown(f"**Key Decision Drivers for {action} Signal** (`{explain_arch}`)")
                    top_drivers = xai_res["ranked_features"][:6]
                    for d in top_drivers:
                        icon = "🟢 Bullish Driver" if d["attribution"] > 0 else "🔴 Bearish Drag"
                        st.markdown(f"{icon}: **{d['feature']}** ({d['attribution']:+.4f})")

                    temp_imp = xai_res["temporal_importance_pct"]
                    fig_temp, ax_temp = plt.subplots(figsize=(6, 2.2))
                    ax_temp.plot(range(1, len(temp_imp) + 1), temp_imp, color="#10b981", lw=2)
                    ax_temp.fill_between(range(1, len(temp_imp) + 1), temp_imp, color="#10b981", alpha=0.2)
                    ax_temp.set_xlabel("Lookback Bar (1 = oldest, 30 = latest)", fontsize=8)
                    ax_temp.set_ylabel("Influence %", fontsize=8)
                    ax_temp.set_title("Temporal Influence Across 30 Bars", fontsize=9)
                    st.pyplot(fig_temp, clear_figure=True)

                with col_x2:
                    st.markdown("**2D Attribution Matrix (Time × Feature)**")
                    top_f_indices = [feat_names.index(d["feature"]) for d in top_drivers if d["feature"] in feat_names][:8]
                    if top_f_indices:
                        sub_matrix = xai_res["attributions_2d"][:, top_f_indices]
                        sub_labels = [feat_names[i] for i in top_f_indices]
                        fig_heat, ax_heat = plt.subplots(figsize=(7, 4.2))
                        vmax = np.max(np.abs(sub_matrix)) + 1e-8
                        im = ax_heat.imshow(sub_matrix.T, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
                        ax_heat.set_yticks(range(len(sub_labels)))
                        ax_heat.set_yticklabels(sub_labels, fontsize=8)
                        ax_heat.set_xlabel("Lookback Bars (1 = oldest → 30 = latest)", fontsize=8)
                        ax_heat.set_title("Attribution Heatmap (Blue = Bullish, Red = Bearish)", fontsize=9)
                        fig_heat.colorbar(im, ax=ax_heat, orientation="horizontal", pad=0.28, shrink=0.7)
                        st.pyplot(fig_heat, clear_figure=True)
        except Exception as xai_err:
            st.caption(f"XAI attribution preview unavailable: {xai_err}")

    with st.expander("Model test metrics"):
        t1, t2, t3 = st.tabs(["Direction (3-class)", "Price regression", "Direction (binary)"])
        with t1:
            metrics_block(meta, "direction3", d3_arch)
            recent_table(df, meta, models, "direction3", d3_arch)
        with t2:
            metrics_block(meta, "regression", reg_arch)
            recent_table(df, meta, models, "regression", reg_arch)
        with t3:
            metrics_block(meta, "binary", bin_arch)
            recent_table(df, meta, models, "binary", bin_arch)


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Settings")

source = st.sidebar.radio("Data source", [*BUNDLED, UPLOAD_LABEL],
                          index=0,
                          help="Upload your own OHLCV CSV to use another "
                               "asset or timeframe.")

model_choice = st.sidebar.selectbox(
    "Architecture", ["Best (auto)", *MODEL_REGISTRY],
    help="'Best (auto)' picks the model with the best validation score "
         "for each task.")

with st.sidebar.expander("Suggestion rules"):
    p_thr = st.slider("Min class probability", 0.35, 0.70,
                      config.SUGGEST_P_THRESHOLD, 0.01)
    ret_thr = st.slider("Min confirming return (%)", -1.0, 1.0,
                        config.SUGGEST_RET_THRESHOLD * 100, 0.05) / 100.0

st.sidebar.caption(
    "Trend/price prediction for research and education only. "
    "Not financial advice."
)

if source in BUNDLED:
    asset, csv_path = BUNDLED[source]
    upload_label = None
elif source == UPLOAD_LABEL:
    asset = "BTC"
    csv_path = str(config.DATA_PATH)
    upload_label = None
    uploaded = st.sidebar.file_uploader("OHLCV CSV (Open time, OHLCV)",
                                        type=["csv"], key="main_upload")
    if uploaded is not None:
        data = uploaded.getvalue()
        h = file_hash(data)
        asset = f"UPLOAD_{h}"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = str(UPLOAD_DIR / f"{asset}.csv")
        Path(csv_path).write_bytes(data)
        upload_label = uploaded.name

# --------------------------------------------------------------------- body
st.title("📈 Next-Period Trend Classification & Price Prediction")
st.caption("Logistic-style baseline comparison aside — recurrent models "
           "(SimpleRNN / LSTM / GRU) classify direction and regress the next "
           "period's close from OHLCV features.")

meta = ensure_trained(asset, csv_path)
if meta is None:
    st.stop()

try:
    raw = prepare_data.load_csv(csv_path)
    detected = prepare_data.infer_interval(raw["Open time"])
    models = load_models(asset)
except Exception as exc:
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

tab_title_main = f"📊 Active Forecast ({asset} · {detected})"
tabs = st.tabs([tab_title_main, "📁 Upload Custom CSV / Timeframe"])

with tabs[0]:
    try:
        render_prediction(raw, meta, models, f"next {detected}", model_choice,
                          p_thr, ret_thr)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

with tabs[1]:
    st.subheader("Upload Any Custom CSV Dataset")
    st.write(
        "Upload any OHLCV CSV file with columns: `Open time`, `Open`, `High`, `Low`, `Close`, `Volume` "
        "to automatically train the models and generate forecasts for your custom asset or timeframe."
    )
    up = st.file_uploader("Drop OHLCV CSV here", type=["csv"], key="custom_uploader_tab")
    if up is not None:
        data = up.getvalue()
        h = file_hash(data)
        u_asset = f"UPLOAD_{h}"
        u_csv = str(UPLOAD_DIR / f"{u_asset}.csv")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        Path(u_csv).write_bytes(data)
        u_meta = ensure_trained(u_asset, u_csv)
        if u_meta is not None:
            try:
                raw_u = prepare_data.load_csv(u_csv)
                u_detected = prepare_data.infer_interval(raw_u["Open time"])
                u_models = load_models(u_asset)
                st.markdown(f"### Results for uploaded file ({u_detected})")
                render_prediction(raw_u, u_meta, u_models, f"next {u_detected}", model_choice,
                                  p_thr, ret_thr)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
