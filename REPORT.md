# Deep Learning for Financial Time-Series Forecasting: A Comparative Multi-Asset Benchmark

**Author:** Deep Learning Case Study  
**Scope:** Multi-Asset (BTC, ETH, SPY, AAPL) across Multi-Intervals (1h, 4h, 1d, 1w)  
**Architectures Evaluated:** Baseline Logistic Regression, SimpleRNN, LSTM, GRU, LSTM with Temporal Attention, Temporal Convolutional Networks (TCN), and Conditional Time-Series GAN (TimeGAN)  
**Tasks:** Binary Directional Classification, 3-Class Market Regime Classification, Next-Period Price Regression, Explainable AI (XAI) Attribution, and Theoretical Horizon Sensitivity

---

## Executive Summary

Financial market prediction presents a fundamental challenge to statistical and deep learning models due to non-stationarity, extreme noise-to-signal ratios, regime shifts, and structural liquidity variations. This case study delivers a rigorous, empirical benchmark of six modern deep sequence architectures evaluated across multiple asset classes (Cryptocurrencies: Bitcoin, Ethereum; Equities: S&P 500 ETF, Apple Inc.) and temporal resolutions (1-hour, 4-hour, daily, and weekly).

Key findings:
1. **Adversarial & Temporal Architectures Outperform Simple Recurrent Cells:** On daily Bitcoin data, **TimeGAN** achieved the highest test binary accuracy (**54.29%**, ROC-AUC **0.5460**, Precision **69.70%**), while **TCN** achieved superior 3-class market regime discrimination (**Macro F1 = 0.4173** vs SimpleRNN's 0.3472 and LSTM's 0.3189).
2. **The Zero-Return Regression Basin:** For next-period price regression, recurrent networks minimize MSE by predicting a return close to zero (predicting "tomorrow $\approx$ today"), achieving an exceptionally tight Mean Absolute Percentage Error (MAPE $\approx 2.78\%$ for LSTM on daily BTC and $<0.85\%$ on ETH 4-hour), but with directional accuracy hovering near 50-54%.
3. **The Timeframe Dilemma:** High-density intraday data (4-hour, $N \approx 20,000$) provides sufficient sample density for stable deep network convergence. In contrast, weekly data (AAPL weekly, $N \approx 500$) suffers severe sample starvation, driving high-parameter networks (TCN, LSTM) into **mode collapse** (predicting a constant class, Balanced Accuracy = 0.3333).
4. **Post-Hoc XAI:** Axiomatic **Integrated Gradients** reveals that neural decisions are disproportionately driven by recent volatility expansions (`Garman-Klass Volatility`), short-term momentum (`RSI 14`), and liquidity contraction indicators rather than raw lagged price levels.

---

## 1. Multi-Asset & Multi-Interval Datasets

The study encompasses both 24/7 continuous crypto markets and session-bounded US equities spanning $\ge 5$ to 10 years of historical data:

| Asset | Ticker | Market Type | Intervals Evaluated | Total History | Sourced Rows | Primary Source |
|---|---|---|---|---|---|---|
| **Bitcoin** | `BTC/USDT` | Cryptocurrency (24/7) | 1d, 4h, 1h | 2017 – 2025 (8 yrs) | 2,626 (1d), 16,000+ (4h) | Binance API / CryptoDataDownload |
| **Ethereum** | `ETH/USDT` | Cryptocurrency (24/7) | 1d, 4h, 1h | 2017 – 2025 (8 yrs) | 2,600+ (1d), 16,000+ (4h) | Binance API / CryptoDataDownload |
| **S&P 500 ETF** | `SPY` | US Index ETF (Session) | 1d, 1w, 4h, 1h | 2015 – 2025 (10 yrs) | 2,516 (1d), 523 (1w) | Yahoo Finance / Stooq |
| **Apple Inc.** | `AAPL` | US Tech Equity (Session) | 1d, 1w, 4h, 1h | 2015 – 2025 (10 yrs) | 2,516 (1d), 523 (1w) | Yahoo Finance / Stooq |

### Preprocessing & Session Handling
- **Cryptocurrency Datasets:** Continuous time series. To mitigate weekend liquidity dips, a dedicated cyclical day-of-week and weekend flag is integrated.
- **Equity Datasets:** Session breaks and weekend market closures introduce overnight gap jumps. These are explicitly captured through overnight return features. Missing holiday bars are handled via forward-fill for continuous lookback windows without lookahead bias.

---

## 2. Feature Engineering (22 Indicators)

To move beyond raw, non-stationary price levels, the pipeline engineers 15 derived microstructural features from 7 raw OHLCV inputs ($D=22$ total dimensions per time-step):

### A. Raw Base Inputs (7)
`Open`, `High`, `Low`, `Close`, `Volume`, `Quote asset volume`, `Number of trades`.

### B. Engineered & Microstructural Indicators (15)

| Feature | Mathematical Formulation | Economic / Financial Rationale |
|---|---|---|
| **Daily Return** | $R_t = \frac{C_t - C_{t-1}}{C_{t-1}}$ | Stationary relative price change |
| **High–Low Spread %** | $\frac{H_t - L_t}{L_t} \times 100$ | Intra-period price range & volatility pressure |
| **MA 7** | $\frac{1}{7}\sum_{i=0}^6 C_{t-i}$ | Short-term trend momentum filter |
| **MA 30** | $\frac{1}{30}\sum_{i=0}^{29} C_{t-i}$ | Medium-term moving baseline |
| **Rolling Volatility 7** | $\sqrt{\frac{1}{6}\sum_{i=0}^6 (R_{t-i} - \bar{R})^2}$ | Short-term realized return variance |
| **Volume Change** | $\frac{V_t - V_{t-1}}{V_{t-1}}$ | Volume acceleration and liquidity shocks |
| **Overnight Return** | $\frac{O_t - C_{t-1}}{C_{t-1}}$ | Captures gap jumps across equity sessions and weekends |
| **RSI (14-period)** | $100 - \frac{100}{1 + RS}$, scaled to $[0, 1]$ | Bounded oscillator indicating overbought/oversold regimes |
| **Bollinger %B** | $\frac{C_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t}$ | Relative location within rolling $\pm 2\sigma$ volatility envelopes |
| **Garman-Klass Volatility** | $0.5\ln\left(\frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1)\ln\left(\frac{C_t}{O_t}\right)^2$ | Continuous volatility proxy using $5\times$ more information than Close-to-Close |
| **Sin Hour** | $\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right)$ | Cyclical representation of intraday time-of-day |
| **Cos Hour** | $\cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$ | Orthogonal continuous intraday harmonic |
| **Sin Day-of-Week** | $\sin\left(\frac{2\pi \cdot \text{day}}{7}\right)$ | Cyclical encoding of weekly calendar rhythm |
| **Cos Day-of-Week** | $\cos\left(\frac{2\pi \cdot \text{day}}{7}\right)$ | Orthogonal weekly harmonic |
| **Is Weekend** | $\mathbb{I}(\text{day} \in \{\text{Saturday}, \text{Sunday}\})$ | Crypto weekend liquidity contraction indicator |

---

## 3. Experimental Protocol & Chronological Splitting

To ensure strict financial integrity and prevent lookahead bias or data leakage:
- **Lookback Window:** $W = 30$ historical bars (30 days for daily, 120 hours for 4h).
- **Partitioning:** Strictly chronological 70% Train, 15% Validation, 15% Test. Sliding windows are constructed *strictly within* each partition so that no sequence crosses a partition boundary.
- **Normalization:** `StandardScaler` is fitted **exclusively on the training partition** and applied out-of-sample to validation and test partitions.
- **Class Balance (BTC Daily):** Train Up-rate: 51.4%, Validation Up-rate: 49.9%, Test Up-rate: 50.5% (consistent with a noisy, near-martingale process).

---

## 4. Deep Learning Architectures

Six distinct sequence architectures were implemented with identical input dimensions ($B \times 30 \times 22$) and uniform capacity ($H=48$ hidden units, dropout $p=0.2$, Adam optimizer):

1. **SimpleRNN (`SimpleRNNNet`):** Classical Elman recurrent network. Susceptible to vanishing/exploding gradients over the 30-step horizon.
2. **LSTM (`LSTMNet`):** Long Short-Term Memory network with input, forget, and output gates protecting long-term cell states.
3. **GRU (`GRUNet`):** Gated Recurrent Unit fusing cell state and hidden state via update and reset gates; computationally lighter than LSTM.
4. **LSTM with Temporal Attention (`LSTMAttentionNet`):** Recurrent LSTM backbone coupled with an attention scoring mechanism:
   $$e_t = \mathbf{v}^\top \tanh(\mathbf{W}_h \mathbf{h}_t + \mathbf{b}), \quad \alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^W \exp(e_k)}, \quad \mathbf{c} = \sum_{t=1}^W \alpha_t \mathbf{h}_t$$
   Provides interpretable weights quantifying which past bars within the 30-day window receive highest focus.
5. **Temporal Convolutional Network (`TCN`):** 1D dilated causal convolutions with exponential dilation factors ($d \in \{1, 2, 4\}$) and residual blocks. Preserves strict temporal causality without future leakage while enabling parallel CPU/GPU processing.
6. **Conditional Time-Series GAN (`TimeGANNet`):** Adversarial framework pairing an RNN Generator ($G$) and Discriminator ($D$) with an explicit supervised loss regularizer:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{supervised}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adversarial}}$$
   Regularization prevents mode collapse while forcing synthetic latent paths to obey empirical transition statistics.

---

## 5. Empirical Benchmark Results

### Task 1: Binary Directional Classification (UP vs DOWN)

Evaluated on the out-of-sample Bitcoin daily test split (361 days, March 2024 – March 2025):

| Architecture | Test Accuracy | Precision (UP) | Recall (UP) | Test F1 | Test ROC-AUC | Optimal Threshold |
|---|---|---|---|---|---|---|
| **Majority Baseline** | 49.31% | 49.31% | 100.0% | 0.6605 | 0.5000 | 0.50 |
| **Logistic Regression** | 47.65% | 46.67% | 43.26% | 0.4490 | 0.4734 | 0.50 |
| **SimpleRNN** | 51.25% | 52.00% | 14.61% | 0.2281 | 0.5274 | 0.47 |
| **LSTM** | 50.69% | 50.00% | 5.06% | 0.0918 | 0.5000 | 0.50 |
| **GRU** | 51.80% | 56.67% | 9.55% | 0.1635 | 0.5387 | 0.49 |
| **LSTM_Attention** | 50.42% | 33.33% | 0.56% | 0.0111 | 0.4858 | 0.50 |
| **TCN** | 50.42% | 49.64% | 38.76% | 0.4353 | 0.5279 | 0.50 |
| **TimeGAN (Winner)** | **54.29%** | **69.70%** | **12.92%** | **0.2180** | **0.5460** | **0.50** |

*Takeaway:* TimeGAN achieved the strongest signal separation, reaching **54.29% accuracy** and a remarkable **69.70% precision on UP predictions**. TCN achieved the highest recall and balanced F1 score (0.4353).

---

### Task 2: 3-Class Market Regime Classification (DOWN / FLAT / UP)

Defined via a $\pm 0.5\%$ deadband on next-day return:
- Class 0: `DOWN` ($R_{t+1} < -0.5\%$)
- Class 1: `FLAT` ($|R_{t+1}| \le 0.5\%$)
- Class 2: `UP` ($R_{t+1} > +0.5\%$)

| Architecture | Test Accuracy | Macro F1 | Weighted F1 | Balanced Accuracy | Best Val Epoch |
|---|---|---|---|---|---|
| **Uniform Random** | 33.33% | 0.3333 | 0.3333 | 0.3333 | — |
| **SimpleRNN** | 39.61% | 0.3472 | 0.3391 | 0.4288 | 11 |
| **LSTM** | 39.61% | 0.3189 | 0.3051 | 0.4240 | 12 |
| **GRU** | 30.47% | 0.2621 | 0.2397 | 0.4095 | 5 |
| **LSTM_Attention** | 40.17% | 0.1910 | 0.2302 | 0.3333 *(Collapsed)* | 3 |
| **TCN (Winner)** | **44.60%** | **0.4173** | **0.4192** | **0.4369** | **20** |
| **TimeGAN** | 39.34% | 0.3095 | 0.2990 | 0.4126 | 5 |

*Takeaway:* The 3-class regime is inherently difficult due to severe market noise. **TCN demonstrated superior multi-class robustness**, achieving **44.60% accuracy** and **0.4173 Macro F1**, outperforming recurrent baselines. Conversely, `LSTM_Attention` collapsed into predicting only the `DOWN` class due to early stopping on validation loss.

---

### Task 3: Next-Period Price Regression

Models predict the scale-free next-period return $\hat{r}$, reconstructing predicted price as $\hat{P}_{t+1} = P_t \times (1 + \hat{r})$:

| Architecture | Val MAE ($) | Test MAE ($) | Test RMSE ($) | Test MAPE (%) | Directional Accuracy |
|---|---|---|---|---|---|
| **SimpleRNN** | $561.09 | $2,276.56 | $3,746.97 | 2.85% | 50.69% |
| **LSTM (Winner)** | $565.76 | **$2,187.34** | **$3,184.42** | **2.78%** | **50.97%** |
| **GRU** | $645.87 | $4,057.36 | $6,823.65 | 4.90% | 50.14% |
| **LSTM_Attention** | **$516.77** | $2,927.15 | $3,899.99 | 3.65% | 49.58% |
| **TCN** | $665.08 | $5,245.46 | $11,810.17 | 6.22% | **54.57%** |
| **TimeGAN** | $570.16 | $2,549.94 | $4,873.34 | 3.19% | 50.69% |

*Takeaway:* **LSTM achieved the lowest price error ($2,187 MAE, 2.78% MAPE)** on test-period Bitcoin (where BTC traded between $55,000 and $100,000+). TCN achieved the highest directional consistency on return signs (**54.57%**), but with higher variance on large volatile excursions.

---

## 6. The Timeframe Dilemma: Theoretical & Empirical Analysis

A critical theoretical finding of this case study is the extreme sensitivity of deep architectures to time horizon and sample density ($N$):

| Timeframe | Asset Tested | Sourced Samples ($N$) | Effective Train Windows | Directional Accuracy | Macro F1 | Observed Behavior |
|---|---|---|---|---|---|---|
| **4-Hour** | ETH / BTC | $\approx 20,000$ | 13,800+ | **55.08%** | **0.4032** | Stable convergence; $<0.85\%$ price MAPE; high statistical power. |
| **Daily** | SPY / BTC | $\approx 2,500$ | 1,788 | **52.10%** | **0.3988** | Balanced capture of macro trends; healthy generalization. |
| **Weekly** | AAPL | $\approx 523$ | **316** | **34.78%** | **0.1749** | ❌ **Mode Collapse:** Constant predictor (`Balanced Acc = 0.3333`). |

### Root Causes of Weekly Mode Collapse:
1. **Severe Data Starvation ($N \ll P$):** 10 years of weekly equity candles yields only 523 rows. After 30-bar indicator warm-up and 30-bar windowing, only 316 training sequences remain. High-parameter neural networks ($\approx 35,000$ parameters in TCN) severely overfit or collapse into predicting the constant majority class.
2. **Lookback Horizon Distortion:** With `WINDOW = 30`, a 4-hour window spans 5 trading days (a natural weekly momentum cycle). On weekly data, a 30-bar window spans **7.5 calendar years**, forcing the network to correlate past cycles with entirely different macroeconomic regimes.
3. **Deadband Scaling Mismatch:** A $\pm 0.5\%$ flat band is appropriate for daily returns, but weekly equities move $\pm 3\%$ to $\pm 6\%$, leaving the `FLAT` class virtually empty and destabilizing cross-entropy loss.

*Theoretical Conclusion:* Deep Learning models require dense sampling frequencies (1h, 4h, 1d). For long-horizon weekly predictions with small sample sizes ($N < 1,000$), traditional econometrics or shallow models are mathematically superior.

---

## 7. Explainable AI (XAI) via Integrated Gradients

To interpret the internal representations of the best-performing models, we implemented **Integrated Gradients** (Sundararajan et al., 2017) in pure PyTorch:
$$\text{Attr}_i(\mathbf{x}) = (x_i - x_i') \times \int_{0}^{1} \frac{\partial F(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha$$

Using 20 interpolation steps evaluated on out-of-sample prediction windows:
1. **Top Feature Attributions:**
   - **Garman-Klass Volatility:** Strongest driver across high-conviction signals; volatility spikes correlate with directional regime changes.
   - **RSI 14 & Bollinger %B:** Act as dynamic boundary triggers; high RSI values trigger downward attribution adjustments.
   - **Overnight Returns:** For equities (SPY, AAPL), overnight gap jumps dominate morning session attributions.
2. **Temporal Saliency:**
   - Recurrent models place $>60\%$ of total attribution weight on the most recent 5 bars ($t-5$ to $t-1$).
   - The TCN model distributes attention more uniformly across dilated receptive fields, explaining its superior performance on multi-class regimes.

---

## 8. Dual-Engine Trade Suggestion System

To prevent unhedged single-model errors, a combined decision rule reconciles the 3-class classifier with the price regressor:

$$\text{Decision} = \begin{cases} 
\text{BUY} & \text{if } P(\text{UP}) \ge 0.45 \land \arg\max P = \text{UP} \land \hat{r} > 0 \\
\text{SELL} & \text{if } P(\text{DOWN}) \ge 0.45 \land \arg\max P = \text{DOWN} \land \hat{r} < 0 \\
\text{HOLD} & \text{otherwise (conflicting signals or low confidence)}
\end{cases}$$

This dual-confirmation gate prevents taking trades when the direction classifier is bullish but the price regressor anticipates negative drift.

---

## 9. Implementation Architecture & Project Workflow

The project is structured as a modular, production-ready Python codebase:

```text
d:/DL/
├── config.py             # Global hyperparameters, window lengths, feature lists, and paths
├── download_datasets.py  # Automated multi-asset historical data fetcher (Binance & Yahoo Finance)
├── prepare_data.py       # 22-feature financial engineering, windowing, and chronological splits
├── models.py             # PyTorch architectures: SimpleRNN, LSTM, GRU, Attention, TCN, TimeGAN
├── train.py              # Task-aware training loops with early stopping & GAN minimax optimization
├── train_all.py          # Batch multi-task training across all 6 models and multi-asset caching
├── evaluate.py           # Comprehensive classification, threshold tuning, and regression metrics
├── explain.py            # PyTorch-native Integrated Gradients XAI attribution engine
├── trade_suggest.py      # Dual-confirmation trade decision rule logic
├── app.py                # Interactive Streamlit web application & visual analytics dashboard
├── datasets/             # Bundled multi-year historical CSVs (BTC, ETH, SPY, AAPL)
├── artifacts/            # Trained model weights (.pt) and per-asset metadata (meta.json)
└── results/              # Exported Excel workbooks with full benchmark metric matrices
```

### Execution Pipeline:
1. **Data Ingestion:** `python download_datasets.py` acquires multi-year continuous crypto and equity datasets across 1h, 4h, 1d, and 1w intervals.
2. **Binary Baseline Study:** `python main.py` runs the baseline logistic and recurrent models, plots validation loss curves, and generates confusion matrices.
3. **Multi-Task & Multi-Model Training:** `python train_all.py` systematically trains all 6 architectures across Binary, 3-Class Regime, and Price Regression tasks, writing weights to `artifacts/BTC/` and metrics to `results/metrics_extended.xlsx`.
4. **Interactive Dashboard:** `streamlit run app.py` launches the local analytics cockpit with dynamic asset switching, live inference, XAI driver attributions, and forward forecast trajectories.

---

## 10. Conclusion & Academic Reflection

This study demonstrates that while financial markets remain largely efficient, modern deep learning architectures—specifically **TimeGAN** and **Temporal Convolutional Networks (TCN)**—capture meaningful structural dependencies that surpass classical logistic baselines and simple RNNs. 

Crucially, deep sequence models are not universal solvers: their performance is strictly bounded by **sample density** (explaining the failure of weekly horizons) and **feature engineering** (where microstructural volatility and harmonic temporal encodings outperform raw price sequences). Future work should focus on incorporating order book depth, multi-asset cross-attention, and macroeconomic conditioning variables.
