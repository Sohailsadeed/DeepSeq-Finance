"""Automated multi-asset data acquisition from Binance (Crypto) and Yahoo Finance (Stocks).
Downloads and formats 1h, 4h, 1d, and 1w historical OHLCV series."""

import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
DATA_DIR.mkdir(exist_ok=True)

CRYPTO_SOURCES = {
    "btc_1d": "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv",
    "btc_1h": "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1h.csv",
    "eth_1d": "https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_d.csv",
    "eth_1h": "https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_1h.csv",
}


def download_crypto(name: str, url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 10000:
        return
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=40) as response, open(dest_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"Saved {dest_path.name}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")


def resample_crypto_to_4h(input_1h_csv: Path, output_4h_csv: Path):
    if not input_1h_csv.exists() or output_4h_csv.exists():
        return
    try:
        df = pd.read_csv(input_1h_csv, skiprows=1)
        date_col = next((c for c in df.columns if c.strip().lower() in ["date", "open time", "timestamp"]), None)
        df["Open time"] = pd.to_datetime(df[date_col], format="mixed")
        df = df.sort_values("Open time").drop_duplicates(subset=["Open time"]).reset_index(drop=True)

        col_map = {str(c).strip().lower(): c for c in df.columns}
        open_c = col_map.get("open")
        high_c = col_map.get("high")
        low_c = col_map.get("low")
        close_c = col_map.get("close")
        vol_c = next((c for c in df.columns if "volume" in c.lower() and "quote" not in c.lower() and "usdt" not in c.lower()), None)
        if not vol_c:
            vol_c = next((c for c in df.columns if "volume" in c.lower()), None)

        resampled = (df.resample("4h", on="Open time")
                       .agg({
                           open_c: "first",
                           high_c: "max",
                           low_c: "min",
                           close_c: "last",
                           vol_c: "sum",
                       })
                       .dropna()
                       .reset_index())

        resampled = resampled.rename(columns={
            open_c: "Open",
            high_c: "High",
            low_c: "Low",
            close_c: "Close",
            vol_c: "Volume",
        })
        resampled.to_csv(output_4h_csv, index=False)
        print(f"Generated {output_4h_csv.name}")
    except Exception as e:
        print(f"Resampling error: {e}")


def fetch_stock_datasets(ticker="SPY"):
    clean_ticker = ticker.lower()
    
    dest_1d = DATA_DIR / f"{clean_ticker}_1d_data.csv"
    if not dest_1d.exists():
        df_1d = yf.download(ticker, period="10y", interval="1d", progress=False)
        if not df_1d.empty:
            df_1d = df_1d.reset_index()
            if isinstance(df_1d.columns, pd.MultiIndex):
                df_1d.columns = [col[0] for col in df_1d.columns]
            date_c = "Date" if "Date" in df_1d.columns else "Datetime"
            df_1d = df_1d.rename(columns={date_c: "Open time"})
            df_1d[["Open time", "Open", "High", "Low", "Close", "Volume"]].to_csv(dest_1d, index=False)
            print(f"Saved {dest_1d.name}")

    dest_1w = DATA_DIR / f"{clean_ticker}_1w_data.csv"
    if not dest_1w.exists():
        df_1w = yf.download(ticker, period="10y", interval="1wk", progress=False)
        if not df_1w.empty:
            df_1w = df_1w.reset_index()
            if isinstance(df_1w.columns, pd.MultiIndex):
                df_1w.columns = [col[0] for col in df_1w.columns]
            date_c = "Date" if "Date" in df_1w.columns else "Datetime"
            df_1w = df_1w.rename(columns={date_c: "Open time"})
            df_1w[["Open time", "Open", "High", "Low", "Close", "Volume"]].to_csv(dest_1w, index=False)
            print(f"Saved {dest_1w.name}")

    dest_1h = DATA_DIR / f"{clean_ticker}_1h_data.csv"
    dest_4h = DATA_DIR / f"{clean_ticker}_4h_data.csv"
    if not dest_1h.exists() or not dest_4h.exists():
        df_1h = yf.download(ticker, period="730d", interval="1h", progress=False)
        if not df_1h.empty:
            df_1h = df_1h.reset_index()
            if isinstance(df_1h.columns, pd.MultiIndex):
                df_1h.columns = [col[0] for col in df_1h.columns]
            date_c = "Datetime" if "Datetime" in df_1h.columns else "Date"
            df_1h = df_1h.rename(columns={date_c: "Open time"})
            df_1h["Open time"] = pd.to_datetime(df_1h["Open time"])
            df_1h[["Open time", "Open", "High", "Low", "Close", "Volume"]].to_csv(dest_1h, index=False)
            print(f"Saved {dest_1h.name}")

            df_4h = (df_1h.resample("4h", on="Open time")
                          .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                          .dropna()
                          .reset_index())
            df_4h[["Open time", "Open", "High", "Low", "Close", "Volume"]].to_csv(dest_4h, index=False)
            print(f"Saved {dest_4h.name}")


def main():
    print(f"Dataset root: {DATA_DIR.resolve()}\n")
    for name, url in CRYPTO_SOURCES.items():
        dest = DATA_DIR / f"{name}_cdd.csv"
        download_crypto(name, url, dest)

    resample_crypto_to_4h(DATA_DIR / "btc_1h_cdd.csv", DATA_DIR / "btc_4h_cdd.csv")
    resample_crypto_to_4h(DATA_DIR / "eth_1h_cdd.csv", DATA_DIR / "eth_4h_cdd.csv")
    fetch_stock_datasets("SPY")
    fetch_stock_datasets("AAPL")
    print("All datasets ready in 'datasets/'.")


if __name__ == "__main__":
    main()
