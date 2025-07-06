import os
import requests
import yfinance as yf
import pandas as pd
from config import TICKER, OUTPUT_RETURNS_DIR, RETURN_WINDOW

# 🛠️ Fix for empty data + User-Agent spoof (some yfinance calls get blocked without this)
import yfinance.shared as yfshared
yfshared._session = requests.Session()
yfshared._session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

def fetch_price_data(start: str, end: str) -> pd.DataFrame:
    print(f"📥 Downloading price data for {TICKER} from {start} to {end}")
    
    try:
        # 🔁 Warm-up hack to prevent first-call failure in fresh envs like Colab
        _ = yf.download("AAPL", start="2022-01-01", end="2022-01-02")

        df = yf.download(TICKER, start=start, end=end)

        if df.empty:
            raise ValueError("Downloaded price data is empty.")
        
        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"❌ Failed to download price data for {TICKER}: {e}")
        return pd.DataFrame()

def get_benchmark_return(df: pd.DataFrame, date: str, days: int = RETURN_WINDOW):
    try:
        event_date = pd.to_datetime(date)
        event_date = df.index[df.index.searchsorted(event_date)]
        entry_idx = df.index.get_loc(event_date)
        exit_idx = entry_idx + days
        if exit_idx >= len(df):
            return None
        entry_price = df['Close'].iloc[entry_idx]
        exit_price = df['Close'].iloc[exit_idx]
        return (exit_price - entry_price) / entry_price
    except Exception:
        return None

def get_post_earnings_return(df: pd.DataFrame, date: str, days: int = RETURN_WINDOW):
    try:
        event_date = pd.to_datetime(date)
        if event_date not in df.index:
            event_date = df.index[df.index.searchsorted(event_date)]
        entry_idx = df.index.get_loc(event_date) + 1
        exit_idx = entry_idx + days
        if exit_idx >= len(df):
            return None
        entry_price = df['Close'].iloc[entry_idx]
        exit_price = df['Close'].iloc[exit_idx]
        return (exit_price - entry_price) / entry_price
    except Exception:
        return None

def compute_alpha_table(signals: dict, earnings_dates: dict, price_data: pd.DataFrame, company: str):
    result = []
    for key, date in earnings_dates.items():
        if key not in signals:
            continue
        strat_return = get_post_earnings_return(price_data, date)
        benchmark_return = get_benchmark_return(price_data, date)
        if strat_return is None or benchmark_return is None:
            continue
        alpha = strat_return - benchmark_return
        row = {
            "Quarter": key,
            "Date": date,
            "Signal": signals[key]["signal"],
            "Strategy_Return": round(strat_return, 4),
            "Benchmark_Return": round(benchmark_return, 4),
            "Alpha": round(alpha, 4)
        }
        result.append(row)

    df = pd.DataFrame(result)
    output_file = os.path.join(OUTPUT_RETURNS_DIR, f"{company}_alpha.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Alpha results saved to {output_file}")
    return df
