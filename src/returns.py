# src/returns.py

import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf
import yfinance.shared as yfshared
from datetime import datetime, timedelta
from config import Config

try:
    from nsepython import nsefetch
except ImportError:
    raise ImportError("Install nsepython via `pip install nsepython`")

# ------------------------- Logging Setup -------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------- Patch YFinance -------------------------

yfshared._session = requests.Session()
yfshared._session.headers.update({"User-Agent": "Mozilla/5.0"})

# ------------------------- NSE Fallback Cache -------------------------

nse_cache = {}

def fetch_price_data_fallback_nse(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch price data from NSE India using nsepython as a fallback.
    """
    if ticker in nse_cache:
        logger.info(f"⚡ Loaded {ticker} from cache.")
        return nse_cache[ticker]

    logger.info(f"⚠️ YFinance failed. Trying fallback using NSE API for {ticker}")

    try:
        url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={ticker}&series=[%22EQ%22]&from={start}&to={end}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        records = json_data.get("data", [])
        if not records:
            raise ValueError("Empty data from NSE")

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["CH_TIMESTAMP"])
        df.set_index("Date", inplace=True)
        df["Close"] = pd.to_numeric(df["CH_CLOSING_PRICE"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)
        df = df[["Close"]]
        nse_cache[ticker] = df
        return df

    except Exception as e:
        logger.error(f"❌ NSE fallback failed for {ticker}: {e}")
        return pd.DataFrame()

# ------------------------- Main Price Fetch -------------------------

def fetch_price_data(start: str, end: str, ticker: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Primary method to fetch historical price data.
    Tries YFinance, then falls back to NSE API.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"📥 Attempt {attempt}: Downloading price data for {ticker} from {start} to {end}")
            df = yf.download(ticker, start=start, end=end)
            if not df.empty:
                df.dropna(inplace=True)
                return df
            else:
                logger.warning(f"⚠️ Attempt {attempt}: Downloaded price data is empty.")
        except Exception as e:
            logger.error(f"❌ Attempt {attempt}: Failed to get ticker '{ticker}' due to: {e}")
        time.sleep(2)

    fallback_ticker = ticker.replace(".NS", "")
    return fetch_price_data_fallback_nse(fallback_ticker, start, end)

# ------------------------- Return Calculations -------------------------

def get_post_earnings_return(df: pd.DataFrame, date: str, days: int) -> float | None:
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
    except Exception as e:
        logger.warning(f"⚠️ Post-earnings return failed for {date}: {e}")
        return None

def get_benchmark_return(df: pd.DataFrame, date: str, days: int) -> float | None:
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
    except Exception as e:
        logger.warning(f"⚠️ Benchmark return failed for {date}: {e}")
        return None

# ------------------------- Alpha Calculation -------------------------

def compute_alpha_table(
    signals: dict,
    earnings_dates: dict,
    price_data: pd.DataFrame,
    company: str,
    config: Config
) -> pd.DataFrame:
    result = []

    for key, date in earnings_dates.items():
        if key not in signals:
            logger.warning(f"⚠️ No signal for {key}, skipping alpha.")
            continue

        strat_return = get_post_earnings_return(price_data, date, config.RETURN_WINDOW)

        # Fetch NIFTY50 fallback benchmark
        benchmark_df = fetch_price_data("2021-09-01", "2022-09-30", "^NSEI")
        if benchmark_df.empty:
            logger.warning(f"⚠️ Benchmark fallback to NSE NIFTY failed.")
            continue

        bench_return = get_benchmark_return(benchmark_df, date, config.RETURN_WINDOW)
        if strat_return is None or bench_return is None:
            logger.warning(f"⚠️ Return calc failed for {key} ({date})")
            continue

        alpha = strat_return - bench_return
        row = {
            "Quarter": key,
            "Date": date,
            "Signal": signals[key]["signal"],
            "Strategy_Return": round(strat_return, 4),
            "Benchmark_Return": round(bench_return, 4),
            "Alpha": round(alpha, 4)
        }
        result.append(row)

    df = pd.DataFrame(result)
    output_file = config.OUTPUT_RETURNS_DIR / f"{company}_alpha.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"✅ Alpha results saved to {output_file}")
    return df
