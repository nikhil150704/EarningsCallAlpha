# src/returns.py

import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf
import yfinance.shared as yfshared
from config import Config

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Patch session headers to avoid Yahoo Finance errors
yfshared._session = requests.Session()
yfshared._session.headers.update({"User-Agent": "Mozilla/5.0"})


def fetch_price_data_fallback_nse(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fallback method: fetch historical Close prices from NSE website.
    ticker = raw NSE ticker like 'INFY'
    """
    try:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

        url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={ticker}&series=[%22EQ%22]&from={start_date.strftime('%d-%m-%Y')}&to={end_date.strftime('%d-%m-%Y')}"

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br"
        }

        session = requests.Session()
        session.headers.update(headers)
        session.get("https://www.nseindia.com")  # set cookies

        resp = session.get(url)
        data = resp.json().get("data", [])

        if not data:
            raise ValueError("Empty data from NSE fallback")

        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["CH_TIMESTAMP"])
        df.set_index("Date", inplace=True)
        df = df.rename(columns={"CH_CLOSING_PRICE": "Close"})
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df[["Close"]].dropna()
        return df

    except Exception as e:
        logger.error(f"❌ NSE fallback failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_price_data(start: str, end: str, ticker: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Download price data using yfinance with fallback to NSE if it fails.
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

    logger.warning(f"⚠️ YFinance failed. Trying fallback using NSE API for {ticker}")
    fallback_ticker = ticker.replace(".NS", "")
    return fetch_price_data_fallback_nse(fallback_ticker)


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


def compute_alpha_table(
    signals: dict,
    earnings_dates: dict,
    price_data: pd.DataFrame,
    company: str,
    config: Config
) -> pd.DataFrame:
    result = []
    for quarter_key, date in earnings_dates.items():
        if quarter_key not in signals:
            logger.warning(f"⚠️ No signal for {quarter_key}, skipping alpha.")
            continue

        strat_return = get_post_earnings_return(price_data, date, config.RETURN_WINDOW)
        bench_return = get_benchmark_return(price_data, date, config.RETURN_WINDOW)
        if strat_return is None or bench_return is None:
            logger.warning(f"⚠️ Return computation failed for {quarter_key} ({date})")
            continue

        alpha = strat_return - bench_return
        row = {
            "Quarter": quarter_key,
            "Date": date,
            "Signal": signals[quarter_key]["signal"],
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
