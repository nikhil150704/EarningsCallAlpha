import os
import argparse
import glob
import logging
from datetime import datetime
import re
from pathlib import Path

from config import Config
from cleaning import process_and_save
from sentiment import run_vader, run_finbert
from signals import compute_deltas, generate_signals, save_signals
from returns import fetch_price_data, compute_alpha_table

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def sort_files_by_quarter(files):
    def extract_key(f):
        try:
            parts = os.path.basename(f).split("_")
            q = int(parts[1][1])
            m = datetime.strptime(parts[2][:3], "%b").month
            y = int("20" + parts[3][:2]) if len(parts) > 3 else 2022
            return (y, q, m)
        except Exception:
            return (0, 0, 0)
    return sorted(files, key=extract_key)

def extract_date_from_text(text: str) -> str | None:
    """
    Extracts earnings call date from transcript body.
    Looks for formats like: July 15, 2022 or 15 July 2022
    """
    patterns = [
        r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})",         # July 15, 2022
        r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})"           # 15 July 2022
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                dt = datetime.strptime(" ".join(match.groups()), "%B %d %Y")
            except:
                try:
                    dt = datetime.strptime(" ".join(match.groups()), "%d %B %Y")
                except:
                    continue
            return dt.strftime("%Y-%m-%d")
    return None



def main(company: str):
    config = Config(company)
    logger.info(f"🚀 Running earnings sentiment pipeline for: {company}")

    files = glob.glob(os.path.join(config.RAW_DIR, "*.pdf"))
    if not files:
        logger.error(f"❌ No transcripts found in {config.RAW_DIR}")
        return

    files = sort_files_by_quarter(files)
    logger.info(f"📁 Found {len(files)} transcripts for {company}")

    vader_scores, finbert_scores, earnings_dates = {}, {}, {}
    total = len(files)

    for idx, file_path in enumerate(files):
        quarter_key = f"prev{total - idx - 1}" if idx < total - 1 else "current"
        filename = os.path.basename(file_path)
        cleaned_path = os.path.join(config.PROCESSED_DIR, f"{company}_{quarter_key}.txt")

        logger.info(f"🧼 Cleaning: {filename} → {quarter_key}")
        try:
            process_and_save(Path(file_path), Path(cleaned_path))

        except Exception as e:
            logger.error(f"❌ Skipping {filename}: {e}")
            continue

        if not os.path.exists(cleaned_path):
            logger.warning(f"⚠️ Cleaned file missing: {cleaned_path}")
            continue

        logger.info(f"🧠 Running sentiment: {quarter_key}")
        vader_scores[quarter_key] = run_vader(cleaned_path, quarter_key, config)
        finbert_scores[quarter_key] = run_finbert(cleaned_path, quarter_key, config)


        with open(cleaned_path, "r", encoding="utf-8") as f:
            cleaned_text = f.read()
        date = extract_date_from_text(cleaned_text)

        if date:
            earnings_dates[quarter_key] = date
        else:
            logger.warning(f"⚠️ Date missing for {filename}. Update extract_earnings_date().")

    if len(vader_scores) < 2 or len(finbert_scores) < 2:
        logger.error("❌ Not enough valid sentiment data to compute deltas.")
        return

    logger.info("📊 Computing deltas...")
    vader_deltas = compute_deltas(vader_scores)
    finbert_deltas = compute_deltas(finbert_scores)

    logger.info("📈 Generating trade signals...")
    signals = generate_signals(
        finbert_scores=finbert_scores,
        vader_scores=vader_scores,
        finbert_deltas=finbert_deltas,
        vader_deltas=vader_deltas,
        config=config
    )
    save_signals(signals, company, config)

    logger.info("💰 Fetching price data + computing alpha...")
    price_data = fetch_price_data("2021-09-01", "2022-09-30", config.TICKER)
    if price_data is None or price_data.empty:
        logger.error("❌ Price data unavailable. Skipping alpha.")
        return

    alpha_df = compute_alpha_table(signals, earnings_dates, price_data, company, config)
    logger.info("✅ Pipeline complete.")
    print(alpha_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Earnings Call Sentiment Pipeline")
    parser.add_argument("--company", type=str, required=True, help="Company ticker (e.g., INFY)")
    args = parser.parse_args()
    main(args.company)
