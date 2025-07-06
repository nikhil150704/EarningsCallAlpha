import os
import argparse
import glob
import logging
from datetime import datetime
import re
from pathlib import Path
import fitz  # PyMuPDF

from config import Config
from cleaning import process_and_save
from sentiment import run_vader, run_finbert
from signals import compute_deltas, generate_signals, save_signals
from returns import fetch_price_data, compute_alpha_table

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_date_from_text(text: str) -> str | None:
    patterns = [
        r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})",
        r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})"
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

def extract_date_from_raw_pdf(pdf_path: str) -> str | None:
    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join([doc[i].get_text("text") for i in range(min(3, len(doc)))])
            return extract_date_from_text(text)
    except Exception as e:
        logger.warning(f"⚠️ Could not extract date from raw PDF {pdf_path}: {e}")
        return None

def generate_quarter_key(date_str: str, latest: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    fy = dt.year + 1 if dt.month > 3 else dt.year
    return f"FY{str(fy)[-2:]}_Q{((dt.month - 1) // 3) % 4 + 1}" if date_str != latest else "current"

def main(company: str):
    config = Config(company)
    logger.info(f"🚀 Running earnings sentiment pipeline for: {company}")

    files = glob.glob(os.path.join(config.RAW_DIR, "*.pdf"))
    if not files:
        logger.error(f"❌ No transcripts found in {config.RAW_DIR}")
        return

    logger.info(f"📁 Found {len(files)} transcripts for {company}")

    intermediate = []
    for file_path in files:
        filename = os.path.basename(file_path)
        cleaned_path = os.path.join(config.PROCESSED_DIR, f"{company}_{filename.replace('.pdf', '.txt')}")

        logger.info(f"🧼 Cleaning: {filename}")
        try:
            process_and_save(Path(file_path), Path(cleaned_path))
        except Exception as e:
            logger.error(f"❌ Skipping {filename}: {e}")
            continue

        if not os.path.exists(cleaned_path):
            logger.warning(f"⚠️ Cleaned file missing: {cleaned_path}")
            continue

        date = extract_date_from_raw_pdf(file_path)
        if not date:
            with open(cleaned_path, "r", encoding="utf-8") as f:
                cleaned_text = f.read()
            date = extract_date_from_text(cleaned_text)

        if date:
            intermediate.append((file_path, cleaned_path, date))
        else:
            logger.warning(f"⚠️ Date missing for {filename}. Update extract_earnings_date().")

    if not intermediate:
        logger.error("❌ No dates could be extracted. Aborting.")
        return

    intermediate.sort(key=lambda x: datetime.strptime(x[2], "%Y-%m-%d"))

    vader_scores, finbert_scores, earnings_dates = {}, {}, {}

    for idx, (file_path, cleaned_path, date) in enumerate(intermediate):
        quarter_key = generate_quarter_key(date, intermediate[-1][2])
        filename = os.path.basename(file_path)

        logger.info(f"🧠 Running sentiment: {quarter_key}")
        vader_scores[quarter_key] = run_vader(cleaned_path, quarter_key, config)
        finbert_scores[quarter_key] = run_finbert(cleaned_path, quarter_key, config)

        earnings_dates[quarter_key] = date

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
