import os
import argparse
import glob
from datetime import datetime

from config import COMPANY, RAW_DIR, PROCESSED_DIR
from cleaning import process_and_save
from sentiment import run_vader, run_finbert
from signals import compute_deltas, generate_signals, save_signals
from returns import fetch_price_data, compute_alpha_table

def extract_quarter_key(filename: str):
    parts = filename.split("_")
    if len(parts) < 3:
        raise ValueError(f"Filename doesn't match expected pattern: {filename}")
    quarter = parts[1].lower()
    month = parts[2].lower().replace(".pdf", "").replace(".txt", "")
    return f"{quarter}_{month}"

def sort_files_by_quarter(files):
    def get_date_key(f):
        try:
            parts = os.path.basename(f).split("_")
            q = int(parts[1][1])
            m = datetime.strptime(parts[2][:3], "%b").month
            y = int("20" + parts[3][:2]) if len(parts) > 3 else 2022
            return (y, q, m)
        except:
            return (0, 0, 0)
    return sorted(files, key=get_date_key)

def main(company: str):
    raw_dir = os.path.join(RAW_DIR.replace(COMPANY, company))
    files = glob.glob(os.path.join(raw_dir, "*.pdf"))

    if not files:
        print(f"❌ No transcripts found in {raw_dir}")
        return

    files = sort_files_by_quarter(files)
    print(f"📁 Found {len(files)} transcripts for {company}")

    vader_score_map = {}
    finbert_score_map = {}
    earnings_dates = {}

    for idx, file_path in enumerate(files):
        quarter_key = f"prev{len(files)-idx-1}" if idx < len(files) - 1 else "current"
        filename = os.path.basename(file_path)
        cleaned_path = os.path.join(PROCESSED_DIR, f"{company}_{quarter_key}.txt")

        print(f"\n🧼 Cleaning transcript: {filename} → {quarter_key}")
        try:
            process_and_save(file_path, cleaned_path)
        except Exception as e:
            print(f"❌ Skipping {filename} due to error: {e}")
            continue

        if not os.path.exists(cleaned_path):
            print(f"❌ Cleaned transcript not found: {cleaned_path}")
            continue

        print(f"🧠 Running sentiment: {quarter_key}")
        v_score = run_vader(cleaned_path, quarter_key)
        f_score = run_finbert(cleaned_path, quarter_key)
        vader_score_map[quarter_key] = v_score
        finbert_score_map[quarter_key] = f_score

        # Fallback: parse date if embedded in file (customize this block)
        if "July" in filename:
            earnings_dates[quarter_key] = "2022-07-25"
        elif "April" in filename:
            earnings_dates[quarter_key] = "2022-04-13"
        elif "January" in filename:
            earnings_dates[quarter_key] = "2022-01-12"
        elif "October" in filename:
            earnings_dates[quarter_key] = "2021-10-13"
        else:
            print(f"⚠️  Manually map date for: {filename}")

    expected_keys = {"prev3", "prev2", "prev1", "current"}
    missing = expected_keys - set(vader_score_map.keys())
    if missing:
        print(f"❌ Missing sentiment scores for: {', '.join(missing)}. Cannot compute deltas.")
        return

    print("\n📊 Computing deltas...")
    vader_deltas = compute_deltas(vader_score_map)
    finbert_deltas = compute_deltas(finbert_score_map)

    print("📈 Generating trade signals...")
    signal_dict = generate_signals(
        finbert_scores=finbert_score_map,
        vader_scores=vader_score_map,
        finbert_deltas=finbert_deltas,
        vader_deltas=vader_deltas
    )
    save_signals(signal_dict, company)

    print("💰 Fetching price data + computing alpha...")
    price_data = fetch_price_data("2021-09-01", "2022-09-30")
    if price_data is None or price_data.empty:
        print(f"❌ Price data for {company} not available. Skipping alpha computation.")
        return

    alpha_df = compute_alpha_table(signal_dict, earnings_dates, price_data, company)

    print("\n✅ Pipeline complete.\n")
    print(alpha_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", type=str, required=True, help="Company ticker (e.g., INFY)")
    args = parser.parse_args()
    main(args.company)
