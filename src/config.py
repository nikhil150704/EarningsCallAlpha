# src/config.py

COMPANY = "INFY"
TICKER = f"{COMPANY}.NS"
RAW_DIR = rf"C:\Users\Nikhil Gudapati\Documents\Projects\Sentiment Analysis Project\data\raw\{COMPANY}"
PROCESSED_DIR = rf"C:\Users\Nikhil Gudapati\Documents\Projects\Sentiment Analysis Project\data\processed"
OUTPUT_SCORES_DIR = rf"C:\Users\Nikhil Gudapati\Documents\Projects\Sentiment Analysis Project\outputs\scores"
OUTPUT_SIGNALS_DIR = rf"C:\Users\Nikhil Gudapati\Documents\Projects\Sentiment Analysis Project\outputs\signals"
OUTPUT_RETURNS_DIR = rf"C:\Users\Nikhil Gudapati\Documents\Projects\Sentiment Analysis Project\outputs\returns"

MODEL_TOGGLE = "ensemble"  # "vader", "finbert", or "ensemble"
VADER_WEIGHT = 0.4
FINBERT_WEIGHT = 0.6
RETURN_WINDOW = 7
