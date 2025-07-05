# src/config.py

import os

COMPANY = "INFY"
TICKER = f"{COMPANY}.NS"

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Define relative paths in a way that works on Windows, Linux, and Colab
RAW_DIR = os.path.join(PROJECT_ROOT, "Data", "Raw", COMPANY)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
OUTPUT_SCORES_DIR = os.path.join(PROJECT_ROOT, "Outputs", "Scores")
OUTPUT_SIGNALS_DIR = os.path.join(PROJECT_ROOT, "Outputs", "Signals")
OUTPUT_RETURNS_DIR = os.path.join(PROJECT_ROOT, "Outputs", "Returns")

# Sentiment model config
MODEL_TOGGLE = "ensemble"  # "vader", "finbert", or "ensemble"
VADER_WEIGHT = 0.4
FINBERT_WEIGHT = 0.6

# Trading signal config
RETURN_WINDOW = 7
