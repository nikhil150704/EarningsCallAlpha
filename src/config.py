# src/config.py

from pathlib import Path
import os

class Config:
    def __init__(self, company: str = "INFY"):
        self.COMPANY = company
        self.TICKER = f"{self.COMPANY}.NS"

        # Absolute path to project root
        self.PROJECT_ROOT = Path(__file__).resolve().parents[1]

        # Data and output directories
        self.RAW_DIR = self.PROJECT_ROOT / "Data" / "Raw" / self.COMPANY
        self.PROCESSED_DIR = self.PROJECT_ROOT / "Data" / "Processed"
        self.OUTPUT_SCORES_DIR = self.PROJECT_ROOT / "Outputs" / "Scores"
        self.OUTPUT_SIGNALS_DIR = self.PROJECT_ROOT / "Outputs" / "Signals"
        self.OUTPUT_RETURNS_DIR = self.PROJECT_ROOT / "Outputs" / "Returns"

        # Sentiment settings
        self.MODEL_TOGGLE = "ensemble"  # "vader", "finbert", "ensemble"
        self.VADER_WEIGHT = 0.4
        self.FINBERT_WEIGHT = 0.6

        # Alpha logic
        self.RETURN_WINDOW = 7

    def ensure_dirs(self):
        """Ensure all necessary output directories exist."""
        for path in [
            self.PROCESSED_DIR,
            self.OUTPUT_SCORES_DIR,
            self.OUTPUT_SIGNALS_DIR,
            self.OUTPUT_RETURNS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)

# Usage:
# from config import Config
# cfg = Config(company="INFY")
# print(cfg.RAW_DIR)
