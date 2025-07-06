# src/signals.py

import os
import json
import logging
from config import Config

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_deltas(score_dict: dict) -> dict:
    """
    Compute deltas between consecutive sorted keys in score_dict.
    Assumes keys are chronological like ["prev3", "prev2", "prev1", "current"]
    """
    sorted_keys = sorted(score_dict.keys(), key=lambda k: int(k.replace("prev", "-")) if "prev" in k else 0)
    deltas = {}

    for i in range(1, len(sorted_keys)):
        k_curr = sorted_keys[i]
        k_prev = sorted_keys[i - 1]
        try:
            deltas[f"delta_{k_curr}"] = score_dict[k_curr] - score_dict[k_prev]
        except KeyError as e:
            logger.warning(f"Missing key in score_dict during delta computation: {e}")
            deltas[f"delta_{k_curr}"] = 0.0

    return deltas

def generate_trade_signal(score: float, delta: float) -> str:
    """
    Basic rules-based strategy.
    """
    if score > 0.05 and delta > -0.05:
        return "LONG"
    elif score < -0.05 and delta < 0.01:
        return "SHORT"
    else:
        return "HOLD"

def generate_signals(
    finbert_scores: dict,
    vader_scores: dict,
    finbert_deltas: dict,
    vader_deltas: dict,
    config: Config
) -> dict:
    """
    Combine FinBERT and VADER signals into a hybrid ensemble trade signal per quarter.
    """
    quarters = sorted(finbert_scores.keys(), key=lambda k: int(k.replace("prev", "-")) if "prev" in k else 0)
    signal_log = {}

    for quarter in quarters:
        delta_key = f"delta_{quarter}"
        if delta_key not in finbert_deltas or delta_key not in vader_deltas:
            logger.warning(f"Skipping signal generation for {quarter} due to missing delta.")
            continue

        f_score = finbert_scores.get(quarter, 0.0)
        v_score = vader_scores.get(quarter, 0.0)
        f_delta = finbert_deltas.get(delta_key, 0.0)
        v_delta = vader_deltas.get(delta_key, 0.0)

        combined_score = config.FINBERT_WEIGHT * f_score + config.VADER_WEIGHT * v_score
        combined_delta = config.FINBERT_WEIGHT * f_delta + config.VADER_WEIGHT * v_delta
        signal = generate_trade_signal(combined_score, combined_delta)

        signal_log[quarter] = {
            "finbert_score": round(f_score, 4),
            "vader_score": round(v_score, 4),
            "finbert_delta": round(f_delta, 4),
            "vader_delta": round(v_delta, 4),
            "combined_score": round(combined_score, 4),
            "combined_delta": round(combined_delta, 4),
            "signal": signal
        }

    return signal_log

def save_signals(signals: dict, company: str, config: Config):
    """
    Write the final JSON of signals to disk.
    """
    output_file = config.OUTPUT_SIGNALS_DIR / f"{company}_signals.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(signals, f, indent=4)
    logger.info(f"✅ Signals saved to {output_file}")
