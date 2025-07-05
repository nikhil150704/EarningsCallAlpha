import os
import json
from config import OUTPUT_SIGNALS_DIR, VADER_WEIGHT, FINBERT_WEIGHT

def compute_deltas(score_dict: dict) -> dict:
    return {
        "delta3": score_dict["prev2"] - score_dict["prev3"],
        "delta2": score_dict["prev1"] - score_dict["prev2"],
        "delta1": score_dict["current"] - score_dict["prev1"]
    }

def generate_trade_signal(score: float, delta: float) -> str:
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
    vader_deltas: dict
) -> dict:
    
    key_to_delta_map = {
        "prev3": None,
        "prev2": "delta3",
        "prev1": "delta2",
        "current": "delta1"
    }

    signal_log = {}

    for key, delta_key in key_to_delta_map.items():
        if delta_key is None:
            continue

        f_score = finbert_scores[key]
        v_score = vader_scores[key]
        f_delta = finbert_deltas[delta_key]
        v_delta = vader_deltas[delta_key]

        combined_score = FINBERT_WEIGHT * f_score + VADER_WEIGHT * v_score
        combined_delta = FINBERT_WEIGHT * f_delta + VADER_WEIGHT * v_delta

        signal = generate_trade_signal(combined_score, combined_delta)

        signal_log[key] = {
            "finbert_score": round(f_score, 4),
            "vader_score": round(v_score, 4),
            "finbert_delta": round(f_delta, 4),
            "vader_delta": round(v_delta, 4),
            "combined_score": round(combined_score, 4),
            "combined_delta": round(combined_delta, 4),
            "signal": signal
        }

    return signal_log

def save_signals(signals: dict, company: str):
    output_file = os.path.join(OUTPUT_SIGNALS_DIR, f"{company}_signals.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # ✅ Prevent crash
    with open(output_file, "w") as f:
        json.dump(signals, f, indent=4)
    print(f"✅ Signals saved to {output_file}")

