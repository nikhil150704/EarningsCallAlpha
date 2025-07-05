import os
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import Dataset
from config import MODEL_TOGGLE, VADER_WEIGHT, FINBERT_WEIGHT, OUTPUT_SCORES_DIR

# Load FinBERT once globally
finbert_model_name = "ProsusAI/finbert"
tokenizer = BertTokenizerFast.from_pretrained(finbert_model_name)
model = BertForSequenceClassification.from_pretrained(finbert_model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def run_vader(file_path: str, suffix: str) -> float:
    analyzer = SentimentIntensityAnalyzer()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sentences = [line.split(":", 1)[1].strip() for line in lines if ":" in line]
    results = []
    for idx, sentence in enumerate(sentences, 1):
        score = analyzer.polarity_scores(sentence)
        results.append({
            "index": idx,
            "sentence": sentence,
            **score
        })

    df = pd.DataFrame(results)
    avg = df["compound"].mean()
    output_file = os.path.join(OUTPUT_SCORES_DIR, f"vader_sentiment_output_{suffix}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ VADER ({suffix}): {avg:.4f}")
    return avg

def run_finbert(file_path: str, suffix: str) -> float:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sentences = [line.split(":", 1)[1].strip() for line in lines if ":" in line]
    data = Dataset.from_dict({"text": sentences})

    def tokenize_fn(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

    data = data.map(tokenize_fn, batched=True)
    data.set_format(type="torch", columns=["input_ids", "attention_mask"])

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(data), 32):
            batch = data[i:i+32]
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores, labels = torch.max(probs, dim=1)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    df = pd.DataFrame({
        "index": np.arange(1, len(sentences)+1),
        "sentence": sentences,
        "label": [label_map[l] for l in all_labels],
        "score": all_scores
    })

    output_file = os.path.join(OUTPUT_SCORES_DIR, f"finbert_sentiment_output_{suffix}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    score_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
    avg = df["label"].map(score_map).mean()
    print(f"✅ FinBERT ({suffix}): {avg:.4f}")
    return avg

def get_sentiment_score(file_path: str, suffix: str) -> float:
    vader_score = finbert_score = None

    if MODEL_TOGGLE == "vader":
        vader_score = run_vader(file_path, suffix)
        return vader_score
    elif MODEL_TOGGLE == "finbert":
        finbert_score = run_finbert(file_path, suffix)
        return finbert_score
    elif MODEL_TOGGLE == "ensemble":
        vader_score = run_vader(file_path, suffix)
        finbert_score = run_finbert(file_path, suffix)
        ensemble_score = VADER_WEIGHT * vader_score + FINBERT_WEIGHT * finbert_score
        print(f"✅ Ensemble Score ({suffix}): {ensemble_score:.4f}")
        return ensemble_score
    else:
        raise ValueError("Invalid MODEL_TOGGLE in config.py")
