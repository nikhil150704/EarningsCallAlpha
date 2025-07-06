import os
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import Dataset
from config import Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- Helper Functions -------------------------

def read_sentences(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sentences = []
    for line in lines:
        parts = line.split("|", 1)
        if len(parts) == 2:
            sentence = parts[1].strip()
        else:
            sentence = line.strip()
        if sentence:
            sentences.append(sentence)

    print(f"📏 Loaded {len(sentences)} sentences from {os.path.basename(file_path)}")
    return sentences


def save_dataframe(df: pd.DataFrame, suffix: str, model_name: str,config):
    output_file = os.path.join(config.OUTPUT_SCORES_DIR, f"{model_name}_sentiment_output_{suffix}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ {model_name.upper()} ({suffix}) results saved")

# ------------------------- Model Interface -------------------------

class SentimentModel:
    def run(self, file_path: str, suffix: str, config=None) -> float:
        raise NotImplementedError("Subclasses must implement run()")

# ------------------------- VADER Model -------------------------

class VaderModel(SentimentModel):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def run(self, file_path: str, suffix: str, config=None) -> float:
        sentences = read_sentences(file_path)
        results = []
        for idx, sentence in enumerate(sentences, 1):
            if not sentence.strip():
                continue
            score = self.analyzer.polarity_scores(sentence)
            if "compound" not in score:
                continue
            results.append({
                "index": idx,
                "sentence": sentence,
                **score
            })

        if not results:
            print(f"⚠️ VADER skipped: No valid sentences for {suffix}")
            return 0.0

        df = pd.DataFrame(results)
        avg = df["compound"].mean()
        save_dataframe(df, suffix, "vader",config)
        print(f"✅ VADER ({suffix}): {avg:.4f}")
        return avg

# ------------------------- FinBERT Model -------------------------

class FinBERTModel(SentimentModel):
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
        self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
        self.model.eval()

    def run(self, file_path: str, suffix: str, config=None) -> float:
        sentences = read_sentences(file_path)
        if not sentences:
            print(f"⚠️ FinBERT skipped: No valid input for {suffix}")
            return 0.0

        data = Dataset.from_dict({"text": sentences})
        data = data.map(lambda x: self.tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)
        data.set_format(type="torch", columns=["input_ids", "attention_mask"])

        all_scores, all_labels = [], []
        with torch.no_grad():
            for i in range(0, len(data), 32):
                batch = data[i:i+32]
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**inputs)
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

        save_dataframe(df, suffix, "finbert",config)
        score_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        avg = df["label"].map(score_map).mean()
        print(f"✅ FinBERT ({suffix}): {avg:.4f}")
        return avg

# ------------------------- Model Registry -------------------------

class ModelRegistry:
    def __init__(self):
        self.models = {
            "vader": VaderModel(),
            "finbert": FinBERTModel()
        }

    def run_model(self, model_name: str, file_path: str, suffix: str, config=None) -> float:
        if model_name not in self.models:
            raise ValueError(f"Unsupported model: {model_name}")
        return self.models[model_name].run(file_path, suffix, config)

model_registry = ModelRegistry()

# ------------------------- API -------------------------

def run_vader(file_path: str, suffix: str, config=None) -> float:
    return model_registry.run_model("vader", file_path, suffix, config)

def run_finbert(file_path: str, suffix: str, config=None) -> float:
    return model_registry.run_model("finbert", file_path, suffix, config)

def get_sentiment_score(file_path: str, suffix: str, config=None) -> float:
    if config.MODEL_TOGGLE == "vader":
        return run_vader(file_path, suffix, config)
    elif config.MODEL_TOGGLE == "finbert":
        return run_finbert(file_path, suffix, config)
    elif config.MODEL_TOGGLE == "ensemble":
        vader_score = run_vader(file_path, suffix, config)
        finbert_score = run_finbert(file_path, suffix, config)
        ensemble_score = config.VADER_WEIGHT * vader_score + config.FINBERT_WEIGHT * finbert_score
        print(f"✅ Ensemble Score ({suffix}): {ensemble_score:.4f}")
        return ensemble_score
    else:
        raise ValueError(f"Invalid MODEL_TOGGLE in config.py: {config.MODEL_TOGGLE}")
