
# 📈 EarningsCallAlpha

**A modular financial machine learning pipeline for extracting alpha from earnings call sentiment.**

---

## 🔍 Project Summary & Motivation

**EarningsCallAlpha** is a Python-based research and production-grade pipeline that leverages NLP models to analyze quarterly earnings call transcripts and generate **market-neutral, cross-sectional alpha signals**. It uses a combination of **FinBERT** and **VADER** for sentiment scoring, calculates **quarter-over-quarter sentiment deltas**, and maps those to predictive trading signals, benchmarked against index returns.

This project is motivated by the growing importance of **unstructured data** in financial markets, and aims to build a **scalable, explainable, and modular framework** for earnings-driven alpha generation.

---

## 🧠 Architecture Overview

The pipeline is structured into clean, modular components:

```
EarningsCallAlpha/
│
├── src/
│   ├── cleaning.py             # Text extraction & preprocessing from PDF transcripts
│   ├── sentiment.py            # FinBERT & VADER sentiment scoring
│   ├── signals.py              # Delta computation and signal logic
│   ├── returns.py              # Price data fetch (yfinance/NSE), return & alpha calculation
│   ├── config.py               # Centralized paths and config
│   └── main.py                 # End-to-end runner
│
├── Data/
│   ├── Raw/                    # Raw PDF transcripts
│   └── Processed/              # Cleaned transcript text files
│
├── Outputs/
│   ├── Sentiment/              # Sentiment scores per transcript
│   ├── Signals/                # Generated buy/hold/sell signals
│   └── Returns/                # Alpha CSVs and backtest logs
│
└── requirements.txt
```

---

## ⚙️ Installation & Environment Setup

### 🔁 Colab Setup

```python
# Clone the repo
!git clone https://github.com/nikhil1507/EarningsCallAlpha.git
%cd EarningsCallAlpha

# Install requirements
!pip install -r requirements.txt
```

### 💻 Local Setup

```bash
git clone https://github.com/nikhil1507/EarningsCallAlpha.git
cd EarningsCallAlpha

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

---

## 📂 Data Requirements

### 📁 Input Transcripts

* Location: `Data/Raw/{COMPANY}/`
* Format: PDF files of quarterly earnings call transcripts.
* Naming: Should ideally include quarter hints, e.g., `INFY_Q3_Jan_22.pdf`

### 🏦 Stock & Benchmark Tickers

* Ticker format: Yahoo-compatible (`INFY.NS`, `^NSEI`)
* Supports fallback using NSE API for Indian equities.

---

## 🚀 Running the Full Pipeline

```bash
python src/main.py --company INFY
```

This command performs:

1. Ingestion & cleaning of transcripts.
2. Sentence-level sentiment scoring (FinBERT + VADER).
3. Sentiment delta computation.
4. Signal generation.
5. Price fetch + alpha computation.

Outputs are saved under `Outputs/`.

---

## 🔍 Stage-wise Breakdown

### 1. 🧼 Cleaning

* Extracts clean conversational text from PDFs.
* Uses PyMuPDF and regex to strip headers, disclaimers, and noise.

### 2. 🧠 Sentiment Scoring

* Applies **FinBERT** (finance-specific BERT) and **VADER** (lexicon-based).
* Sentence-level scores are averaged for each quarter.
* Saved in `Outputs/Sentiment/{company}_{quarter}.json`

### 3. 📊 Delta Computation

* Computes quarter-over-quarter sentiment change (ΔFinBERT, ΔVADER).
* Signals are generated based on thresholds and direction of change.

### 4. 📈 Signal Logic

* Simple heuristic rules:
  `buy` if both deltas positive and above threshold,
  `sell` if both negative and below threshold,
  `hold` otherwise.

### 5. 💰 Returns & Alpha

* Fetches price data via `yfinance` .
* Computes **strategy return**, **benchmark return**, and **alpha**.
* Final results saved in CSV: `Outputs/Returns/{company}_alpha.csv`

---

## 🧪 Planned Extensions

* 🔮 **Model Support**:
  Integrate `LMFinance`, `FinGPT`, or fine-tuned LLMs for nuanced sentiment.

* 📈 **Supervised ML Mode**:
  Convert signals into features for XGBoost/LightGBM classifiers or regressors to predict alpha.

* 🧠 **Explainability**:
  Add SHAP/LIME support to interpret sentiment-alpha linkages.

* 🕸 **Multistock Mode**:
  Enable multi-company, cross-sectional predictions for long-short portfolios.

---

## 📊 Logging & Benchmarking

* All stages log execution time and status using Python’s `logging` module.
* Output files include metadata timestamps for reproducibility.

---

## 🐛 Known Issues

* **YFinance NaNs in Colab**:
  Colab occasionally returns empty DataFrames due to timezone/session bugs. Reinstall `yfinance`, and ensure `User-Agent` patch is set:

  ```python
  import yfinance.shared as yfshared
  yfshared._session.headers.update({"User-Agent": "Mozilla/5.0"})
  ```

* **Transcript Date Missing**:
  Ensure dates are extractable from either cleaned or raw PDF (uses regex + PyMuPDF).

---


