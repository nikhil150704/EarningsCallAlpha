# src/cleaning.py

import re
import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_text_from_pdf(path: Path) -> str:
    try:
        with fitz.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        raise RuntimeError(f"❌ PDF extraction failed for {path.name}: {e}")

def read_text_file(path: Path) -> str:
    for encoding in ['utf-8', 'windows-1252', 'latin-1', 'utf-16']:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"❌ Could not decode file {path.name}")

def load_transcript(path: Path) -> str:
    if path.suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif path.suffix == ".txt":
        return read_text_file(path)
    else:
        raise ValueError(f"❌ Unsupported file format: {path.suffix}")

def remove_noise(lines: List[str]) -> List[str]:
    noise_patterns = [
        r"©.*?\d{4}.*?",
        r".*(Earnings Conference Call|Transcript of|Republished).*",
        r".*- \d+ -.*",
        r".*Event Date/Time:.*",
        r".*Transcription:.*",
        r".*(thank you.*joining|Recording of this call|Q&A|next question).*",
    ]
    line_counts = Counter(lines)
    frequent_lines = {line for line, count in line_counts.items() if count > 2 and len(line) < 100}

    return [
        line for line in lines
        if not any(re.search(pat, line, re.IGNORECASE) for pat in noise_patterns)
        and line.strip() not in frequent_lines
        and not line.strip().isdigit()
        and len(line.strip()) > 0
    ]

def parse_speakers(lines: List[str], filename: str) -> List[str]:
    output = []
    idx = 1
    for line in lines:
        sentences = sent_tokenize(line.strip())
        for sent in sentences:
            if sent:
                output.append(f"{filename}_{idx} | {sent}")
                idx += 1
    return output

def clean_transcript(text: str, filename: str) -> str:
    lines = text.splitlines()
    start_index = next((i for i, line in enumerate(lines) if "Moderator" in line or "Operator" in line), 0)
    main_lines = lines[start_index:]
    clean_lines = remove_noise(main_lines)
    parsed = parse_speakers(clean_lines, Path(filename).stem)
    return "\n".join(parsed)

def process_and_save(input_path: Path, output_path: Path):
    try:
        text = load_transcript(input_path)
        if not text.strip():
            raise ValueError("Empty transcript")
        cleaned = clean_transcript(text, input_path.name)
        if not cleaned.strip():
            raise ValueError("Empty output after cleaning")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned, encoding="utf-8")
        logging.info(f"✅ Cleaned and saved: {output_path}")
    except Exception as e:
        logging.error(f"❌ Failed to clean {input_path.name}: {e}")
        raise
