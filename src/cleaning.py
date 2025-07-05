import os
import re
import logging
from nltk.tokenize import sent_tokenize
from collections import Counter
import fitz  # PyMuPDF
import nltk
nltk.download('punkt', quiet=True)


# Ensure NLTK sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r'[^a-zA-Z0-9\-]', '_', name)

def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        logging.error(f"Failed to extract PDF text: {e}")
        raise

def read_file(file_path):
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        encodings = ['utf-8', 'windows-1252', 'latin-1', 'utf-16']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Cannot decode file: {file_path}")
    else:
        raise ValueError("Only .pdf and .txt files supported")

def clean_transcript(full_text, filename):
    start_terms = ["Moderator", "Operator", "Host", "Conference Call Facilitator", "PRESENTATION", "Final Transcript"]
    noise_patterns = [
        r"©.*?\d{4}.*?",
        r".*?(Earnings Conference Call|Transcript of|Republished with permission).*",
        r".*? - \d+ -",
        r".*?Event Date/Time:.*",
        r".*?Transcription:.*",
        r".*?(Good (morning|afternoon|evening) and (thank you|welcome)|Before I hand over|Recording of this call|Thank you.*for joining|In the interest of time).*"
    ]

    lines = full_text.splitlines()
    start_index = next((i for i, line in enumerate(lines) if any(term in line for term in start_terms)), 0)
    transcript_lines = lines[start_index:]

    line_counts = Counter(transcript_lines)
    frequent_lines = [line for line, count in line_counts.items() if count > 2 and len(line.strip()) < 150]

    clean_lines = [
        line for line in transcript_lines
        if not any(re.search(p, line, re.IGNORECASE) for p in noise_patterns)
        and line.strip() not in frequent_lines
        and not line.strip().isdigit()
        and len(line.strip()) > 0
    ]

    speaker_blocks = []
    current_speaker = current_role = None
    current_text = []
    role_pattern = r"(CEO|CFO|Director|President|Head of|Analyst|Investor Relations|Managing Director|Vice President|Chairman)"

    for line in clean_lines:
        match = re.match(r"(\w+.*?)?:?\s*(.*?)(?: - (.*?))?(?::.*)?$", line)
        if match and len(match.group(2).strip()) < 50 and all(word[0].isupper() for word in match.group(2).split() if word):
            speaker = match.group(2).strip()
            role = re.search(role_pattern, match.group(3) or "", re.IGNORECASE)
            role = role.group(1) if role else None
            speaker = re.sub(r"\s*(–|-)\s*" + role_pattern + ".*", "", speaker, flags=re.IGNORECASE).strip()
            if current_speaker:
                speaker_blocks.append((current_speaker, current_role, " ".join(current_text)))
            current_speaker, current_role, current_text = speaker, role, [line]
        else:
            current_text.append(line)

    if current_speaker:
        speaker_blocks.append((current_speaker, current_role, " ".join(current_text)))

    procedural_patterns = r"(thank you (for joining|everyone|all)|hand over|remind.*participants|next question|bring this call to a close|please go ahead|now take.*questions)"
    relevant_blocks = [
        (speaker, role, text) for speaker, role, text in speaker_blocks
        if not re.search(procedural_patterns, text.lower(), re.IGNORECASE)
    ]

    final_output = []
    sanitized_filename = sanitize_filename(filename)
    sentence_idx = 1

    for speaker, role, text in relevant_blocks:
        cleaned_text = re.sub(r'^\s*[-•–]\s+', '', text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        for sentence in sent_tokenize(cleaned_text):
            if sentence.strip():
                role_suffix = f" ({role})" if role else ""
                final_output.append(f"{sanitized_filename}_{sentence_idx} | {speaker}{role_suffix}: {sentence}")
                sentence_idx += 1

    return "\n".join(final_output)

def process_and_save(file_path: str, output_path: str):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # ← THIS LINE IS CRUCIAL
        full_text = read_file(file_path)
        if not full_text.strip():
            raise ValueError("Transcript content is empty")
        cleaned = clean_transcript(full_text, file_path)
        if not cleaned.strip():
            raise ValueError("Cleaned transcript is empty")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        logging.info(f"✅ Cleaned and saved: {output_path}")
    except Exception as e:
        logging.error(f"❌ Failed to clean transcript {file_path}: {e}")
        raise


