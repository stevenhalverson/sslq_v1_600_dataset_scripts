#!/usr/bin/env python3

"""
Dataset Preparation for LoRA Training
-------------------------------------
This script filters a dataset based on specific keywords (STRONG_TERMS),
cleans up description text, creates caption files (.txt), and organizes 
images into an output directory for training.

Features:
- Scans ALL columns (Description, Notes, Tags, etc.)
- Matches ONLY if exact phrases from STRONG_TERMS are found.
- Scrubs "junk" phrases and quality artifacts (e.g., "4k", "blurry").
- Injects a specific trigger word and/or concept tag into the captions.
"""

import csv
import os
import re
import shutil
from pathlib import Path
from typing import Dict

# ==========================================
#              CONFIGURATION
# ==========================================

# Paths (Update these to match your local structure)
INPUT_CSV_PATH = "./dataset/metadata.csv"
OUTPUT_DIR = "./training_data"
IMAGE_SOURCE_DIR = "./dataset/images"

# Column Mapping
FILE_COLUMN = "file_name"
TRIAGE_COLUMN = "triage"
TERM_SOURCE_FIELDS = [
    "category", "attributes", "tags", "composition", "mood", "palette", "notes"
]

# Filtering Thresholds
STYLE_MATCH_THRESHOLD = 1
QUALITY_THRESHOLD = 1
ADDITIONAL_FILTERS: Dict[str, tuple] = {} 

# Trigger Word (The main activation token for your LoRA)
# Leave empty string "" if you do not want a trigger word at the start.
TRIGGER_WORD = "my_trigger_word" 

# Mandatory Concept Tag
# If this string is missing from the description, it will be appended to the end.
# Useful if you are training a specific concept (e.g., "stone ocean") and want to ensure consistency.
MANDATORY_CONCEPT = "stone ocean" 

# ==========================================
#           STRONG TERMS (FILTER)
# ==========================================
# The script will search for ANY of these keys or their synonyms in the CSV data.
# If a match is found, the image is INCLUDED in the training set.
# Format: "Term": (weight, [synonyms])
STRONG_TERMS = {
    "Stone ocean": (1.75, ["stone_ocean", "stone ocean"]),
    # Add more terms here (see template file for examples)
}


# ==========================================
#           CLEANING LISTS
# ==========================================

# Phrases to strip out completely
JUNK_PHRASES = [
    "other category", "other tags", "other attributes", 
    "other mood", "other palette", "other composition", 
    "misc", "miscellaneous", "bad quality"
]

# Quality/Artifact terms to scrub from sentences
ISSUE_TERMS = {
    "blur", "blurry", "blurred", "blur_noise", "noise", "noisy",
    "artifact", "artifacts", "artifacting", "watermark", "watermarked",
    "watermark_text", "cropped", "crop", "duplicate", "duplicated",
    "jpeg", "compression", "banding", "pixelation", "pixelated",
}

# ==========================================
#              LOGIC
# ==========================================

def scrub_junk_phrases(text: str) -> str:
    if not text: return text
    for phrase in JUNK_PHRASES:
        pattern = r'\b' + re.escape(phrase) + r'\b(?:,\s*)?'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r',\s*,', ',', text)
    return text

def _normalize_phrase_spaces_lower(s: str) -> str:
    return s.strip().replace("_", " ").lower()

def build_strong_index(terms):
    lookup = {}
    for term, (w, syns) in terms.items():
        variants = {_normalize_phrase_spaces_lower(term)}
        variants |= {_normalize_phrase_spaces_lower(x) for x in syns}
        for v in variants:
            lookup[v] = (v, w) 
    keys_sorted = sorted(lookup.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys_sorted) + r")\b", re.IGNORECASE)
    return lookup, pattern

STRONG_LOOKUP, STRONG_PATTERN = build_strong_index(STRONG_TERMS)

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

def scrub_quality_language(text: str):
    if not text: return text, False
    removed = False; kept = []
    for sent in _SENTENCE_SPLIT.split(text):
        if any(tok in sent.lower() for tok in ISSUE_TERMS):
            removed = True
            continue
        kept.append(sent)
    return " ".join(s.strip() for s in kept if s.strip()), removed

def sanitize_text(text):
    if text is None: return ""
    s = str(text).strip()
    if s.lower() in {"nan", "none", ""}: return ""
    s = re.sub(r'^"|"$', "", s)
    s = re.sub(r"\s+", " ", s)
    s = s[0].upper() + s[1:] if s else s
    if s and not s.endswith(('.', '!', '?')): s += '.'
    return s

def extract_strong_terms(row):
    bucket = []; seen = set()
    for field in TERM_SOURCE_FIELDS:
        v = row.get(field)
        if not v: continue
        for val in re.split(r"[;,]\s*", str(v)):
            val = val.strip()
            if not val: continue
            human_val = val.replace("_", " ")
            key = _normalize_phrase_spaces_lower(human_val)
            if any(tok in key for tok in ISSUE_TERMS): continue
            if key not in seen:
                seen.add(key); bucket.append(human_val)
    return " ".join(bucket).strip()

def apply_term_weighting(text: str):
    if not text: return ""
    out = []; last = 0
    for m in STRONG_PATTERN.finditer(text):
        out.append(text[last:m.start()])
        matched = m.group(0)
        key = _normalize_phrase_spaces_lower(matched)
        syntax, _ = STRONG_LOOKUP.get(key, (matched, 1.0))
        out.append(syntax)
        last = m.end()
    out.append(text[last:])
    return "".join(out)

def generate_description(row):
    desc = ""
    # Prioritize LLM description, fallback to human
    for field in ["llm_description", "human_description"]:
        v = row.get(field)
        if v:
            desc = sanitize_text(v)
            if desc: break
            
    extra = extract_strong_terms(row)
    combined = f"{desc} {extra}".strip() if extra else desc
    weighted = apply_term_weighting(combined)
    cleaned, _ = scrub_quality_language(weighted)
    cleaned = scrub_junk_phrases(cleaned)
    return cleaned

def matches_filters(row, min_style, min_quality, extra_filters):
    # 1. Gather all text data
    combined_text = (
        str(row.get("file_name", "")) + " " + 
        str(row.get("category", "")) + " " + 
        str(row.get("tags", "")) + " " + 
        str(row.get("human_description", "")) + " " + 
        str(row.get("llm_description", "")) + " " + 
        str(row.get("notes", "")) + " " +
        str(row.get("attributes", "")) + " " +
        str(row.get("mood", "")) + " " +
        str(row.get("palette", ""))
    ).lower()

    # 2. Strong Term Lookup
    found_match = False
    for term, (weight, synonyms) in STRONG_TERMS.items():
        if term.lower() in combined_text:
            found_match = True
            break
        for syn in synonyms:
            if syn.lower() in combined_text:
                found_match = True
                break
    
    if not found_match:
        return False

    # 3. Quality Check
    try:
        s_score = float(row.get("style_match", 0))
        q_score = float(row.get("quality", 0))
        if s_score < min_style or q_score < min_quality:
            return False
    except (ValueError, TypeError):
        pass

    return True

def process_csv():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    counts = {"total": 0, "written": 0}

    try:
        with open(INPUT_CSV_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                counts["total"] += 1
                
                # Filter check
                if not matches_filters(row, STYLE_MATCH_THRESHOLD, QUALITY_THRESHOLD, ADDITIONAL_FILTERS):
                    continue

                rel_path = str(row.get(FILE_COLUMN, "")).strip()
                if not rel_path: continue
                
                src = Path(IMAGE_SOURCE_DIR) / os.path.normpath(rel_path)
                if not src.exists():
                    print(f"Warning: Missing image file at {src}")
                    continue

                desc = generate_description(row)
                if not desc: continue

                # Apply Trigger Word
                if TRIGGER_WORD and TRIGGER_WORD not in desc:
                    desc = f"{TRIGGER_WORD} {desc}"

                # Ensure Mandatory Concept Tag is present
                if MANDATORY_CONCEPT and MANDATORY_CONCEPT.lower() not in desc.lower():
                    desc = desc.strip() + " " + MANDATORY_CONCEPT

                # Final Text Polish (Whitespace/Punctuation)
                desc = re.sub(r',\s+\.', '.', desc)
                desc = re.sub(r'\s+\.\s+', ' ', desc)
                desc = re.sub(r'\s+\.', '.', desc)
                desc = re.sub(r'\.\.+', '.', desc)
                desc = re.sub(r'\s+', ' ', desc).strip()

                # Write Output
                txt_path = (Path(OUTPUT_DIR) / rel_path).with_suffix(".txt")
                img_path = txt_path.with_suffix(src.suffix)
                txt_path.parent.mkdir(parents=True, exist_ok=True)

                with open(txt_path, "w", encoding="utf-8") as fo:
                    fo.write(desc)
                try:
                    shutil.copy2(src, img_path)
                    counts["written"] += 1
                except Exception as e:
                    print(f"Error copying {src}: {e}")

    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {INPUT_CSV_PATH}")
        return

    print(f"Done! Processed {counts['total']} rows. Created {counts['written']} training pairs.")

if __name__ == "__main__":
    process_csv()