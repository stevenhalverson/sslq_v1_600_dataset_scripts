#!/usr/bin/env python3

"""
Better syntax and cleaner code than generate_sslq_captions_lora_public_light_training.py. Use this if possible.
Suitable for training on RTX 5090, with heavy emphasis on text annotations. 

Features:
- Filters images and annotations by style match and quality values
- Creates new folder for images and annotations, with annotations in .txt format ready for training
- Scans for keyword terms and assigns weight values
- Merges human_description and llm_description, plus tags and column categories, into one cohesive text annotation including weights
- Avoids dupes of weights in .txt files to avoid complications during training
"""



import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

# ===== DEFAULT CONFIG (can be overridden by CLI) =====

INPUT_CSV_PATH = "path/to/your/metadata.csv"
OUTPUT_DIR = "path/to/your/dataset"
IMAGE_SOURCE_DIR = "path/to/your/images"
FILE_COLUMN = "file_name"
TRIAGE_COLUMN = "triage"



PRIMARY_DESCRIPTION_FIELDS = [
    "llm_description",
    "human_description",
]

TERM_SOURCE_FIELDS = [
    "category",
    "attributes",
    "tags",
    "composition",
    "mood",
    "palette",
    "notes",
    "human_description",
    "llm_description",
]

STYLE_MATCH_THRESHOLD = 3
QUALITY_THRESHOLD = 3
ADDITIONAL_FILTERS: Dict[str, tuple] = {}  # e.g., {"category": ("portrait","landscape")}

# ===== STRONG TERM WEIGHTING =====
# Term: (base_weight, synonyms)
STRONG_TERMS = {
    "Stone ocean": (1.5, ["stone_ocean", "stone ocean"]),
    "Columbia alternata": (1.4, ["columbia_alternata"]),
    "blue": (1.4, ["azure", "cobalt", "cerulean"]),
    "pyramids": (1.4, ["pyramid", "giza", "egyptian_pyramids"]),
    "Space": (1.4, ["cosmic", "galactic", "astral", "celestial"]),
    "gold": (1.5, ["golden"]),
    "lore": (1.5, ["story", "scenic"]),
}
TERM_WEIGHT_FORMAT = "({term}:{weight:.1f})"  # Kohya/LoRA weight format

# ===== STRONG TERM INDEX (built once) =====
def _normalize_phrase_spaces_lower(s: str) -> str:
    return s.strip().replace("_", " ").lower()

def _snake_lower(s: str) -> str:
    return _normalize_phrase_spaces_lower(s).replace(" ", "_")

def build_strong_index(terms: Dict[str, Tuple[float, list]]):
    """
    Returns:
      lookup: dict key=normalized (spaces, lower) -> (snake_lower, weight)
      pattern: compiled regex that matches the longest phrases first
    """
    lookup: Dict[str, Tuple[str, float]] = {}
    for term, (w, syns) in terms.items():
        variants = {_normalize_phrase_spaces_lower(term)}
        variants |= {_normalize_phrase_spaces_lower(x) for x in syns}
        for v in variants:
            # If duplicates collide, keep the higher weight to be safe
            snake = v.replace(" ", "_")
            prev = lookup.get(v)
            if prev is None or w > prev[1]:
                lookup[v] = (snake, w)

    # Sort by length desc to avoid partial matches (e.g., "gold" inside "golden")
    keys_sorted = sorted(lookup.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys_sorted) + r")\b", re.IGNORECASE)
    return lookup, pattern

STRONG_LOOKUP, STRONG_PATTERN = build_strong_index(STRONG_TERMS)

# ===== UTILITIES =====
def sanitize_text(text):
    if text is None:
        return ""
    s = str(text).strip()
    if s.lower() in {"nan", "none", ""}:
        return ""
    s = re.sub(r'^"|"$', "", s)           # strip surrounding quotes
    s = re.sub(r"\s+", " ", s)            # collapse whitespace
    # Capitalize first character only; don't smash acronyms
    s = s[0].upper() + s[1:] if s else s
    if s and not s.endswith(('.', '!', '?')):
        s += '.'
    return s

def extract_strong_terms(row):
    """Extract terms from multiple fields, dedup (space-lower) to reduce spam."""
    bucket = []
    seen = set()
    for field in TERM_SOURCE_FIELDS:
        v = row.get(field)
        if not v:
            continue
        # semicolon or comma separated
        values = re.split(r"[;,]\s*", str(v))
        for val in values:
            val = val.strip()
            if not val:
                continue
            human_val = val.replace("_", " ")
            key = _normalize_phrase_spaces_lower(human_val)
            if key not in seen:
                seen.add(key)
                bucket.append(human_val)
    return " ".join(bucket).strip()

def apply_term_weighting(text: str):
    """
    Keep the matched token verbatim in the sentence, and add a single weighted token
    (snake_case, lower) the first time it appears to avoid over-weighting.
    """
    if not text:
        return ""
    processed_snake = set()
    out = []
    last = 0
    for m in STRONG_PATTERN.finditer(text):
        out.append(text[last:m.start()])
        matched = m.group(0)
        key = _normalize_phrase_spaces_lower(matched)
        snake, w = STRONG_LOOKUP.get(key, (_snake_lower(matched), 1.0))
        # Only add the (term:weight) once per unique snake token
        if snake not in processed_snake:
            weighted = f"{matched} {TERM_WEIGHT_FORMAT.format(term=snake, weight=w)}"
            out.append(weighted)
            processed_snake.add(snake)
        else:
            out.append(matched)
        last = m.end()
    out.append(text[last:])
    return "".join(out)

def matches_filters(row, min_style: float, min_quality: float, extra_filters: Dict[str, tuple]):
    try:
        if float(row.get("style_match", 0)) < float(min_style):
            return False
        if float(row.get("quality", 0)) < float(min_quality):
            return False
    except (ValueError, TypeError):
        return False

    for column, allowed_values in (extra_filters or {}).items():
        if not allowed_values:
            continue
        col_value = str(row.get(column, "")).lower()
        if not any(str(v).lower() in col_value for v in allowed_values):
            return False
    return True

def generate_description(row):
    # prefer llm_description then human_description (as configured)
    desc = ""
    for field in PRIMARY_DESCRIPTION_FIELDS:
        v = row.get(field)
        if v:
            desc = sanitize_text(v)
            if desc:
                break

    extra = extract_strong_terms(row)
    combined = f"{desc} {extra}".strip() if extra else desc
    return apply_term_weighting(combined)

# ===== CORE =====
def process_csv(
    csv_path,
    output_dir,
    image_source_dir,
    file_col,
    triage_col,
    min_style,
    min_quality,
    extra_filters,
    dry_run=False,
):
    output_dir = Path(output_dir)
    image_source_dir = Path(image_source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "total_rows": 0,
        "passed_filters": 0,
        "annotations_written": 0,
        "images_copied": 0,
        "missing_images": 0,
        "copy_errors": 0,
        "empty_descriptions": 0,
    }

    examples = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts["total_rows"] += 1

            if str(row.get(triage_col, "")).strip().lower() == "skip":
                continue
            if not matches_filters(row, min_style, min_quality, extra_filters):
                continue

            counts["passed_filters"] += 1

            rel_path_raw = str(row.get(file_col, "")).strip()
            if not rel_path_raw:
                continue
            relative_img_path = os.path.normpath(rel_path_raw)

            source_img_path = image_source_dir / relative_img_path
            if not source_img_path.exists():
                print(f"⚠️ Missing image: {source_img_path}")
                counts["missing_images"] += 1
                continue

            description = generate_description(row)
            if not description:
                print(f"⚠️ Empty description for: {relative_img_path}")
                counts["empty_descriptions"] += 1
                continue

            # Prepare output .txt next to the mirrored path under OUTPUT_DIR
            txt_output_path = (output_dir / relative_img_path).with_suffix(".txt")
            img_output_path = txt_output_path.with_suffix(source_img_path.suffix)  # keep original extension
            txt_output_path.parent.mkdir(parents=True, exist_ok=True)

            if not dry_run:
                # Write caption
                with open(txt_output_path, "w", encoding="utf-8") as fo:
                    fo.write(description)
                counts["annotations_written"] += 1

                # Copy image preserving metadata
                try:
                    shutil.copy2(source_img_path, img_output_path)
                    counts["images_copied"] += 1
                except Exception as e:
                    print(f"⚠️ Copy failed for {source_img_path} -> {img_output_path}: {e}")
                    counts["copy_errors"] += 1

            if len(examples) < 3:
                examples.append(
                    {
                        "source_img": str(source_img_path),
                        "out_txt": str(txt_output_path),
                        "out_img": str(img_output_path),
                        "description_preview": description[:180],
                    }
                )

    # Report
    print("\n===== DATASET CREATION REPORT =====")
    for k, v in counts.items():
        print(f"{k.replace('_',' ').title()}: {v}")
    print("\nStrong terms applied (canonical → weight):")
    for term, (w, syns) in STRONG_TERMS.items():
        print(f"- {term} → {w}  (synonyms: {syns})")
    print(f"\nFinal dataset location: {output_dir.resolve()}")
    print("Dataset is ready for LoRA training!" if not dry_run else "(dry-run complete; no files written)")

    report = {
        "counts": counts,
        "output_dir": str(output_dir.resolve()),
        "examples": examples,
        "min_style": min_style,
        "min_quality": min_quality,
        "file_column": file_col,
        "triage_column": triage_col,
    }
    # write a small machine-readable summary alongside the output
    try:
        if not dry_run:
            with open(output_dir / "report.json", "w", encoding="utf-8") as jf:
                json.dump(report, jf, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Could not write report.json: {e}")

    return counts

# ===== CLI =====
def parse_args():
    p = argparse.ArgumentParser(description="Build Kohya/LoRA-ready dataset (captions + images).")
    p.add_argument("--input_csv", default=INPUT_CSV_PATH, help="Path to metadata CSV")
    p.add_argument("--out", default=OUTPUT_DIR, help="Output dataset directory")
    p.add_argument("--images", default=IMAGE_SOURCE_DIR, help="Root folder of source images")
    p.add_argument("--file_col", default=FILE_COLUMN, help="CSV column with relative image path")
    p.add_argument("--triage_col", default=TRIAGE_COLUMN, help='CSV column for triage (e.g., "skip")')
    p.add_argument("--min_style", type=float, default=STYLE_MATCH_THRESHOLD, help="Minimum style_match to include")
    p.add_argument("--min_quality", type=float, default=QUALITY_THRESHOLD, help="Minimum quality to include")
    p.add_argument("--dry_run", action="store_true", help="Do not write files; print report only")
    return p.parse_args()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Support running without CLI flags (uses defaults above)
        process_csv(
            INPUT_CSV_PATH,
            OUTPUT_DIR,
            IMAGE_SOURCE_DIR,
            FILE_COLUMN,
            TRIAGE_COLUMN,
            STYLE_MATCH_THRESHOLD,
            QUALITY_THRESHOLD,
            ADDITIONAL_FILTERS,
            dry_run=False,
        )
    else:
        args = parse_args()
        process_csv(
            args.input_csv,
            args.out,
            args.images,
            args.file_col,
            args.triage_col,
            args.min_style,
            args.min_quality,
            ADDITIONAL_FILTERS,
            dry_run=args.dry_run,
        )
