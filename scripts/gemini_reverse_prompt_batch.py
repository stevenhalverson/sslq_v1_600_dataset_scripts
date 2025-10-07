#!/usr/bin/env python3

"""
Generates structured "reverse prompts" (image-to-text descriptions) for each image
in the SSLQ (Synthetic Scenic Lore Image Quality) dataset using Google Vertex AI.

Model: Gemini 2.5 Flash Lite

Each image is passed through a standardized instruction prompt that requests
a 30–60 word factual description of visible content, ordered by:
Subject → Setting → Style → Composition → Palette → Lighting → Mood → 
Details → Camera → Post → Avoid. 

Features:
- Resumable CSV writing (skips already processed files)
- Exponential backoff on transient API errors (ResourceExhausted, etc.)
- Configurable image glob, output path, and Vertex AI project settings
- Produces `reverse_prompts.csv` for later integration into metadata

Edit `PROJECT_ID`, `LOCATION`, and `IMAGE_GLOB" the top of the file
to match your environment and local image path.

Usage:
  python generate_reverse_prompt_batch.py
"""

import os, csv, glob, time
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded, ServiceUnavailable

# ---- CONFIG (edit these two) ----
PROJECT_ID = "reverse-prompts-from-image"          # e.g. reverse-prompt-dataset-123456
LOCATION   = "us-central1"              # or "us-east1"
IMAGE_GLOB = r"ROOT\label_studio\data\images_flat\*.*"            # e.g. "images/*.jpg" or r"ROOT\images\*.png"
OUT_CSV    = "reverse_prompts.csv"

PROMPT = (
    "Create a 30–60 word reverse prompt that describes only what is visible. "
    "Order exactly: Subject; Setting; Style; Composition; Palette; Lighting; Mood; "
    "Details (2–3 nouns); Camera; Post; Avoid. "
    "No lore, brands, or software/model names."
)

# ---- INIT ----
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-2.5-flash-lite")

def call_model(img_path, retries=3):
    with open(img_path, "rb") as f:
        img = Image.from_bytes(f.read())
    for attempt in range(retries):
        try:
            resp = model.generate_content([PROMPT, img])
            return (resp.text or "").strip()
        except (ResourceExhausted, DeadlineExceeded, ServiceUnavailable) as e:
            # exponential backoff: 2s, 4s, 8s...
            wait = 2 ** (attempt + 1)
            print(f"  transient error: {e.__class__.__name__}; retry in {wait}s...")
            time.sleep(wait)
    # last attempt raised or returned empty -> raise a clean error
    raise RuntimeError("Max retries exceeded")

def load_done_set(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.reader(f)):
                if i == 0:  # header
                    continue
                if row and row[0]:
                    done.add(row[0])
    return done

def main():
    files = sorted(glob.glob(IMAGE_GLOB))
    print(f"Found {len(files)} images matching: {IMAGE_GLOB}")

    done = load_done_set(OUT_CSV)
    mode = "a" if os.path.exists(OUT_CSV) else "w"
    with open(OUT_CSV, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["file_path", "reverse_prompt"])

        total = len(files)
        processed = 0
        for idx, path in enumerate(files, 1):
            if path in done:
                print(f"[{idx}/{total}] skip (already in CSV): {os.path.basename(path)}")
                continue
            try:
                rp = call_model(path)
                w.writerow([path, rp])
                f.flush()  # write immediately so you can resume safely
                processed += 1
                print(f"[{idx}/{total}] ✓ {os.path.basename(path)}")
                # optional gentle pacing to avoid bursts; adjust or remove:
                time.sleep(0.2)
            except Exception as e:
                print(f"[{idx}/{total}] × {os.path.basename(path)} -> {e}")

    print(f"\nDone. Added {processed} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
