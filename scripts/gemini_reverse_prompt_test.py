#!/usr/bin/env python3

"""
Quick test script to validate Vertex AI reverse-prompt generation on a single image.

Uses Google Vertex AI (Gemini 2.5 Flash Lite) to produce a structured,
factual reverse prompt describing only what is visible in the image.

Intended as a lightweight verification step before running the full
`generate_reverse_prompts_vertex.py` batch process.

Usage:
  python test_reverse_prompt_single.py

Edit `PROJECT_ID`, `LOCATION`, and `IMAGE_PATH` at the top of the file
to match your environment and local image path.
"""


import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image

PROJECT_ID = "reverse-prompts-from-image"      # <-- e.g. reverse-prompt-dataset-123
LOCATION   = "us-central1"          # or "us-east1"
IMAGE_PATH = "ROOT/label_studio/data/images_flat/robot_birds_00015.png"          # any test image on disk

vertexai.init(project=PROJECT_ID, location=LOCATION)  # uses your JSON key via env var

model = GenerativeModel("gemini-2.5-flash-lite")

prompt = (
    "Create a 30–60 word reverse prompt that describes only what is visible. "
    "Order exactly: Subject; Setting; Style; Composition; Palette; Lighting; Mood; "
    "Details (2–3 nouns); Camera; Post; Avoid. "
    "No lore, brands, or software/model names."
)

with open(IMAGE_PATH, "rb") as f:
    img = Image.from_bytes(f.read())

resp = model.generate_content([prompt, img])
print("\n--- Reverse Prompt ---\n")
print(resp.text.strip())
