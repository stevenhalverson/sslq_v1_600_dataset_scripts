#!/usr/bin/env python3

"""
Scans your dataset CSV for common Hugging Face viewer / parquet issues.

Checks for:
- Missing or empty image paths
- Paths not starting with "images/"
- Files that don't actually exist
- Overlong or strange filenames
- Non-integer IDs / numeric columns
"""

import csv
import os
import re
import sys
from pathlib import Path

# ==== CONFIG =====================================================

# Set these paths to your dataset files
CSV_PATH = Path(r"[INPUT_PATH_CSV]")
IMAGES_DIR = Path(r"[INPUT_Path_IMAGES]")

# =================================================================

def is_intlike(v: str) -> bool:
    try:
        int(float(v))
        return True
    except Exception:
        return False


def main():
    if not CSV_PATH.exists():
        print(f"❌ CSV not found: {CSV_PATH}")
        sys.exit(1)
    if not IMAGES_DIR.exists():
        print(f"⚠️ Warning: image folder not found: {IMAGES_DIR}")

    bad, longnames, badchars, typen = [], [], [], []
    rows_scanned = 0

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "file_name" not in fields:
            print("❌ Missing 'file_name' column.")
            sys.exit(1)

        for i, row in enumerate(reader, start=2):
            rows_scanned += 1
            fn = (row.get("file_name") or "").strip().replace("\\", "/")

            if not fn:
                bad.append((i, "empty file_name"))
                continue

            if not fn.startswith("images/"):
                bad.append((i, f"not under images/: {fn}"))

            img_path = IMAGES_DIR / Path(fn).name
            if not img_path.exists():
                bad.append((i, f"missing file: {fn}"))

            # Check filename sanity
            if len(str(img_path)) > 230:
                longnames.append((i, len(str(img_path)), img_path.name))

            if re.search(r'[^A-Za-z0-9._\- ]', img_path.name):
                badchars.append((i, img_path.name))

            for col in ("id", "quality", "style_match"):
                v = (row.get(col) or "").strip()
                if v and not is_intlike(v):
                    typen.append((i, col, v))

    # === SUMMARY ==================================================
    print(f"\n✅ Scan complete: {rows_scanned} rows checked.")
    print(f" - Missing or bad paths: {len(bad)}")
    print(f" - Overlong paths: {len(longnames)}")
    print(f" - Suspicious characters: {len(badchars)}")
    print(f" - Numeric type issues: {len(typen)}")

    if bad[:5]:
        print("\nExamples of bad paths:")
        for b in bad[:5]:
            print("  line", b[0], "-", b[1])

    if longnames[:3]:
        print("\nExamples of overlong paths:")
        for l in longnames[:3]:
            print("  line", l[0], f"({l[1]} chars)", "-", l[2])

    if badchars[:3]:
        print("\nExamples of suspicious names:")
        for b in badchars[:3]:
            print("  line", b[0], "-", b[1])

    if typen[:3]:
        print("\nExamples of numeric issues:")
        for t in typen[:3]:
            print("  line", t[0], "-", t[1], "=", t[2])

    print("\n--- Done ---\n")


if __name__ == "__main__":
    main()
