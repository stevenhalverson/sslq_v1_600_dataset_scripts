#!/usr/bin/env python3

"""
Fix long/unsafe image filenames and update metadata.csv accordingly.

- Shortens basenames to <= MAX_BASENAME_LEN (default 64)
- Keeps extension
- ASCII-only, lowercase, [a-z0-9._-]
- Ensures uniqueness via 8-char hash suffix
- Writes:
    * metadata.backup-YYYYMMDD-HHMMSS.csv
    * rename_mapping.csv  (old_path,new_path)
- Works with a single image column: file_name
- Only flattens/targets files in images/ (no subdirs)

Set DRY_RUN=True to preview without changing anything.
"""

import csv, hashlib, re, unicodedata
from datetime import datetime
from pathlib import Path

# ====== CONFIG ======
CSV_PATH  = Path(r"[INSERT_FILEPATH_CSV]")
IMAGES_DIR = Path(r"[INSERT_FILEPATH_IMAGES]")
MAX_BASENAME_LEN = 64      # safe length for basename (not full path)
DRY_RUN = True             # set to False to actually rename & write CSV
# =====================

def ascii_slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "img"

def make_new_name(stem: str, ext: str, salt: str) -> str:
    # ensure uniqueness with deterministic 8-char hash of original path
    h = hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8]
    room = max(1, MAX_BASENAME_LEN - 1 - len(h))  # space for "-<hash>"
    base = ascii_slug(stem)[:room]
    return f"{base}-{h}{ext.lower()}"

def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")
    if not IMAGES_DIR.exists():
        raise SystemExit(f"images folder not found: {IMAGES_DIR}")

    rows = list(csv.DictReader(CSV_PATH.open("r", encoding="utf-8", newline="")))
    if "file_name" not in rows[0]:
        raise SystemExit("CSV must contain a 'file_name' column (single image column).")

    # outputs
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_backup = CSV_PATH.with_name(f"metadata.backup-{stamp}.csv")
    map_path   = CSV_PATH.with_name("rename_mapping.csv")

    to_write_rows = []
    mapping = []
    renamed = 0
    missing = 0

    for r in rows:
        rel = (r.get("file_name") or "").replace("\\", "/").strip()
        if not rel:
            to_write_rows.append(r)
            continue

        src = IMAGES_DIR / Path(rel).name
        if not src.exists():
            missing += 1
            to_write_rows.append(r)
            continue

        stem, ext = src.stem, src.suffix
        new_name = make_new_name(stem, ext, salt=str(src))

        if new_name != src.name:
            mapping.append((src.name, new_name))
            if not DRY_RUN:
                (IMAGES_DIR / src.name).rename(IMAGES_DIR / new_name)
                renamed += 1

        # update CSV path
        r["file_name"] = f"images/{new_name}"
        to_write_rows.append(r)

    print(f"\nScan complete: {len(rows)} rows")
    print(f" - Files to rename (incl. dry-run): {len(mapping)}")
    print(f" - Missing files: {missing}")
    if mapping[:5]:
        print(" examples:", mapping[:5])

    # Write outputs
    if DRY_RUN:
        print("\nDRY_RUN=True → no files renamed, CSV not overwritten.")
        print("Set DRY_RUN=False and run again to apply changes.")
        return

    # backup original CSV
    csv_backup.write_text(CSV_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Backed up original CSV to {csv_backup.name}")

    # mapping
    with map_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_name","new_name"])
        for a,b in mapping:
            w.writerow([a,b])
    print(f"Wrote rename mapping: {map_path.name} ({len(mapping)} entries)")

    # write updated CSV (preserve original columns; re-sequence id if present)
    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(to_write_rows, 1):
            if "id" in r and r["id"] not in ("", None):
                # keep existing id if present; else resequence
                pass
            elif "id" in r:
                r["id"] = str(i)
            w.writerow(r)

    print(f"✅ Renamed {renamed} files and updated {CSV_PATH.name}\n")

if __name__ == "__main__":
    main()
