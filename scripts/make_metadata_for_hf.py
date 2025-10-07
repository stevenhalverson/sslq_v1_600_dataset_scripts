#!/usr/bin/env python3

"""
Converts a Label Studio JSON export into a standardized metadata CSV
for the SSLQ (Synthetic Scenic Lore Image Quality) dataset pipeline.

Features:
- Extracts annotations and task data into a clean table matching Hugging Face conventions.
- Maps multiple Label Studio field names (textareas, choices, numbers) into unified columns.
- Recovers relative image paths for 'file_name' to align with dataset storage.
- Outputs metadata.csv with consistent column ordering for analysis or upload.

Usage:
  python ls_export_to_metadata.py \
      --ls_export label_studio_export.json \
      --images_root path/to/images_flat \
      --out metadata.csv
"""

import json, argparse, re, urllib.parse, csv
from pathlib import Path

# ---- OUTPUT SCHEMA (order) ----
COLUMNS = [
    "id",
    "file_name",
    "attributes",
    "category",
    "composition",
    "human_description",
    "issue",
    "mood",
    "notes",
    "palette",
    "quality",
    "llm_description",
    "style_match",
    "triage",
]

# Map OUTPUT KEY  ->  possible Label Studio from_name keys (lowercased)
FIELD_MAP = {
    "attributes":        ["attributes", "attrs"],
    "category":          ["category", "class", "label"],
    "composition":       ["composition", "comp"],
    "human_description": ["human_description", "description", "desc", "human_desc"],
    "issue":             ["issue", "issues", "quality_issue"],
    "mood":              ["mood", "vibe"],
    "notes":             ["notes", "note", "comment", "comments"],
    "palette":           ["palette", "color_palette", "colors"],
    "quality":           ["quality", "score", "rating", "aesthetic", "stars"],
    "llm_description":   ["reverse_prompt", "rev_prompt", "revprompt", "prompt_reverse"],
    "style_match":       ["style_match", "style", "match"],
    "triage":            ["triage", "flag", "status"],
}

def pick_from_results(result_list, want_names):
    want = set(x.lower() for x in want_names)
    for r in result_list:
        fn = (r.get("from_name") or "").lower()
        if fn not in want:
            continue
        t = (r.get("type") or "").lower()
        v = r.get("value") or {}
        if t == "textarea":
            vals = v.get("text", [])
            return vals[0] if vals else ""
        if t == "choices":
            vals = v.get("choices", [])
            # join multiselect into a readable string
            return "; ".join(vals) if vals else ""
        if t in ("number", "rating"):
            num = v.get("number", v.get("rating", ""))
            return "" if num is None else str(num)
        if t == "text":
            val = v.get("text")
            if isinstance(val, list):
                return val[0] if val else ""
            return val or ""
        # fallback: stringify whatever came through
        return json.dumps(v, ensure_ascii=False)
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls_export", required=True, help="Label Studio JSON export (Classic format)")
    ap.add_argument("--images_root", required=True, help="Root folder where images live")
    ap.add_argument("--out", required=True, help="Output CSV path (metadata.csv)")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    tasks = json.loads(Path(args.ls_export).read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit("Expected a list of tasks in the Label Studio export")

    rows = []
    for idx, task in enumerate(tasks, start=1):

        # ----- Build file_name (HF expects the relative image path in CSV) -----
        img_url = task.get("data", {}).get("image") or task.get("data", {}).get("img") or ""
        img_url = urllib.parse.unquote(img_url)                       # decode %5C, etc.
        img_url = re.sub(r"^/data/local-files/\?d=", "", img_url)     # strip LS local prefix
        img_url = img_url.replace("\\", "/")                          # backslashes -> slashes
        just_name = Path(img_url).name

        # If images are nested, try to recover the subpath under images_root
        rel = None
        for p in images_root.rglob(just_name):
            rel = p.relative_to(images_root).as_posix()
            break
        if rel is None:
            rel = just_name

        # Pull the first annotation/completion results
        ann = (task.get("annotations") or task.get("completions") or [])
        result_list = (ann[0].get("result", []) if ann else [])

        # Initialize record with id + file_name
        rec = {k: "" for k in COLUMNS}
        rec["id"] = str(idx)
        rec["file_name"] = f"images/{rel}"

        # Populate mapped fields from results; fallback to task["data"] if needed
        for out_key, candidates in FIELD_MAP.items():
            val = pick_from_results(result_list, candidates)
            if not val:
                d = task.get("data", {})
                for c in candidates:
                    if c in d and d[c] not in (None, ""):
                        val = str(d[c])
                        break
            rec[out_key] = val

        rows.append(rec)

    # ----- Write CSV -----
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows → {outp}")

if __name__ == "__main__":
    main()
