#!/usr/bin/env python3

"""
Utility script to flatten a nested image directory into Label Studio import files.
Used during SSLQ dataset preparation to merge category subfolders into a single flat structure.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import argparse
import csv
import json
import sys
import os
from typing import List, Tuple, Dict, Iterable
from urllib.request import pathname2url

# Common image & media extensions Label Studio can handle (extend if needed)
DEFAULT_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'}

@dataclass
class Task:
    image_uri: str
    label: str
    rel_dir: str  # relative directory (no filename), POSIX style

def file_uri(path: Path) -> str:
    # Cross-platform file:// URI
    return "file://" + pathname2url(str(path.resolve()))

def find_images(root: Path, exts: Iterable[str]) -> List[Path]:
    exts = {e.lower() for e in exts}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def rel_parts(root: Path, file_path: Path) -> Tuple[str, List[str]]:
    rel = file_path.relative_to(root)
    parts = list(rel.parts)
    fname = parts[-1]
    dirs = parts[:-1]
    return fname, dirs

def label_from_dirs(dirs: List[str], depth: int, fallback: str) -> str:
    if depth <= 0:
        return fallback
    if len(dirs) >= depth:
        return "/".join(dirs[:depth])
    return "/".join(dirs) if dirs else fallback

def to_posix(s: str) -> str:
    return s.replace("\\", "/")

def build_tasks(root: Path, images: List[Path], label_depth: int, absolute_paths: bool, root_label: str) -> List[Task]:
    tasks: List[Task] = []
    for img in images:
        fname, dirs = rel_parts(root, img)
        rel_dir = "/".join(dirs) if dirs else ""
        label = label_from_dirs(dirs, label_depth, root_label)
        image_uri = file_uri(img) if absolute_paths else to_posix(str(Path(*([*dirs, fname]))))
        tasks.append(Task(image_uri=image_uri, label=label, rel_dir=rel_dir))
    return tasks

def write_json(tasks: List[Task], out_path: Path) -> None:
    payload = []
    for t in tasks:
        payload.append({
            "data": {"image": t.image_uri},
            "meta": {"folder_label": t.label, "rel_dir": t.rel_dir}
        })
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def write_csv(tasks: List[Task], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image","label","rel_dir"])
        for t in tasks:
            w.writerow([t.image_uri, t.label, t.rel_dir])

def write_labels(tasks: List[Task], out_path: Path) -> None:
    labels = sorted({t.label for t in tasks if t.label})
    out_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Label Studio import files from a folder tree.")
    parser.add_argument("root", type=str, help="Root folder containing images and subfolders.")
    parser.add_argument("--exts", type=str, default=",".join(sorted(DEFAULT_EXTS)),
                        help="Comma-separated list of file extensions to include (e.g. .jpg,.png)")
    parser.add_argument("--label-depth", type=int, default=1,
                        help="How many directory levels (from root) to use as the label. Default=1.")
    parser.add_argument("--root-label", type=str, default="__root__",
                        help="Label to use for images directly under the root (no subfolder). Default='__root__'")
    parser.add_argument("--absolute-paths", action="store_true",
                        help="If set, write absolute file:// URIs instead of relative paths.")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Prefix for output files. Default: derived from root folder name.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report counts, but don't write files.")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Root folder not found or not a directory: {root}", file=sys.stderr)
        return 2

    exts = [e.strip() if e.strip().startswith(".") else "." + e.strip() for e in args.exts.split(",") if e.strip()]
    images = find_images(root, exts)
    if not images:
        print("[WARN] No images found with given extensions.", file=sys.stderr)

    tasks = build_tasks(root, images, args.label_depth, args.absolute_paths, args.root_label)

    # Report
    by_label: Dict[str, int] = {}
    for t in tasks:
        by_label[t.label] = by_label.get(t.label, 0) + 1

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Files matched: {len(images)} | Tasks built: {len(tasks)}")
    print(f"[INFO] Label depth: {args.label_depth} | Absolute paths: {args.absolute_paths}")
    print("[INFO] Label distribution (top 20):")
    for i, (lab, cnt) in enumerate(sorted(by_label.items(), key=lambda kv: (-kv[1], kv[0]))[:20], start=1):
        print(f"  {i:>2}. {lab} -> {cnt}")

    if args.dry_run:
        return 0

    prefix = args.output_prefix or f"{root.name}".replace(" ", "_")
    json_path = root.parent / f"{prefix}.json"
    csv_path = root.parent / f"{prefix}.csv"
    labels_path = root.parent / f"{prefix}.labels.txt"

    write_json(tasks, json_path)
    write_csv(tasks, csv_path)
    write_labels(tasks, labels_path)

    print(f"[OK] Wrote: {json_path}")
    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {labels_path}")
    print("\nNext steps in Label Studio:")
    print("  - Create a project with an Image classification/segmentation config.")
    print("  - Import -> 'Upload JSON' or 'Upload CSV' (the files above).")
    print("  - Optional: Map 'meta.folder_label' to a prechoice if using the JSON, or 'label' if using CSV.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
