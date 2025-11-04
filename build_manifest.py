#!/usr/bin/env python3
"""
Phase 3 – DAN Content Manifest Builder
--------------------------------------
Builds a unified manifest from raw Markdown, extracted tables, and images.
"""

import json, hashlib, platform
from datetime import datetime
from pathlib import Path
import csv

ROOT = Path("/Users/donarundas/Projects/DAN/DAN_Publications")
OUT_JSON = ROOT.parent / "dan_content_manifest.json"
OUT_CSV  = ROOT.parent / "dan_content_manifest.csv"

def sha256(path: Path):
    if not path.exists(): return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def count_words(path: Path) -> int:
    if not path.exists(): return 0
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return len(text.split())
    except Exception:
        return 0

def sizeof_fmt(num, suffix="B"):
    for unit in ["","K","M","G"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}T{suffix}"

def build_manifest():
    records, errors = [], []

    for pub in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        record = {"publication": pub.name}
        pdf = next(pub.glob("*.pdf"), None)
        md_file = pub / "raw.md"
        tables_dir = pub / "extracted" / "tables"
        imgs_dir   = pub / "extracted" / "images"

        record.update({
            "markdown_file": str(md_file) if md_file.exists() else "",
            "tables": [str(f) for f in tables_dir.glob("*.csv")] if tables_dir.exists() else [],
            "images": [str(f) for f in imgs_dir.glob("*.*")] if imgs_dir.exists() else [],
            "word_count": count_words(md_file),
            "tables_count": len(list(tables_dir.glob("*.csv"))) if tables_dir.exists() else 0,
            "images_count": len(list(imgs_dir.glob("*.*"))) if imgs_dir.exists() else 0,
            "sha256": sha256(pdf) if pdf else None,
            "pdf": str(pdf) if pdf else "",
            "pdf_size": sizeof_fmt(pdf.stat().st_size) if pdf else "0B",
            "last_modified": datetime.utcfromtimestamp(
                max([f.stat().st_mtime for f in pub.rglob("*") if f.is_file()])
            ).isoformat() + "Z",
        })
        records.append(record)

    manifest = {
        "created": datetime.utcnow().isoformat() + "Z",
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform()
        },
        "records": records,
        "total_publications": len(records)
    }

    OUT_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"✅ JSON manifest written: {OUT_JSON}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "publication","word_count","tables_count","images_count","pdf_size","markdown_file"
        ])
        for r in records:
            writer.writerow([
                r["publication"], r["word_count"], r["tables_count"],
                r["images_count"], r["pdf_size"], r["markdown_file"]
            ])
    print(f"📘 CSV summary written: {OUT_CSV}")

if __name__ == "__main__":
    build_manifest()
