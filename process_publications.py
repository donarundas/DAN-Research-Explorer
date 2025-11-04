#!/usr/bin/env python3
"""
DAN Publication Processor (MarkItDown Edition)
Author: Donarun Das + GPT-5 (Persistent Coding Mode)
---------------------------------------------------
Processes each PDF in Projects/DAN/DAN_Publications/* into:
  - raw.md (full Markdown via MarkItDown)
  - text_clean.txt (cleaned narrative, placeholders for tables)
  - tables/ (CSV files per table)
  - metadata.json (audit + traceability)
"""

import os, re, csv, json, sys, platform, hashlib, traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

# Dependency check
REQUIRED_LIBS = ["markitdown", "pandas"]
missing = []
for lib in REQUIRED_LIBS:
    try:
        __import__(lib)
    except ImportError:
        missing.append(lib)
if missing:
    sys.exit(f"❌ Missing dependencies: {', '.join(missing)}. Please install them first.")

from markitdown import MarkItDown
import pandas as pd


# === CONFIG ===
ROOT_DIR = Path("/Users/donarundas/Projects/DAN/DAN_Publications")
MAX_TABLES_PER_FILE = 100
SKIP_KEYWORDS = [
    "table of contents", "contents", "acknowledgements",
    "acknowledgments", "copyright", "foreword", "about dan",
]
# ==============


def hash_file(path: Path) -> str:
    """Compute SHA-256 hash for reproducibility."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(text: str, max_len=50):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return text[:max_len] or "table"


def extract_tables(md_text: str) -> Tuple[str, List[Dict]]:
    """
    Detect markdown pipe tables and extract them.
    Returns cleaned_text, tables_info
    """
    lines = md_text.splitlines()
    cleaned_lines, tables = [], []
    i, tcount = 0, 1

    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*\|.*\|\s*$", line):
            # collect full table block
            block = [line]
            i += 1
            while i < len(lines) and re.match(r"^\s*\|.*\|\s*$", lines[i]):
                block.append(lines[i])
                i += 1

            # look back for caption
            caption = ""
            for back in range(1, 3):
                if len(cleaned_lines) >= back:
                    cand = cleaned_lines[-back].strip()
                    if re.match(r"^table\s*\d+", cand, re.I):
                        caption = cand
                        cleaned_lines = cleaned_lines[:-back]
                        break

            table_id = f"TABLE_{tcount:02d}"
            tcount += 1
            tables.append({
                "id": table_id,
                "caption": caption,
                "block": block,
                "slug": slugify(caption),
            })
            cleaned_lines.append(f"[{table_id}]")
        else:
            cleaned_lines.append(line)
            i += 1

    return "\n".join(cleaned_lines), tables


def md_table_to_csv_rows(block: List[str]) -> List[List[str]]:
    rows = []
    for idx, l in enumerate(block):
        l = l.strip().strip("|")
        cells = [c.strip() for c in l.split("|")]
        if idx == 1 and all(re.match(r"^:?-{2,}:?$", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def remove_boilerplate(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    skip = False
    for line in lines:
        low = line.lower().strip()
        if any(k in low for k in SKIP_KEYWORDS):
            skip = True
            continue
        if skip and not low:
            skip = False
            continue
        if not skip:
            cleaned.append(line)
    return "\n".join(cleaned)


def process_publication(pub_dir: Path, md: MarkItDown):
    pdfs = list(pub_dir.glob("*.pdf"))
    if not pdfs:
        print(f"⚠️  No PDF in {pub_dir.name}, skipping.")
        return None
    pdf_path = pdfs[0]

    print(f"📄 Processing {pdf_path.name} …")

    raw_md_path = pub_dir / "raw.md"
    text_path = pub_dir / "text_clean.txt"
    tables_dir = pub_dir / "tables"
    meta_path = pub_dir / "metadata.json"
    tables_dir.mkdir(exist_ok=True)

    metadata = {
        "source_pdf": pdf_path.name,
        "sha256": hash_file(pdf_path),
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "markitdown_version": getattr(md, "__version__", "unknown"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "tables": [],
        "errors": [],
    }

    try:
        result = md.convert(str(pdf_path))
        md_text = result.text_content
        raw_md_path.write_text(md_text, encoding="utf-8")
        metadata["markitdown_metadata"] = getattr(result, "metadata", {})

        cleaned_text, tables = extract_tables(md_text)
        cleaned_text = remove_boilerplate(cleaned_text)
        text_path.write_text(cleaned_text, encoding="utf-8")

        for t in tables[:MAX_TABLES_PER_FILE]:
            rows = md_table_to_csv_rows(t["block"])
            csv_name = f"{t['id']}_{t['slug']}.csv"
            out = tables_dir / csv_name
            with open(out, "w", newline="", encoding="utf-8") as f:
                csv.writer(f, quoting=csv.QUOTE_ALL).writerows(rows)
            metadata["tables"].append({
                "id": t["id"],
                "caption": t["caption"],
                "file": out.name,
                "rows": len(rows)
            })

        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"✅ {pub_dir.name}: {len(tables)} tables, text_clean.txt saved.")

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        metadata["errors"].append(err)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"❌ {pub_dir.name}: {err}")
        traceback.print_exc()


def main():
    print(f"🚀 Starting processing in {ROOT_DIR}")
    md = MarkItDown(enable_plugins=False)
    pubs = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()])
    for pub in pubs:
        process_publication(pub, md)
    print("🏁 All publications processed.")


if __name__ == "__main__":
    main()
