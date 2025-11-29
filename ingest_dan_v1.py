#!/usr/bin/env python3
# ingest_dan_v1.py
#
# Build DAN chunks JSONL.zst compatible with DHM RAG v4 pipeline.

import os
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, Generator, List

import zstandard as zstd

# Adjust if needed
DAN_ROOT = Path("DAN_Publications")
OUT_DIR = Path("data/ingestion_jsonl")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "dan_chunks_v1.jsonl.zst"


def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def guess_year(name: str) -> str:
    m = re.search(r"(19|20)\d{2}", name)
    return m.group(0) if m else ""


def iter_reports(root: Path):
    """
    Yield (report_dir, name, pdf_path, text_path, tables_dir) for each DAN report.
    """
    if not root.exists():
        raise SystemExit(f"DAN root not found: {root}")

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        pdf_candidates = list(entry.glob("*.pdf"))
        text_path = entry / "text_clean.txt"
        tables_dir = entry / "extracted" / "tables"

        if not text_path.exists():
            # Skip folders without pre-extracted text
            continue

        pdf_path = pdf_candidates[0] if pdf_candidates else None
        yield {
            "dir": entry,
            "name": name,
            "pdf_path": pdf_path,
            "text_path": text_path,
            "tables_dir": tables_dir if tables_dir.exists() else None,
        }


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Assumes text_clean is already reasonably paragraph-separated.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap  # overlap backwards
        if start < 0:
            start = 0

    return chunks


def read_text_clean(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def csv_to_text(path: Path, max_rows: int = 30, max_cols: int = 10) -> str:
    """
    Convert a CSV table to a compact text representation.
    Limit rows/cols so chunks stay manageable.
    """
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for r_i, row in enumerate(reader):
            if r_i >= max_rows:
                lines.append("… [truncated rows]")
                break
            row = row[:max_cols]
            lines.append(" | ".join(cell.strip() for cell in row))
    return "\n".join(lines).strip()


def iter_dan_chunks() -> Generator[Dict[str, Any], None, None]:
    """
    Yield chunk dicts for all DAN reports.
    Each chunk has .text and .meta compatible with DHM chunks.
    """
    for report in iter_reports(DAN_ROOT):
        report_dir: Path = report["dir"]
        name: str = report["name"]
        pdf_path: Path | None = report["pdf_path"]
        text_path: Path = report["text_path"]
        tables_dir: Path | None = report["tables_dir"]

        report_slug = slugify(name)
        year = guess_year(name)
        report_id = f"DAN::{report_slug}"

        pdf_rel = str(pdf_path) if pdf_path else ""
        text = read_text_clean(text_path)
        body_chunks = chunk_text(text)

        # Body text chunks
        for i, chunk in enumerate(body_chunks):
            yield {
                "id": f"{report_id}::body::{i:04d}",
                "text": chunk,
                "meta": {
                    "corpus": "DAN",
                    "report_id": report_id,
                    "report_name": name,
                    "year": year,
                    "section": "body",
                    "chunk_index": i,
                    "pdf_path": pdf_rel,
                    "source_file": str(text_path),
                },
            }

        # Table chunks
        if tables_dir and tables_dir.exists():
            for table_file in sorted(tables_dir.glob("*.csv")):
                table_text = csv_to_text(table_file)
                if not table_text:
                    continue
                yield {
                    "id": f"{report_id}::table::{table_file.stem}",
                    "text": table_text,
                    "meta": {
                        "corpus": "DAN",
                        "report_id": report_id,
                        "report_name": name,
                        "year": year,
                        "section": "table",
                        "table_file": str(table_file),
                        "pdf_path": pdf_rel,
                    },
                }


def write_jsonl_zst(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=6)
    count = 0

    with out_path.open("wb") as f:
        with cctx.stream_writer(f) as zw:
            for obj in iter_dan_chunks():
                line = json.dumps(obj, ensure_ascii=False)
                zw.write(line.encode("utf-8"))
                zw.write(b"\n")
                count += 1

    print(f"[ingest_dan_v1] Wrote {count} DAN chunks to {out_path}")


def main():
    write_jsonl_zst(OUT_PATH)


if __name__ == "__main__":
    main()
