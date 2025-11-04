#!/usr/bin/env python3
"""
dan_extractor_v2.0.py
DAN Structured Extractor v2.0 — Tables, Figures, Layout

What it does per PDF:
  - (optional) wipe and recreate ./extracted/
  - text extraction (continuous text for LLM context)
  - layout extraction (bbox-level JSON per page)
  - table extraction to CSV (pdfplumber + camelot fallback)
  - figure extraction to PNG with bbox + caption guess
  - summary.json with counts
Global:
  - threaded over all PDFs from manifest.json
  - writes dan_extraction_audit_v2.csv at the end

Author: Donarun Das
"""

import os, io, json, time, shutil, traceback, argparse, concurrent.futures
from pathlib import Path
from datetime import datetime, UTC
import hashlib
import pandas as pd
from tqdm import tqdm

import fitz            # PyMuPDF
from PIL import Image
import pdfplumber
import camelot

# --------------------------- config ---------------------------------
THREADS_DEFAULT = 4
MIN_FILE_SIZE = 1024
AUDIT_CSV_DEFAULT = "dan_extraction_audit_v2.csv"
TEXT_FLAGS = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_IMAGES

# --------------------------- helpers --------------------------------
def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------- text + layout --------------------------
def extract_text_and_layout(pdf_path: Path, text_dir: Path, layout_dir: Path):
    """
    - Writes text_dir/content.txt (continuous text with page headers)
    - Writes one JSON per page in layout_dir/page_###.json
      Layout JSON is PyMuPDF page.get_text("json") for later spatial reasoning.
    Returns: pages_count, text_block_count
    """
    ensure_dir(text_dir)
    ensure_dir(layout_dir)

    all_text_chunks = []
    pages_count = 0
    total_blocks = 0

    with fitz.open(pdf_path) as doc:
        pages_count = len(doc)
        for page_index, page in enumerate(doc, start=1):
            # raw text for humans / LLM
            page_text = page.get_text("text", flags=TEXT_FLAGS)
            all_text_chunks.append(f"\n=== PAGE {page_index} ===\n{page_text}")

            # structured layout for post-processing (tables / captions)
            layout_json = page.get_text("json")
            layout_obj = json.loads(layout_json)
            total_blocks += len(layout_obj.get("blocks", []))

            with open(layout_dir / f"page_{page_index:03d}.json", "w", encoding="utf-8") as f:
                json.dump(layout_obj, f, indent=2, ensure_ascii=False)

    # write combined text file
    (text_dir / "content.txt").write_text("\n".join(all_text_chunks), encoding="utf-8")

    return pages_count, total_blocks

# --------------------------- tables ---------------------------------
def extract_tables(pdf_path: Path, tables_dir: Path):
    """
    Attempt table extraction 2 ways:
      1. pdfplumber (text-based tables / stream)
      2. camelot (line-based 'lattice'/'stream')
    Output:
      tables_dir/table_###.csv for each detected table
    Returns: table_count
    """
    ensure_dir(tables_dir)
    table_idx = 0

    # Pass 1: pdfplumber per page, tables() method
    try:
        with pdfplumber.open(pdf_path) as plumb_doc:
            for pnum, page in enumerate(plumb_doc.pages, start=1):
                try:
                    extracted_tables = page.extract_tables() or []
                except Exception:
                    extracted_tables = []
                for tbl in extracted_tables:
                    # tbl is list[list[str]]
                    table_idx += 1
                    out_csv = tables_dir / f"table_{table_idx:03d}_p{pnum:03d}_plumber.csv"
                    pd.DataFrame(tbl).to_csv(out_csv, index=False, header=False)
    except Exception as e:
        print(f"⚠️ pdfplumber failed on {pdf_path.name}: {e}")

    # Pass 2: camelot over full PDF (lattice then stream)
    def camelot_pass(flavor):
        nonlocal table_idx
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
            for i, t in enumerate(tables, start=1):
                table_idx += 1
                out_csv = tables_dir / f"table_{table_idx:03d}_camelot_{flavor}.csv"
                t.to_csv(str(out_csv), index=False)
        except Exception as e:
            # quiet-ish; camelot fails a lot on graphics-heavy pages
            print(f"ℹ️ camelot {flavor} fail/partial on {pdf_path.name}: {e}")

    camelot_pass("lattice")
    camelot_pass("stream")

    return table_idx

# --------------------------- figures --------------------------------
def extract_figures(pdf_path: Path, figures_dir: Path, figures_meta_path: Path):
    """
    Extracts raster images as PNG.
    Also records figure metadata (page, bbox, dims, colorspace guess, caption guess)
    Returns: figure_count
    """

    ensure_dir(figures_dir)

    figures_meta = []
    figure_count = 0

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            # we will use layout blocks to guess nearby caption later
            # grab blocks once so we don't call twice
            layout_obj = json.loads(page.get_text("json"))
            blocks = layout_obj.get("blocks", [])

            # page.get_images(full=True) returns tuples:
            # (xref, smask, width, height, bpc, colorspace, alt, name, bbox)
            for img_idx, imginfo in enumerate(page.get_images(full=True), start=1):
                xref, smask, w, h, bpc, cs, alt, name, bbox = imginfo
                base_name = f"p{page_index:03d}_img{img_idx:03d}.png"
                out_path = figures_dir / base_name

                # try to build Pixmap
                saved_ok = False
                colorspace_name = None

                try:
                    try:
                        pix = fitz.Pixmap(doc, xref)
                    except ValueError:
                        # "source colorspace must not be None" etc.
                        pix = None

                    if pix is not None:
                        # record colorspace info if available
                        if pix.colorspace:
                            # PyMuPDF >=1.23
                            try:
                                colorspace_name = pix.colorspace.name
                            except Exception:
                                colorspace_name = str(pix.colorspace)
                        else:
                            colorspace_name = "Unknown"

                        # normalize to RGB/GRAY for PNG
                        # n = samples per pixel (1=gray,3=rgb,4+=cmyk etc)
                        n = pix.n
                        if (pix.colorspace is None) or (n > 3):
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        # drop alpha if present
                        if pix.alpha:
                            pix0 = fitz.Pixmap(pix, 0)
                            pix0.save(out_path)
                            pix0 = None
                        else:
                            if n not in (1, 3):
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            pix.save(out_path)

                        saved_ok = True
                        pix = None

                except Exception as e:
                    # fallback: make 1x1 white so downstream pipelines never break
                    img_stub = Image.new("RGB", (1, 1), (255, 255, 255))
                    img_stub.save(out_path)

                # caption guess: nearest text block under or overlapping bbox
                caption_text = guess_caption_from_blocks(blocks, bbox)

                figure_count += 1
                figures_meta.append({
                    "page": page_index,
                    "img_id": img_idx,
                    "file": base_name,
                    "bbox": bbox,
                    "width_px": w,
                    "height_px": h,
                    "colorspace": colorspace_name,
                    "saved_ok": saved_ok,
                    "caption_guess": caption_text
                })

    with open(figures_meta_path, "w", encoding="utf-8") as f:
        json.dump(figures_meta, f, indent=2, ensure_ascii=False)

    return figure_count

def guess_caption_from_blocks(blocks, bbox):
    """
    Very lightweight heuristic:
    - find text blocks whose bbox vertical start is just below or overlaps img bbox
    - join lines into a caption string
    Returns short caption or "".
    """
    x0, y0, x1, y1 = bbox  # bbox from fitz is (x0, y0, x1, y1)

    # collect candidate text blocks
    candidates = []
    for b in blocks:
        if "lines" not in b:
            continue
        bx0, by0, bx1, by1 = b.get("bbox", [None]*4)
        if bx0 is None:
            continue

        # simple proximity rule:
        #   block top within ~30px below image bottom
        #   OR overlaps vertically with figure bbox
        vertical_gap = by0 - y1
        overlaps = not (by0 > y1+30 or by1 < y0-30)

        if (0 <= vertical_gap <= 30) or overlaps:
            # flatten block text
            txt_lines = []
            for line in b["lines"]:
                spans = line.get("spans", [])
                line_text = " ".join([s.get("text","") for s in spans]).strip()
                if line_text:
                    txt_lines.append(line_text)
            block_text = " ".join(txt_lines).strip()
            if block_text:
                candidates.append((vertical_gap, block_text))

    # take the closest / first sensible candidate
    if not candidates:
        return ""
    # sort by absolute vertical_gap (prefer just-below vs overlapping random headers)
    candidates.sort(key=lambda c: abs(c[0] if c[0] is not None else 9999))
    return candidates[0][1][:500]  # truncate long captions

# --------------------------- per-PDF pipeline -----------------------
def process_single_pdf(entry, root_dir: Path, overwrite: bool):
    """
    For one manifest entry:
      - build extracted/ with subfolders
      - run text/layout/tables/figures
      - write summary.json
    Returns dict for the audit row.
    """

    start_time = time.time()

    pdf_rel = entry["pdf_path"]
    pdf_path = root_dir / pdf_rel
    pdf_dir = pdf_path.parent
    extracted_dir = pdf_dir / "extracted"
    text_dir = extracted_dir / "text"
    layout_dir = extracted_dir / "layout"
    tables_dir = extracted_dir / "tables"
    figures_dir = extracted_dir / "figures"
    figures_meta_path = extracted_dir / "figures_metadata.json"
    summary_path = extracted_dir / "summary.json"

    result = {
        "title": entry.get("title"),
        "pdf_path": str(pdf_path),
        "sha256": None,
        "pages": 0,
        "text_blocks": 0,
        "tables": 0,
        "figures": 0,
        "status": "FAILED",
        "error": "",
        "timestamp": now_iso(),
        "elapsed_sec": 0.0,
    }

    try:
        if not pdf_path.exists() or pdf_path.stat().st_size < MIN_FILE_SIZE:
            raise FileNotFoundError("File missing or too small")

        # overwrite mode: nuke extracted/ first
        if overwrite and extracted_dir.exists():
            shutil.rmtree(extracted_dir)

        ensure_dir(extracted_dir)

        # 1. text + layout
        pages_count, text_blocks = extract_text_and_layout(pdf_path, text_dir, layout_dir)

        # 2. tables
        try:
            tables_count = extract_tables(pdf_path, tables_dir)
        except Exception as e:
            print(f"⚠️ table extraction failed hard on {pdf_path.name}: {e}")
            tables_count = 0
        

        # 3. figures
        figures_count = extract_figures(pdf_path, figures_dir, figures_meta_path)

        # 4. summary.json
        summary_data = {
            "title": entry.get("title"),
            "pdf_file": pdf_path.name,
            "sha256": sha256sum(pdf_path),
            "pages": pages_count,
            "text_blocks": text_blocks,
            "tables": tables_count,
            "figures": figures_count,
            "extracted_on": now_iso()
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # finalize result row
        result.update({
            "sha256": summary_data["sha256"],
            "pages": pages_count,
            "text_blocks": text_blocks,
            "tables": tables_count,
            "figures": figures_count,
            "status": "OK",
        })

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    finally:
        result["elapsed_sec"] = round(time.time() - start_time, 3)

    return result

# --------------------------- main() ---------------------------------
def main():
    ap = argparse.ArgumentParser(description="DAN Structured Extractor v2.0 (tables + figures + layout)")
    ap.add_argument("--manifest", required=True, help="Path to manifest.json created by scraper")
    ap.add_argument("--root", default="DAN_Publications", help="Root folder containing PDFs")
    ap.add_argument("--out", default=AUDIT_CSV_DEFAULT, help="Global audit CSV path")
    ap.add_argument("--threads", type=int, default=THREADS_DEFAULT)
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, delete and rebuild each PDF's extracted/ folder")
    args = ap.parse_args()

    root_dir = Path(args.root)
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries = json.loads(manifest_path.read_text())
    audit_rows = []

    # threaded execution across PDFs
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = []
        for entry in entries:
            futures.append(pool.submit(process_single_pdf, entry, root_dir, args.overwrite))

        for f in tqdm(concurrent.futures.as_completed(futures),
                      total=len(futures),
                      desc="Extracting PDFs",
                      unit="pdf"):
            audit_rows.append(f.result())

    # write audit csv
    pd.DataFrame(audit_rows).to_csv(args.out, index=False)

    print(f"\n✅ Complete. Audit written to {args.out}")
    print("Sample row:")
    if audit_rows:
        print(audit_rows[0])

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
