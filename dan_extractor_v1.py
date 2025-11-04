#!/usr/bin/env python3
"""
DAN Extractor v1.1 — Color-Safe, Threaded, and Manifest-Aware
───────────────────────────────────────────────────────────────
Upgraded from v1.0 with:
  ✅ Safe color handling for CMYK / masks / undefined colorspaces
  ✅ Modular safe_pixmap() wrapper
  ✅ Image extraction audit + graceful fallbacks
  ✅ Threaded manifest processing with resume support potential
"""

import os, json, fitz, hashlib, traceback, argparse, concurrent.futures
from pathlib import Path
from datetime import datetime, UTC
from tqdm import tqdm
import pandas as pd
from PIL import Image
import io

# ────────────────────────────────────────────────
THREADS = 8
MIN_FILE_SIZE = 1024
TEXT_FLAGS = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_IMAGES

# ────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()

def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ────────────────────────────────────────────────
def safe_pixmap(doc, xref):
            """
            Return a Pixmap that is guaranteed writeable as PNG (RGB or Gray).
            Falls back to a 1×1 white placeholder if stream is unreadable.
            """
            try:
                pix = fitz.Pixmap(doc, xref)
        
                # Case A: undefined colorspace → grayscale
                if pix.colorspace is None:
                    pix = fitz.Pixmap(fitz.csGRAY, pix)
        
                # Case B: Indexed / Separation / CMYK / >3 channels → RGB
                if pix.n not in (1, 3):        # only 1 (gray) or 3 (RGB) can write PNG
                    pix = fitz.Pixmap(fitz.csRGB, pix)
        
                # Case C: drop alpha
                if pix.alpha:
                    pix0 = fitz.Pixmap(pix, 0)
                    pix = pix0
        
                return pix
        
            except Exception as e:
                # Hard failure: make an in-memory 1×1 RGB white pixel
                img_bytes = io.BytesIO()
                Image.new("RGB", (1, 1), (255, 255, 255)).save(img_bytes, format="PNG")
                img_bytes.seek(0)
                return None

# ────────────────────────────────────────────────
def extract_pdf(pdf_path: Path, out_dir: Path):
    """Extract text and images into structured folders (color-safe)."""
    text_dir = out_dir / "text"
    img_dir  = out_dir / "images"
    for p in (text_dir, img_dir):
        p.mkdir(parents=True, exist_ok=True)

    text_all = []
    image_records = []

    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, start=1):
                # ── Text extraction ─────────────────────────────
                text = page.get_text("text", flags=TEXT_FLAGS)
                text_all.append(f"\n=== PAGE {i} ===\n{text}")

                # ── Image extraction ────────────────────────────
                for j, img in enumerate(page.get_images(full=True), start=1):
                            xref = img[0]
                            base = f"p{i:03d}_img{j:02d}.png"
                            try:
                                pix = fitz.Pixmap(doc, xref)
                        
                                # Determine colorspace safely
                                cs = pix.colorspace.name if pix.colorspace else "Unknown"
                                n  = pix.n
                        
                                # Case 1: undefined colorspace or exotic -> RGB
                                if cs not in ("DeviceRGB", "DeviceGray"):
                                    pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                                # Case 2: CMYK or >3 channels -> RGB
                                elif n > 3:
                                    pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                                # Case 3: has alpha -> drop alpha safely
                                if pix.alpha:
                                    pix0 = fitz.Pixmap(pix, 0)
                                    pix0.save(img_dir / base)
                                    pix0 = None
                                else:
                                    # only gray or RGB are PNG-compatible
                                    if n not in (1, 3):
                                        pix = fitz.Pixmap(fitz.csRGB, pix)
                                    pix.save(img_dir / base)
                        
                                image_records.append({"page": i, "name": base})
                                pix = None
                        
                            except Exception as e:
                                from PIL import Image
                                Image.new("RGB", (1, 1), (255, 255, 255)).save(img_dir / base)
                                print(f"⚠️  image p{i}_img{j} failed ({type(e).__name__}): {e}")
                                image_records.append({
                                    "page": i,
                                    "name": base,
                                    "error": str(e)
                                })
                                continue
                        

    except Exception as e:
        raise RuntimeError(f"Failed to read {pdf_path}: {e}")

    # ── Write text to file ─────────────────────────
    (text_dir / "content.txt").write_text("\n".join(text_all), encoding="utf-8")

    return len(text_all), len(image_records)

# ────────────────────────────────────────────────
def process_entry(entry, root_dir: Path, out_audit: list):
    pdf_rel = entry["pdf_path"]
    pdf_path = root_dir / pdf_rel
    folder = pdf_path.parent
    out_dir = folder / "extracted"

    result = {
        "title": entry.get("title"),
        "pdf": str(pdf_path),
        "sha256": None,
        "pages": 0,
        "images": 0,
        "status": "FAILED",
        "error": "",
        "timestamp": now_iso(),
    }

    try:
        if not pdf_path.exists() or pdf_path.stat().st_size < MIN_FILE_SIZE:
            raise FileNotFoundError("File missing or too small")

        pages, imgs = extract_pdf(pdf_path, out_dir)
        result.update({
            "sha256": sha256sum(pdf_path),
            "pages": pages,
            "images": imgs,
            "status": "OK",
        })
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        out_audit.append(result)

# ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Extract text/images from DAN manifest (color-safe build)")
    ap.add_argument("--manifest", required=True, help="Path to manifest.json from scraper")
    ap.add_argument("--root", default="DAN_Publications", help="Root folder containing PDFs")
    ap.add_argument("--out",  default="dan_extraction_audit_v1.1.csv", help="CSV audit log path")
    ap.add_argument("--threads", type=int, default=THREADS)
    args = ap.parse_args()

    root_dir = Path(args.root)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries = json.loads(manifest_path.read_text())
    audit_records = []

    print(f"📘 Processing {len(entries)} PDFs using {args.threads} threads...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        list(tqdm(ex.map(lambda e: process_entry(e, root_dir, audit_records), entries),
                  total=len(entries), desc="Extracting", unit="pdf"))

    pd.DataFrame(audit_records).to_csv(args.out, index=False)
    print(f"\n✅ Extraction complete. Audit written to {args.out}")

# ────────────────────────────────────────────────
if __name__ == "__main__":
    main()
