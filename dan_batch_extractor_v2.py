#!/usr/bin/env python3
"""
dan_batch_extractor_v3.1.py
--------------------------------------------------
Silent, Metadata-aware OCR + Table Extractor for DAN Diving Reports
• Detects scanned vs text PDFs
• Extracts tables near 'Table n.n' references
• Cleans OCR junk, headers, footers
• Adds metadata header to every CSV
• Compatible with Pandas ≥ 2.2
Author: Donarun Das
Date: 2025-11-04
"""

import os, re, warnings, logging, subprocess, contextlib
from pathlib import Path
import fitz  # PyMuPDF
import camelot
import pdfplumber
import pandas as pd

# ---------- Global Silence ----------
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("camelot").setLevel(logging.ERROR)
pd.options.mode.copy_on_write = True

# ---------- Config ----------
ROOT_DIR = Path("DAN_Publications")
LANGUAGE = "eng"


# ---------- Detect Scanned PDFs ----------
def is_scanned_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        for page in doc:
            if page.get_text("text").strip():
                return False
    return True


# ---------- OCR Layer ----------
def run_ocr(input_file, output_file):
    print(f"🔍 OCR → {input_file.name}")
    cmd = [
        "ocrmypdf", "--deskew", "--clean", "--rotate-pages",
        "--language", LANGUAGE, "--output-type", "pdfa",
        str(input_file), str(output_file)
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ OCR done → {output_file.name}")


# ---------- Identify Table Pages ----------
def find_table_pages(pdf_path):
        """Return {page_index: caption_text} only for pages with actual table-like structure."""
        table_pages = {}
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                # --- find candidate captions ---
                matches = re.findall(r"(Table\s+\d[\.\d\-]*[^\n]*)", text, re.IGNORECASE)
                for m in matches:
                    caption = m.strip()
                    # skip cross-references
                    if re.search(r"\b(see|refer|appendix|of contents|below)\b", caption, re.I):
                        continue
    
                    # --- quick structural test: any table-like region? ---
                    preview = page.extract_table()
                    if preview and len(preview) > 2 and len(preview[0]) > 1:
                        table_pages[i] = caption
                        table_pages[i + 1] = caption  # table may span to next page
                        break  # one valid table per page is enough
                    else:
                        # fallback: check numeric density to guess tabular layout
                        words = re.findall(r"\d+(\.\d+)?", text)
                        if len(words) > 15:
                            table_pages[i] = caption
                            table_pages[i + 1] = caption
                            break
        return {k: v for k, v in table_pages.items() if k >= 0}
        


# ---------- Clean Tables ----------
def clean_table(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(how="all").fillna("")
    df = df.apply(lambda c: c.map(lambda x: str(x).strip()))
    df = df.apply(lambda c: c.map(lambda x: re.sub(r"\s+", " ", str(x)).strip()))

    drop_patterns = [
        r"Page\s*\d+", r"Diving and Hyperbaric", r"Annual\s+Diving\s+Report",
        r"Divers\s+Alert\s+Network", r"SPUMS", r"UHMS"
    ]
    mask = df.apply(lambda col: col.astype(str)
                    .apply(lambda x: any(re.search(p, x, re.I) for p in drop_patterns)))
    df = df[~mask.any(axis=1)]
    df = df[df.apply(lambda r: len("".join(map(str, r))) > 5, axis=1)]
    df = df[df.apply(
        lambda r: sum(c.isalnum() for c in "".join(map(str, r))) /
        max(len("".join(map(str, r))), 1) > 0.3, axis=1)]
    return df


# ---------- PDFPlumber ----------
def extract_with_pdfplumber(pdf_path, output_dir, table_pages):
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if (i - 1) not in table_pages:
                continue
            caption = table_pages.get(i - 1, "Unknown Table")
            tables = page.extract_tables()
            for j, table in enumerate(tables):
                df = pd.DataFrame(table[1:], columns=table[0])
                df = clean_table(df)
                if df.empty:
                    continue
                meta = [
                    f"# Source: {pdf_path.name}",
                    f"# Page: {i}",
                    f"# Extraction: pdfplumber",
                    f"# Caption: {caption}", ""
                ]
                out_path = output_dir / f"table_{i:03d}_{j}_pdfplumber.csv"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(meta))
                df.to_csv(out_path, mode="a", index=False)
                print(f"📄 {out_path.name}")


# ---------- Camelot ----------
def extract_with_camelot(pdf_path, output_dir, table_pages):
    if not table_pages:
        return
    page_list = ",".join(str(p + 1) for p in sorted(table_pages.keys()))
    try:
        with contextlib.redirect_stderr(open(os.devnull, "w")):
            tables = camelot.read_pdf(str(pdf_path), pages=page_list, flavor="stream")
        for i, t in enumerate(tables):
            df = t.df
            df.columns = [c.strip() for c in df.iloc[0]]
            df = df.drop(0).reset_index(drop=True)
            df = clean_table(df)
            if df.empty:
                continue
            caption = table_pages.get(int(t.page) - 1, "Unknown Table")
            meta = [
                f"# Source: {pdf_path.name}",
                f"# Page: {t.page}",
                f"# Extraction: Camelot (stream)",
                f"# Caption: {caption}", ""
            ]
            out_path = output_dir / f"table_{i+1:03d}_camelot.csv"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(meta))
            df.to_csv(out_path, mode="a", index=False)
            print(f"📊 {out_path.name}")
    except Exception as e:
        print(f"⚠️ Camelot failed on {pdf_path.name}: {e}")


# ---------- Per-Report Handler ----------
def process_pdf(pdf_path, output_dir):
    print(f"\n📘 Processing: {pdf_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.glob("table_*.csv"):
        f.unlink()

    scanned = is_scanned_pdf(pdf_path)
    print(f"   ├── Scanned: {scanned}")

    target_pdf = pdf_path
    if scanned:
        ocr_path = pdf_path.parent / f"OCR_{pdf_path.name}"
        run_ocr(pdf_path, ocr_path)
        target_pdf = ocr_path

    table_pages = find_table_pages(target_pdf)
    if not table_pages:
        print("   ⚠️ No 'Table' references found, skipping.")
        return

    extract_with_pdfplumber(target_pdf, output_dir, table_pages)
    extract_with_camelot(target_pdf, output_dir, table_pages)

    print(f"✅ Completed → {output_dir}")


# ---------- Batch Runner ----------
def main():
    reports = sorted([d for d in ROOT_DIR.iterdir() if d.is_dir()])
    print(f"📂 Found {len(reports)} report folders in {ROOT_DIR}")
    for report_dir in reports:
        pdfs = list(report_dir.glob("*.pdf"))
        if not pdfs:
            print(f"⚠️ No PDF found in {report_dir.name}")
            continue
        pdf_path = pdfs[0]
        output_dir = report_dir / "extracted" / "tables"
        process_pdf(pdf_path, output_dir)
    print("\n🎯 All DAN reports processed successfully!")


if __name__ == "__main__":
    main()
