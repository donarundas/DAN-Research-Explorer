#!/usr/bin/env python3
"""
DAN Publication Visual Extraction
---------------------------------
Phase 2: detect and extract tables (Camelot/pdfplumber) and figures (PyMuPDF)
from each publication PDF already processed by MarkItDown.
"""

import fitz, camelot, pdfplumber, json, csv
from pathlib import Path
from datetime import datetime
from PIL import Image
import io, traceback

ROOT_DIR = Path("/Users/donarundas/Projects/DAN/DAN_Publications")

def safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def extract_tables(pdf_path: Path, out_dir: Path, metadata: dict):
    """Use Camelot and pdfplumber to extract tables to CSV."""
    safe_mkdir(out_dir)
    tables_meta = []
    try:
        # Camelot lattice + stream
        all_tbls = []
        for flavor in ["lattice", "stream"]:
            try:
                tbls = camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
                all_tbls += list(tbls)
            except Exception as e:
                metadata.setdefault("warnings", []).append(f"{flavor} fail: {e}")
        # pdfplumber fallback
        with pdfplumber.open(pdf_path) as pdf:
            for pno, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables():
                    all_tbls.append((pno, table))

        count = 0
        for idx, tbl in enumerate(all_tbls, start=1):
            # pdfplumber tables come as tuple(page_no, list)
            if isinstance(tbl, tuple):
                page_no, table_data = tbl
                engine = "pdfplumber"
            else:
                table_data = tbl.df.values.tolist()
                page_no = tbl.page
                engine = "camelot"

            csv_name = f"table_{idx:02d}_{engine}.csv"
            csv_path = out_dir / csv_name
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerows(table_data)
            tables_meta.append({
                "id": f"TABLE_{idx:02d}",
                "page": page_no,
                "engine": engine,
                "file": csv_name,
                "rows": len(table_data)
            })
            count += 1

        print(f"🧮 {count} tables saved for {pdf_path.name}")
    except Exception as e:
        metadata.setdefault("errors", []).append(f"table extraction: {e}")
        print(f"⚠️ Table extraction error: {e}")
        traceback.print_exc()

    metadata["tables_found"] = len(tables_meta)
    metadata["table_details"] = tables_meta


def extract_images(pdf_path: Path, out_dir: Path, metadata: dict):
    """Extract embedded images from PDF using PyMuPDF."""
    safe_mkdir(out_dir)
    count = 0
    image_meta = []
    try:
        doc = fitz.open(pdf_path)
        for page_no, page in enumerate(doc, start=1):
            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_ext = base_image["ext"]
                image = Image.open(io.BytesIO(image_bytes))
                out_name = f"figure_{count+1:02d}_page{page_no}.{img_ext}"
                out_path = out_dir / out_name
                image.save(out_path)
                image_meta.append({
                    "file": out_name,
                    "page": page_no,
                    "width": image.width,
                    "height": image.height,
                    "ext": img_ext
                })
                count += 1
        print(f"🖼️  {count} images extracted from {pdf_path.name}")
    except Exception as e:
        metadata.setdefault("errors", []).append(f"image extraction: {e}")
        print(f"⚠️ Image extraction error: {e}")
        traceback.print_exc()

    metadata["images_found"] = len(image_meta)
    metadata["image_details"] = image_meta


def process_publication(pub_dir: Path):
    pdfs = list(pub_dir.glob("*.pdf"))
    if not pdfs: return
    pdf_path = pdfs[0]
    meta_path = pub_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    extracted_root = pub_dir / "extracted"
    tbl_dir = extracted_root / "tables"
    img_dir = extracted_root / "images"

    metadata["phase2_run"] = datetime.utcnow().isoformat() + "Z"

    extract_tables(pdf_path, tbl_dir, metadata)
    extract_images(pdf_path, img_dir, metadata)

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    pubs = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()])
    print(f"🚀 Extracting tables & images from {len(pubs)} PDFs")
    for pub in pubs:
        process_publication(pub)
    print("🏁 Phase 2 complete.")


if __name__ == "__main__":
    main()
