"""
DAN Publication Library Scraper
Form POST version (no JS required)
Author: Donarun Das
Version: v3.0 — Stable DHM-style architecture
"""

import os
import re
import json
import time
import hashlib
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

# ---------- CONFIG ----------
BASE_URL = "https://apps.dan.org/Publication-Library/"
ROOT_DIR = "DAN_Publications"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://templeadventures.com)"}
REQUEST_DELAY = 2.0
TIMEOUT = 45
RETRIES = 3

# ---------- HELPERS ----------
def make_dir(p): os.makedirs(p, exist_ok=True)

def md5sum(fpath):
    h = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def polite_post(session, payload):
    """Retry wrapper for polite POSTs."""
    for attempt in range(RETRIES):
        try:
            r = session.post(BASE_URL, headers=HEADERS, data=payload, timeout=TIMEOUT)
            if r.status_code == 200:
                time.sleep(REQUEST_DELAY)
                return r
        except Exception as e:
            print(f"⚠️  Attempt {attempt+1}/{RETRIES} failed: {e}")
            time.sleep(3)
    return None

def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip())

# ---------- SCRAPER ----------
def fetch_form_lists(session):
    """Extract all category + document pairs."""
    r = session.get(BASE_URL, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Categories
    cats = [opt.get("value").strip() for opt in soup.select("select#Categories option") if opt.get("value")]

    # All publications (static list)
    docs = [
        (opt.get("value").strip(), opt.text.strip())
        for opt in soup.select("select#DownloadableDocumentId option")
        if opt.get("value")
    ]

    print(f"📚 Categories: {cats}")
    print(f"📄 Total documents found: {len(docs)}")
    return cats, docs

def download_document(session, category, doc_id, title, manifest):
    folder = safe_name(title)
    folder_path = os.path.join(ROOT_DIR, folder)
    make_dir(folder_path)
    pdf_path = os.path.join(folder_path, f"{folder}.pdf")
    meta_path = os.path.join(folder_path, "metadata.json")

    if os.path.exists(pdf_path):
        return  # resume-safe

    payload = {
        "Categories": category,
        "DownloadableDocumentId": doc_id,
        "submit": "Request Document"
    }

    r = polite_post(session, payload)
    if not r or "application/pdf" not in r.headers.get("Content-Type", ""):
        print(f"❌ Failed: {title}")
        return

    with open(pdf_path, "wb") as f:
        f.write(r.content)

    meta = {
        "title": title,
        "category": category,
        "document_id": doc_id,
        "download_url": BASE_URL,
        "pdf_path": os.path.relpath(pdf_path, ROOT_DIR),
        "file_size_bytes": os.path.getsize(pdf_path),
        "checksum_md5": md5sum(pdf_path),
        "downloaded_on": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "OK"
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    manifest.append(meta)

# ---------- MAIN ----------
def main():
    make_dir(ROOT_DIR)
    manifest = []
    session = requests.Session()

    print("🔍 Fetching publication list...")
    cats, docs = fetch_form_lists(session)

    for category in cats:
        print(f"\n📂 Category: {category}")
        for doc_id, title in tqdm(docs, desc=f"{category}", unit="file"):
            try:
                download_document(session, category, doc_id, title, manifest)
            except Exception as e:
                print(f"⚠️  Error on {title}: {e}")

    # Save manifest
    with open(os.path.join(ROOT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print(f"\n✅ Done! {len(manifest)} files indexed.")
    print(f"🗂 Manifest saved at {os.path.join(ROOT_DIR, 'manifest.json')}")

if __name__ == "__main__":
    main()
