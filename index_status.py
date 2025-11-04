#!/usr/bin/env python3
"""
index_status.py
----------------
Quick audit tool for DAN embedding corpus.
Reports progress, cache status, and metadata integrity.
"""

import os, json, sqlite3, numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import faiss

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
ROOT = Path("/Users/donarundas/Projects/DAN")
INDEX_DIR = ROOT / "index"
CACHE_DB = INDEX_DIR / "embed_cache.sqlite3"
META_PATH = INDEX_DIR / "vector_metadata.json"
FAISS_PATH = INDEX_DIR / "vector_index.faiss"
MANIFEST_PATH = ROOT / "dan_content_manifest.json"

load_dotenv(ROOT / ".env")

def count_cache_entries(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT COUNT(*) FROM cache")
    rows = cur.fetchone()[0]
    conn.close()
    return rows

def inspect_faiss(path):
    if not path.exists():
        return 0, None
    idx = faiss.read_index(str(path))
    return idx.ntotal, idx.d

def inspect_metadata(path):
    if not path.exists():
        return 0, {}
    data = json.loads(path.read_text())
    chunks = data.get("chunks", [])
    built = data.get("built", "")
    models = list(set(c.get("model_used", "unknown") for c in chunks))
    avg_len = int(np.mean([len(c.get("source","")) for c in chunks])) if chunks else 0
    return len(chunks), {"built": built, "models": models, "avg_len": avg_len}

def inspect_manifest(path):
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    total_records = len(data.get("records", []))
    return total_records

def main():
    print("📊 DAN Corpus Status Report")
    print("──────────────────────────────────────────────")

    total_manifest = inspect_manifest(MANIFEST_PATH)
    meta_chunks, meta_info = inspect_metadata(META_PATH)
    faiss_count, faiss_dim = inspect_faiss(FAISS_PATH)
    cache_count = count_cache_entries(CACHE_DB)

    print(f"📁 Manifest records:        {total_manifest}")
    print(f"📜 Metadata chunks:         {meta_chunks}")
    print(f"💾 Cached embeddings:       {cache_count}")
    print(f"🧠 FAISS index entries:     {faiss_count}")
    print(f"📐 Embedding dimension:     {faiss_dim if faiss_dim else '—'}")
    print(f"🕒 Last build timestamp:    {meta_info.get('built','—')}")
    print(f"🤖 Models used:             {', '.join(meta_info.get('models', []))}")
    print(f"✂️  Avg. source length:     {meta_info.get('avg_len', 0)} characters")

    completion = 0
    if meta_chunks:
        completion = (cache_count / meta_chunks) * 100
    print(f"✅ Completion ratio:        {completion:.1f}%")

    if completion < 100:
        print("\n⚠️  Some chunks not cached. Re-run index_corpus_v4.py to finish embedding.")
    else:
        print("\n🎉 All Markdown chunks embedded successfully!")

    print("──────────────────────────────────────────────")

if __name__ == "__main__":
    main()
