#!/usr/bin/env python3
"""
index_tables_v1.py
------------------
Converts all extracted CSV tables into natural-language text,
embeds them, and stores vectors in FAISS with metadata.
"""

import os, glob, csv, json, sqlite3, asyncio, aiohttp
import numpy as np, faiss
from pathlib import Path
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
ROOT = Path("/Users/donarundas/Projects/DAN")
EXTRACT_DIR = ROOT / "DAN_Publications"
INDEX_DIR = ROOT / "index"
TABLE_INDEX = INDEX_DIR / "index_tables.faiss"
TABLE_META = INDEX_DIR / "vector_metadata_tables.json"
CACHE_DB = INDEX_DIR / "embed_cache.sqlite3"
MODEL = "text-embedding-3-small"   # lightweight for structured data

load_dotenv(ROOT / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def csv_to_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = [" | ".join(r) for r in reader if any(r)]
    return " | ".join(rows[:30])  # cap 30 rows for brevity

async def embed_batch(session, texts):
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": MODEL, "input": texts}
    async with session.post(url, json=payload, headers=headers) as r:
        data = await r.json()
        return [d["embedding"] for d in data.get("data", [])]

async def main():
    csvs = list(EXTRACT_DIR.rglob("*.csv"))
    print(f"🧾 Found {len(csvs)} CSV tables")
    texts, metas = [], []

    for path in csvs:
        t = csv_to_text(path)
        if len(t) < 50: 
            continue
        texts.append(t)
        metas.append({
            "source": str(path.relative_to(ROOT)),
            "len": len(t),
            "type": "table",
        })

    async with aiohttp.ClientSession() as s:
        B = 64
        embeddings = []
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            vecs = await embed_batch(s, batch)
            embeddings.extend(vecs)

    # build FAISS
    arr = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, str(TABLE_INDEX))

    json.dump({"built": datetime.utcnow().isoformat()+"Z", "chunks": metas},
              open(TABLE_META, "w"), indent=2)
    print(f"✅ {len(metas)} tables embedded → {TABLE_INDEX}")

if __name__ == "__main__":
    asyncio.run(main())
