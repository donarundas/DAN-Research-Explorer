#!/usr/bin/env python3
"""
index_tables_v2.py
------------------
Indexes DAN table CSVs with embedded metadata headers
for improved LLM retrieval and report reconstruction.
"""

import os, csv, json, sqlite3, asyncio, aiohttp, hashlib
import numpy as np, faiss
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
ROOT = Path("/Users/donarundas/Projects/DAN")
EXTRACT_DIR = ROOT / "DAN_Publications"
INDEX_DIR = ROOT / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

TABLE_INDEX = INDEX_DIR / "index_tables.faiss"
TABLE_META = INDEX_DIR / "vector_metadata_tables.json"
CACHE_DB = INDEX_DIR / "embed_cache.sqlite3"

MODEL = "text-embedding-3-small"
BATCH = 64

load_dotenv(ROOT / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Missing OpenAI API key"


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def parse_csv_with_metadata(path):
    """Return (meta_text, table_text, combined_text)."""
    meta_lines, data_lines = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                meta_lines.append(line.strip("# ").strip())
            else:
                data_lines.append(line)
    meta_text = " | ".join(meta_lines)
    reader = csv.reader(data_lines)
    rows = [" | ".join(r) for r in reader if any(r)]
    table_text = " || ".join(rows[:30])
    combined = f"{meta_text} || {table_text}"
    return meta_text, table_text, combined


def make_cache_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def embed_batch(session, texts):
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": MODEL, "input": texts}
    async with session.post(url, json=payload, headers=headers) as r:
        data = await r.json()
        return [d["embedding"] for d in data.get("data", [])]


def init_cache():
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS cache (
        key TEXT PRIMARY KEY,
        embedding BLOB
    )""")
    conn.commit()
    return conn


def get_cached_embeddings(conn, keys):
    cur = conn.cursor()
    results = {}
    for k in keys:
        cur.execute("SELECT embedding FROM cache WHERE key=?", (k,))
        row = cur.fetchone()
        if row:
            results[k] = np.frombuffer(row[0], dtype=np.float32)
    return results


def save_cached_embeddings(conn, key_vec_pairs):
    cur = conn.cursor()
    for k, v in key_vec_pairs.items():
        cur.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (k, v.tobytes()))
    conn.commit()


# -------------------------------------------------------------------
# MAIN ASYNC INDEXING
# -------------------------------------------------------------------
async def main():
    csvs = list(EXTRACT_DIR.rglob("*.csv"))
    print(f"🧾 Found {len(csvs)} CSV tables")

    conn = init_cache()
    all_texts, metas, new_vecs = [], [], []

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(csvs), BATCH):
            batch = csvs[i:i+BATCH]
            cache_keys = []
            inputs = []

            for csv_path in batch:
                meta_text, table_text, combined = parse_csv_with_metadata(csv_path)
                cache_key = make_cache_key(combined)
                cache_keys.append(cache_key)

                # Prepare metadata record
                metas.append({
                    "path": str(csv_path),
                    "meta_text": meta_text,
                    "table_text": table_text,
                    "timestamp": datetime.now().isoformat()
                })
                all_texts.append(combined)
            
            cached = get_cached_embeddings(conn, cache_keys)
            missing_texts = [t for t, k in zip(all_texts[-len(batch):], cache_keys) if k not in cached]
            if missing_texts:
                new_embs = await embed_batch(session, missing_texts)
                # Save new embeddings
                new_pairs = {
                    make_cache_key(t): np.array(e, dtype=np.float32)
                    for t, e in zip(missing_texts, new_embs)
                }
                save_cached_embeddings(conn, new_pairs)
                cached.update(new_pairs)

            # Aggregate vectors in batch order
            batch_vecs = []
            for k in cache_keys:
                if k in cached:
                    batch_vecs.append(cached[k])
                else:
                    print(f"⚠️ Missing embedding for key {k[:8]}... (skipped)")
            if not batch_vecs:
                print("⚠️ Entire batch failed — moving on.")
                continue
            new_vecs.extend(batch_vecs)
            

    # Build FAISS index
    embeddings = np.vstack(new_vecs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(TABLE_INDEX))

    with open(TABLE_META, "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2)

    print(f"✅ Indexed {len(new_vecs)} tables → {TABLE_INDEX.name}")
    print(f"🗂️ Metadata → {TABLE_META.name}")


if __name__ == "__main__":
    asyncio.run(main())
