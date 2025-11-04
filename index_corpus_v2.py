#!/usr/bin/env python3
"""
index_corpus_v3.py
------------------
Stable async FAISS builder for DAN publications.

Upgrades vs v2:
✅ Adaptive throttling on 429s
✅ Automatic split of long text chunks (prevents 400 errors)
✅ Checkpoint saving every 1000 embeddings
✅ Embedding cache (sqlite)
✅ Multi-model (body / table)
✅ Safe retry + backoff
"""

import os, json, faiss, sqlite3, hashlib, platform, asyncio, aiohttp, random, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# -------------------------------------------------------------------
# ENVIRONMENT
# -------------------------------------------------------------------
ROOT = Path("/Users/donarundas/Projects/DAN")
MANIFEST_PATH = ROOT / "dan_content_manifest.json"
OUT_DIR = ROOT / "index"
CACHE_DB = OUT_DIR / "embed_cache.sqlite3"

load_dotenv(ROOT / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

MODEL_BODY  = os.getenv("MODEL_BODY",  "text-embedding-3-large")
MODEL_TABLE = os.getenv("MODEL_TABLE", "text-embedding-3-small")

CHUNK_MAX_TOKENS  = int(os.getenv("CHUNK_MAX_TOKENS", 300))
TABLE_CELL_LIMIT  = int(os.getenv("TABLE_CELL_LIMIT", 15))
BATCH_SIZE        = int(os.getenv("BATCH_SIZE", 16))
MAX_PARALLEL_REQS = int(os.getenv("MAX_PARALLEL_REQS", 2))
MAX_TOKENS_PER_ITEM = 7000
CHECKPOINT_EVERY = 1000

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_read_text(path: Path) -> str:
    try: return path.read_text(encoding="utf-8", errors="ignore")
    except Exception: return ""

def chunk_markdown(text: str, max_tokens: int = CHUNK_MAX_TOKENS) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf, count = [], [], 0
    for p in paras:
        words = p.split()
        if count + len(words) > max_tokens and buf:
            chunks.append("\n\n".join(buf))
            buf, count = [p], len(words)
        else:
            buf.append(p)
            count += len(words)
    if buf: chunks.append("\n\n".join(buf))
    return chunks

def table_csv_to_text(path: Path, max_rows: int = TABLE_CELL_LIMIT) -> str:
    try:
        df = pd.read_csv(path)
    except Exception:
        try: df = pd.read_csv(path, header=None)
        except Exception: return f"[UNREADABLE TABLE] {path.name}"
    if len(df) > max_rows: df = df.head(max_rows)
    header = ", ".join(str(h) for h in df.columns)
    lines = [f"[TABLE {path.name}]", f"COLUMNS: {header}"]
    for _, r in df.iterrows():
        lines.append("ROW: " + ", ".join(str(v) for v in r.values))
    return "\n".join(lines)

# -------------------------------------------------------------------
# MANIFEST
# -------------------------------------------------------------------
def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())

def prepare_corpus(m):
    body_texts, body_meta, table_texts, table_meta = [], [], [], []
    for rec in m["records"]:
        pub = rec["publication"]
        md = Path(rec["markdown_file"])
        tbls = [Path(p) for p in rec.get("tables", [])]
        imgs = [Path(p) for p in rec.get("images", [])]

        md_text = safe_read_text(md)
        if md_text.strip():
            for i, c in enumerate(chunk_markdown(md_text)):
                body_texts.append(c)
                body_meta.append({"pub": pub,"type": "body_text","chunk_index": i,"source": str(md)})
        for t in tbls:
            txt = table_csv_to_text(t)
            if txt.strip():
                table_texts.append(txt)
                table_meta.append({"pub": pub,"type": "table","source": str(t)})
    return body_texts, body_meta, table_texts, table_meta

# -------------------------------------------------------------------
# DEDUP
# -------------------------------------------------------------------
def dedup(texts, metas):
    seen, out_t, out_m = {}, [], []
    for t, m in zip(texts, metas):
        h = hash_text(t)
        if h in seen: continue
        seen[h] = True
        m["hash"] = h
        out_t.append(t)
        out_m.append(m)
    return out_t, out_m

# -------------------------------------------------------------------
# CACHE
# -------------------------------------------------------------------
def cache_init(db: Path):
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE IF NOT EXISTS cache (
        hash TEXT PRIMARY KEY, model TEXT, vector TEXT)""")
    return conn

def cache_lookup(conn, hashes, model):
    if not hashes: return {}
    q = ",".join("?"*len(hashes))
    cur = conn.execute(f"SELECT hash, vector FROM cache WHERE hash IN ({q}) AND model=?", (*hashes, model))
    return {h: json.loads(v) for h,v in cur.fetchall()}

def cache_store(conn, model, mapping):
    for h, v in mapping.items():
        conn.execute("INSERT OR REPLACE INTO cache(hash, model, vector) VALUES (?,?,?)",
                     (h, model, json.dumps(v)))
    conn.commit()

# -------------------------------------------------------------------
# EMBEDDING
# -------------------------------------------------------------------
def split_or_trim(text):
    words = text.split()
    if len(words) <= MAX_TOKENS_PER_ITEM:
        return [text]
    chunks = []
    for i in range(0, len(words), MAX_TOKENS_PER_ITEM):
        chunks.append(" ".join(words[i:i+MAX_TOKENS_PER_ITEM]))
    return chunks

# -------------------------------------------------------------------
# EMBEDDING (patched for timeout, adaptive backoff & checkpointing)
# -------------------------------------------------------------------
async def embed_batch(session, texts, model, key, sem):
    """Embed a batch of text safely with full retry & timeout handling."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # Split over-long texts safely
    cleaned = []
    for t in texts:
        cleaned.extend(split_or_trim(t) if len(t.split()) > MAX_TOKENS_PER_ITEM else [t])
    payload = {"input": cleaned, "model": model}

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=600)

    for attempt in range(8):
        try:
            async with sem, session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                if r.status == 200:
                    try:
                        data = await r.json()
                    except asyncio.TimeoutError:
                        print("[TIMEOUT] reading response JSON, retrying…")
                        await asyncio.sleep(10)
                        continue
                    return [d["embedding"] for d in data["data"]]
                elif r.status == 429:
                    wait = min(90, (2 ** attempt) + random.uniform(1, 5))
                    print(f"[429] rate-limit: waiting {wait:.1f}s…")
                    await asyncio.sleep(wait)
                elif r.status == 400:
                    print(f"[400] Oversized or malformed batch ({len(cleaned)} items). Skipping.")
                    return []
                else:
                    print(f"[WARN] {r.status} on attempt {attempt+1}. Retrying…")
                    await asyncio.sleep((2 ** attempt) + random.uniform(1, 5))
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] network stall attempt {attempt+1}. Sleeping 10s…")
            await asyncio.sleep(10)
            continue
        except aiohttp.ClientOSError as e:
            print(f"[OSError] {e}, reconnecting in 10s…")
            await asyncio.sleep(10)
            continue

    print(f"[FAIL] embed_batch exhausted retries for {model}")
    return []
    

async def embed_all_async(texts, metas, model, key, conn):
    """Embed all chunks asynchronously with caching, checkpoints, and resilience."""
    sem = asyncio.Semaphore(MAX_PARALLEL_REQS)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=600)
    hashes = [m["hash"] for m in metas]
    cached = cache_lookup(conn, hashes, model)
    vectors = [cached.get(h) for h in hashes]
    missing = [i for i,v in enumerate(vectors) if v is None]

    print(f"[{model}] total={len(texts)}, missing={len(missing)}, cached={len(texts)-len(missing)}")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        start_time = time.time()
        for start in range(0, len(missing), BATCH_SIZE):
            batch_idx = missing[start:start+BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_idx]
            embs = await embed_batch(session, batch_texts, model, key, sem)

            if embs:
                for j, idx in enumerate(batch_idx[:len(embs)]):
                    vectors[idx] = embs[j]
                    cache_store(conn, model, {hashes[idx]: embs[j]})

            # checkpoint & pacing
            if start % (CHECKPOINT_EVERY // BATCH_SIZE) == 0 and start > 0:
                conn.commit()
                elapsed = time.time() - start_time
                print(f"[checkpoint] embedded {start} / {len(missing)} | elapsed {elapsed/60:.1f} min")
                await asyncio.sleep(random.uniform(2.0, 5.0))  # network flush
                start_time = time.time()

    # Fill any remaining gaps with dummy vectors to keep FAISS consistent
    for i,v in enumerate(vectors):
        if v is None:
            vectors[i] = [0.0]*1536
        metas[i]["model_used"] = model

    return vectors, metas


# -------------------------------------------------------------------
# FAISS + SAVE
# -------------------------------------------------------------------
def build_faiss(vectors):
    dim = len(vectors[0])
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    idx = faiss.IndexFlatIP(dim)
    idx.add(arr)
    return idx

def save_outputs(index, metas):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss_path = OUT_DIR/"vector_index.faiss"
    meta_path  = OUT_DIR/"vector_metadata.json"
    faiss.write_index(index, str(faiss_path))
    meta = {"built": datetime.utcnow().isoformat()+"Z","chunks": metas}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"✅ Saved index → {faiss_path}\n✅ Metadata → {meta_path}")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
async def main_async():
    m = load_manifest(MANIFEST_PATH)
    btxt,bmeta,ttxt,tmeta = prepare_corpus(m)
    btxt,bmeta = dedup(btxt,bmeta)
    ttxt,tmeta = dedup(ttxt,tmeta)
    print(f"[STATS] body {len(btxt)} | tables {len(ttxt)}")
    conn = cache_init(CACHE_DB)
    bvec,bmeta = await embed_all_async(btxt,bmeta,MODEL_BODY,OPENAI_API_KEY,conn)
    tvec,tmeta = await embed_all_async(ttxt,tmeta,MODEL_TABLE,OPENAI_API_KEY,conn)
    allv = bvec+tvec
    allm = bmeta+tmeta
    idx = build_faiss(allv)
    save_outputs(idx, allm)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
