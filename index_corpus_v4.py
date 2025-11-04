#!/usr/bin/env python3
"""
index_corpus_v4.py
------------------
Stable async FAISS builder for DAN corpus (Markdown-only mode).

Upgrades:
✅ Markdown-only mode (ignores tables/images)
✅ Adaptive throttling on 429s
✅ Automatic split of long text chunks
✅ Hard cooldown for persistent rate-limits
✅ Checkpointing + SQLite cache
✅ Ctrl-C safe graceful exit
"""

import os, json, faiss, sqlite3, hashlib, asyncio, aiohttp, random, time, contextlib
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
ROOT = Path("/Users/donarundas/Projects/DAN")
MANIFEST_PATH = ROOT / "dan_content_manifest.json"
OUT_DIR = ROOT / "index"
CACHE_DB = OUT_DIR / "embed_cache.sqlite3"

load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

MODEL_BODY  = os.getenv("MODEL_BODY", "text-embedding-3-large")
CHUNK_MAX_TOKENS  = int(os.getenv("CHUNK_MAX_TOKENS", 300))
BATCH_SIZE        = int(os.getenv("BATCH_SIZE", 4))
MAX_PARALLEL_REQS = int(os.getenv("MAX_PARALLEL_REQS", 1))
CHECKPOINT_EVERY  = 1000
MAX_TOKENS_PER_ITEM = 7000
HARD_COOLDOWN_SECONDS = int(os.getenv("HARD_COOLDOWN_SECONDS", 240))

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def chunk_markdown(text: str, max_tokens: int = CHUNK_MAX_TOKENS):
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
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

def load_manifest(path: Path):
    return json.loads(path.read_text())

def dedup(texts, metas):
    seen, out_t, out_m = {}, [], []
    for t, m in zip(texts, metas):
        h = hash_text(t)
        if h in seen:
            continue
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
        hash TEXT PRIMARY KEY,
        model TEXT,
        vector TEXT)""")
    return conn

def cache_lookup(conn, hashes, model):
    if not hashes:
        return {}
    q = ",".join("?" * len(hashes))
    cur = conn.execute(
        f"SELECT hash, vector FROM cache WHERE hash IN ({q}) AND model=?",
        (*hashes, model),
    )
    return {h: json.loads(v) for h, v in cur.fetchall()}

def cache_store(conn, model, mapping):
    for h, v in mapping.items():
        conn.execute(
            "INSERT OR REPLACE INTO cache(hash, model, vector) VALUES (?,?,?)",
            (h, model, json.dumps(v)),
        )
    conn.commit()

# -------------------------------------------------------------------
# EMBEDDING HELPERS
# -------------------------------------------------------------------
def split_or_trim(text):
    words = text.split()
    if len(words) <= MAX_TOKENS_PER_ITEM:
        return [text]
    chunks = []
    for i in range(0, len(words), MAX_TOKENS_PER_ITEM):
        chunks.append(" ".join(words[i:i + MAX_TOKENS_PER_ITEM]))
    return chunks

# -------------------------------------------------------------------
# EMBEDDING (patch: 429, 400, timeouts, cooldown)
# -------------------------------------------------------------------
async def embed_batch(session, texts, model, key, sem):
    """Embed a batch robustly with full backoff and cooldown."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    cleaned = []
    for t in texts:
        cleaned.extend(split_or_trim(t) if len(t.split()) > MAX_TOKENS_PER_ITEM else [t])

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=600)
    attempt = 0

    while True:  # infinite retry until success or manual interrupt
        payload = {"input": cleaned, "model": model}
        try:
            async with sem, session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                if r.status == 200:
                    try:
                        data = await r.json()
                        return [d["embedding"] for d in data["data"]]
                    except asyncio.TimeoutError:
                        print("[TIMEOUT] reading JSON; retrying…")
                        await asyncio.sleep(10)
                        continue

                elif r.status == 429:
                    wait = min(90, (2 ** min(attempt, 6)) + random.uniform(1, 5))
                    print(f"[429] rate-limit: waiting {wait:.1f}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                    attempt += 1
                    if attempt % 6 == 0:
                        print(f"[429] hard cooldown {HARD_COOLDOWN_SECONDS}s…")
                        await asyncio.sleep(HARD_COOLDOWN_SECONDS)

                elif r.status == 400:
                    # Deep split fallback
                    from tiktoken import get_encoding
                    enc = get_encoding("cl100k_base")
                    lens = [len(enc.encode(t)) for t in cleaned]
                    print(f"[400] Oversized batch (max {max(lens)} tokens). Re-splitting…")

                    safe = []
                    for t in cleaned:
                        toks = enc.encode(t)
                        if len(toks) > 6000:
                            step = 4000
                            for i in range(0, len(toks), step):
                                safe.append(enc.decode(toks[i:i + step]))
                        else:
                            safe.append(t)
                    cleaned = safe
                    attempt = 0
                    continue

                else:
                    wait = min(60, (2 ** min(attempt, 5)) + random.uniform(1, 4))
                    print(f"[WARN] HTTP {r.status}; retrying in {wait:.1f}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                    attempt += 1

        except asyncio.TimeoutError:
            print("[TIMEOUT] network stall; sleeping 15s then retrying batch…")
            await asyncio.sleep(15)
            attempt += 1
        except aiohttp.ClientOSError as e:
            print(f"[OSError] {e}; sleeping 15s then retrying batch…")
            await asyncio.sleep(15)
            attempt += 1
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Gracefully stopping after current batch…")
            raise

# -------------------------------------------------------------------
# EMBED ALL (Markdown-only)
# -------------------------------------------------------------------
async def embed_all_async(texts, metas, model, key, conn):
    sem = asyncio.Semaphore(MAX_PARALLEL_REQS)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=600)
    hashes = [m["hash"] for m in metas]
    cached = cache_lookup(conn, hashes, model)
    vectors = [cached.get(h) for h in hashes]
    missing = [i for i, v in enumerate(vectors) if v is None]

    print(f"[{model}] total={len(texts)}, missing={len(missing)}, cached={len(texts)-len(missing)}")

    processed_since_checkpoint = 0
    start_time = time.time()

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for start in range(0, len(missing), BATCH_SIZE):
                batch_idx = missing[start:start + BATCH_SIZE]
                batch_texts = [texts[i] for i in batch_idx]

                embs = await embed_batch(session, batch_texts, model, key, sem)
                for j, idx in enumerate(batch_idx[:len(embs)]):
                    vectors[idx] = embs[j]
                    cache_store(conn, model, {hashes[idx]: embs[j]})
                    processed_since_checkpoint += 1

                await asyncio.sleep(random.uniform(0.5, 1.5))

                if processed_since_checkpoint >= max(1, CHECKPOINT_EVERY // max(1, BATCH_SIZE)):
                    conn.commit()
                    elapsed = time.time() - start_time
                    print(f"[checkpoint] processed {processed_since_checkpoint} | elapsed {elapsed/60:.1f} min")
                    processed_since_checkpoint = 0
                    start_time = time.time()
                    await asyncio.sleep(random.uniform(2.0, 5.0))

    except KeyboardInterrupt:
        with contextlib.suppress(Exception):
            conn.commit()
        print("\n[INTERRUPT] Cache committed. Safe to rerun and resume.")
        raise

    for i, v in enumerate(vectors):
        if v is None:
            vectors[i] = [0.0] * 1536
        metas[i]["model_used"] = model
    return vectors, metas

# -------------------------------------------------------------------
# FAISS
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
    faiss_path = OUT_DIR / "vector_index.faiss"
    meta_path  = OUT_DIR / "vector_metadata.json"
    faiss.write_index(index, str(faiss_path))
    meta = {"built": datetime.utcnow().isoformat() + "Z", "chunks": metas}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"✅ Saved index → {faiss_path}\n✅ Metadata → {meta_path}")

# -------------------------------------------------------------------
# MAIN (Markdown-only embedding)
# -------------------------------------------------------------------
async def main_async():
    print("🔍 Markdown-only embedding mode active.")

    m = load_manifest(MANIFEST_PATH)
    body_texts, body_meta = [], []

    for rec in m["records"]:
        pub = rec["publication"]
        md = Path(rec["markdown_file"])
        if not md.exists():
            continue
        text = safe_read_text(md)
        if text.strip():
            chunks = chunk_markdown(text)
            for i, c in enumerate(chunks):
                body_texts.append(c)
                body_meta.append({
                    "pub": pub,
                    "type": "markdown_text",
                    "chunk_index": i,
                    "source": str(md)
                })

    body_texts, body_meta = dedup(body_texts, body_meta)
    print(f"[STATS] Markdown chunks (post-dedupe): {len(body_texts)}")

    conn = cache_init(CACHE_DB)
    vectors, meta = await embed_all_async(body_texts, body_meta, MODEL_BODY, OPENAI_API_KEY, conn)

    index = build_faiss(vectors)
    save_outputs(index, meta)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
