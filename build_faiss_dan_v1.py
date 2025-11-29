#!/usr/bin/env python3
# build_faiss_dan_v1.py
# FAISS index builder for DAN corpus:
# - citation-aware embeddings (title/authors/journal/year + chunk text)
# - zstd JSONL streaming
# - tqdm progress bar
# - identical architectural style to DHM v4 builder

import os
import sys
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import faiss
import zstandard as zstd
from tqdm import tqdm
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────
# Config (EDIT THESE)
# ─────────────────────────────────────────────────────────────

# Path to your DAN chunks (produced by your DAN ingestion pipeline)
CHUNKS_ZST = "data/ingestion_jsonl/dan_chunks_v1.jsonl.zst"

# Output folder for FAISS + idmap
OUT_DIR = "data/faiss_index_dan_v1"

# Embedding
EMBED_MODEL = "text-embedding-3-large"
BATCH_SIZE = 128
MAX_TOK_CHARS = 4000  # safety cap

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def println(msg: str):
    sys.stdout.write(msg + ("\n" if not msg.endswith("\n") else ""))
    sys.stdout.flush()

def mkpath_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_env_openai() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)

def read_jsonl_zst(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks file not found: {path}")
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        with dctx.stream_reader(f) as r:
            text_stream = io.TextIOWrapper(r, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    println(f"[WARN] Failed to parse JSON line: {e}")
                    continue

def build_citation_aware_text(obj: Dict[str, Any]) -> str:
    """
    Build text for embedding:
    [TITLE | AUTHORS | JOURNAL | YEAR]\n\n[chunk text]
    """
    meta = obj.get("meta") or {}
    title = meta.get("title") or meta.get("verified_title") or ""
    authors = meta.get("authors") or meta.get("first_author") or ""
    journal = meta.get("journal") or ""
    year = str(meta.get("year") or "")

    header_parts = [p for p in [title, authors, journal, year] if p]
    header = " | ".join(header_parts)

    body = obj.get("text", "") or ""
    if len(body) > MAX_TOK_CHARS:
        body = body[:MAX_TOK_CHARS]

    return f"{header}\n\n{body}" if header else body

def embed_batch(client: OpenAI, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    vecs = np.array([x.embedding for x in resp.data], dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs / norms  # cosine-friendly

# ─────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────

def build_faiss_dan_v1():
    println(f"[START] {utcnow()}  DAN FAISS v1 builder")
    println(f"[INFO] Reading: {CHUNKS_ZST}")

    client = load_env_openai()

    out_dir = mkpath_dir(OUT_DIR)
    index_path = out_dir / "index.faiss"
    idmap_path = out_dir / "idmap.npy"
    meta_path = out_dir / "build_meta.json"

    # Pre-count lines for tqdm
    println("[INFO] Counting chunks for progress bar…")
    total_est = sum(1 for _ in read_jsonl_zst(CHUNKS_ZST))
    println(f"[INFO] Estimated chunks: {total_est}")

    all_ids = []
    index = None
    batch_texts = []
    batch_ids = []
    dim = None
    total_chunks = 0

    # Main loop
    for obj in tqdm(read_jsonl_zst(CHUNKS_ZST), total=total_est, desc="Embedding DAN"):
        cid = obj.get("id") or f"DAN::auto::{total_chunks:06d}"

        text_for_embed = build_citation_aware_text(obj)
        if not text_for_embed.strip():
            continue

        batch_texts.append(text_for_embed)
        batch_ids.append(str(cid))
        total_chunks += 1

        if len(batch_texts) >= BATCH_SIZE:
            vecs = embed_batch(client, batch_texts)
            if vecs.shape[0] > 0:
                if index is None:
                    dim = vecs.shape[1]
                    println(f"[INFO] Embedding dim: {dim}")
                    index = faiss.IndexFlatIP(dim)
                index.add(vecs)
                all_ids.extend(batch_ids)

            batch_texts.clear()
            batch_ids.clear()

    # Flush remainder
    if batch_texts:
        vecs = embed_batch(client, batch_texts)
        if vecs.shape[0] > 0:
            if index is None:
                dim = vecs.shape[1]
                println(f"[INFO] Embedding dim: {dim}")
                index = faiss.IndexFlatIP(dim)
            index.add(vecs)
            all_ids.extend(batch_ids)

    println(f"[DONE] Total embedded DAN chunks: {len(all_ids)}")
    println(f"[INFO] FAISS index size: {index.ntotal}")

    # Save
    faiss.write_index(index, str(index_path))
    np.save(idmap_path, np.array(all_ids, dtype=object), allow_pickle=True)

    meta = {
        "built_at": utcnow(),
        "chunks_source": CHUNKS_ZST,
        "faiss_index": str(index_path),
        "idmap": str(idmap_path),
        "embedding_model": EMBED_MODEL,
        "index_type": "IndexFlatIP (L2-normalised cosine)",
        "total_chunks_indexed": len(all_ids),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    println(f"[DONE] FAISS v1 DAN index → {index_path}")
    println(f"[DONE] idmap written → {idmap_path}")
    println(f"[DONE] meta → {meta_path}")
    println(f"[END] {utcnow()} DAN FAISS v1 build complete")

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        build_faiss_dan_v1()
    except KeyboardInterrupt:
        println("\n[ABORT] Interrupted")
        sys.exit(1)
    except Exception as e:
        println(f"[FATAL] {e}")
        sys.exit(2)
