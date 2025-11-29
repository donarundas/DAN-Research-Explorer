#!/usr/bin/env python3
"""
hybrid_retriever.py
────────────────────
Hybrid retriever for DAN corpus:
- Markdown text (primary, high-weight)
- Tables and Images (contextual, low-weight)
"""

import faiss
import json
import numpy as np
from pathlib import Path
from openai import OpenAI

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
ROOT = Path("/Users/donarundas/Projects/DAN/index")

SOURCES = {
    "text": {
        "index": ROOT / "vector_index.faiss",
        "meta": ROOT / "vector_metadata.json",
        "model": "text-embedding-3-large",
        "top_k": 10,
        "weight": 1.0
    },
    "tables": {
        "index": ROOT / "index_tables.faiss",
        "meta": ROOT / "vector_metadata_tables.json",
        "model": "text-embedding-3-small",
        "top_k": 5,
        "weight": 1.2
    },
    "images": {
        "index": ROOT / "index_images.faiss",
        "meta": ROOT / "vector_metadata_images.json",
        "model": "text-embedding-3-small",
        "top_k": 3,
        "weight": 1.4
    },
}

client = OpenAI()
np.set_printoptions(precision=4, suppress=True)


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def load_index_and_meta(path_idx, path_meta):
    """Load FAISS index and metadata JSON (supports dict or list structure)."""
    idx = faiss.read_index(str(path_idx))
    with open(path_meta) as f:
        meta = json.load(f)
    if isinstance(meta, dict) and "chunks" in meta:
        meta = meta["chunks"]
    return idx, meta



def embed_query(query, model):
    emb = client.embeddings.create(model=model, input=query)
    return np.array(emb.data[0].embedding, dtype="float32").reshape(1, -1)


def search_faiss(vec, index, meta, k=5, weight=1.0):
    D, I = index.search(vec, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(meta):
            results.append({
                "score": float(dist) * weight,
                "source": meta[idx].get("source", ""),
                "type": meta[idx].get("type", "unknown"),
                "preview": meta[idx].get("caption", meta[idx].get("source", ""))[:200]
            })
    return results


# -------------------------------------------------------------------
# HYBRID SEARCH
# -------------------------------------------------------------------
def hybrid_search(query, top_k_text=8):
    """Primary focus on Markdown text; tables and images are secondary."""
    final_results = []

    # 🔹 TEXT (Markdown)
    tcfg = SOURCES["text"]
    tidx, tmeta = load_index_and_meta(tcfg["index"], tcfg["meta"])
    tvec = embed_query(query, tcfg["model"])
    text_res = search_faiss(tvec, tidx, tmeta, k=tcfg["top_k"], weight=tcfg["weight"])
    for r in text_res:
        r["modality"] = "text"
    final_results.extend(text_res)

    # 🔸 TABLES
    tcfg = SOURCES["tables"]
    tidx, tmeta = load_index_and_meta(tcfg["index"], tcfg["meta"])
    tvec = embed_query(query, tcfg["model"])
    table_res = search_faiss(tvec, tidx, tmeta, k=tcfg["top_k"], weight=tcfg["weight"])
    for r in table_res:
        r["modality"] = "tables"
    final_results.extend(table_res)

    # 🔸 IMAGES
    icfg = SOURCES["images"]
    iidx, imeta = load_index_and_meta(icfg["index"], icfg["meta"])
    ivec = embed_query(query, icfg["model"])
    image_res = search_faiss(ivec, iidx, imeta, k=icfg["top_k"], weight=icfg["weight"])
    for r in image_res:
        r["modality"] = "images"
    final_results.extend(image_res)

    # ✅ Rank globally by weighted score (lower = better)
    final_results.sort(key=lambda x: x["score"])

    # Return markdown-heavy mix
    top_texts = [r for r in final_results if r["modality"] == "text"][:top_k_text]
    top_tables = [r for r in final_results if r["modality"] == "tables"][:3]
    top_images = [r for r in final_results if r["modality"] == "images"][:2]

    return top_texts + top_tables + top_images


# -------------------------------------------------------------------
# CLI TEST
# -------------------------------------------------------------------
if __name__ == "__main__":
    query = input("🔍 Enter your search query: ").strip()
    results = hybrid_search(query)
    print(f"\n📊 Hybrid Results for: {query}\n───────────────────────────────")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['modality'].upper()}] {r['preview'][:160]}...")
        print(f"   ↳ {r['source']} (Score={r['score']:.4f})")
        if r.get('page'): print(f"   Page: {r['page']}")
        print()
