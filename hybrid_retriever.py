#!/usr/bin/env python3
"""
hybrid_retriever.py
────────────────────
Hybrid multimodal retriever for DAN corpus:
- text (Markdown)
- tables (CSV)
- images (captioned)

Each modality uses its own embedding model & FAISS index.
"""

import faiss, json, numpy as np
from openai import OpenAI
from pathlib import Path

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
ROOT = Path("/Users/donarundas/Projects/DAN/index")

SOURCES = {
    "text": {
        "index": ROOT / "vector_index.faiss",
        "meta": ROOT / "vector_metadata.json",
        "model": "text-embedding-3-large"
    },
    "tables": {
        "index": ROOT / "index_tables.faiss",
        "meta": ROOT / "vector_metadata_tables.json",
        "model": "text-embedding-3-small"
    },
    "images": {
        "index": ROOT / "index_images.faiss",
        "meta": ROOT / "vector_metadata_images.json",
        "model": "text-embedding-3-small"
    },
}

TOP_K = 5  # results per modality
client = OpenAI()

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def load_index_and_meta(path_idx, path_meta):
    idx = faiss.read_index(str(path_idx))
    meta = json.load(open(path_meta))["chunks"]
    return idx, meta

def embed_query(query, model):
    emb = client.embeddings.create(model=model, input=query)
    return np.array(emb.data[0].embedding, dtype="float32").reshape(1, -1)

def search_faiss(vec, index, meta, k=5):
    D, I = index.search(vec, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(meta):
            results.append({
                "score": float(dist),
                "source": meta[idx].get("source", ""),
                "type": meta[idx].get("type", "unknown"),
                "preview": meta[idx].get("caption", meta[idx].get("source", ""))[:160]
            })
    return results

# -------------------------------------------------------------------
# HYBRID SEARCH
# -------------------------------------------------------------------
def hybrid_search(query, top_k=5):
    final_results = []

    for mod, cfg in SOURCES.items():
        idx, meta = load_index_and_meta(cfg["index"], cfg["meta"])
        qvec = embed_query(query, cfg["model"])
        res = search_faiss(qvec, idx, meta, k=top_k)
        for r in res:
            r["modality"] = mod
        final_results.extend(res)

    # Rank all results globally by L2 distance (lower = better)
    final_results.sort(key=lambda x: x["score"])
    return final_results[:top_k]

# -------------------------------------------------------------------
# CLI EXAMPLE
# -------------------------------------------------------------------
if __name__ == "__main__":
    query = input("🔍 Enter your search query: ")
    results = hybrid_search(query, top_k=10)
    print("\n📊 Hybrid Multimodal Results:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r['modality'].upper()}] {r['preview']}")
        print(f"   ↳ {r['source']}")
        print(f"   Score: {r['score']:.4f}")
