#!/usr/bin/env python3
"""
index_unified_builder.py
────────────────────────
Merges all modality indexes (text, tables, images)
into a single unified FAISS index with metadata audit.

Inputs:
- vector_index.faiss
- index_tables.faiss
- index_images.faiss
Outputs:
- index_unified.faiss
- vector_metadata_unified.json
"""

import os, json, faiss
import numpy as np
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
ROOT = Path("/Users/donarundas/Projects/DAN/index")
INDEX_FILES = {
    "text": ROOT / "vector_index.faiss",
    "tables": ROOT / "index_tables.faiss",
    "images": ROOT / "index_images.faiss",
}
META_FILES = {
    "text": ROOT / "vector_metadata.json",
    "tables": ROOT / "vector_metadata_tables.json",
    "images": ROOT / "vector_metadata_images.json",
}
OUT_FAISS = ROOT / "index_unified.faiss"
OUT_META = ROOT / "vector_metadata_unified.json"

# -------------------------------------------------------------------
# LOAD HELPERS
# -------------------------------------------------------------------
def load_faiss(path):
    if not path.exists():
        print(f"⚠️ Missing FAISS index: {path}")
        return None
    idx = faiss.read_index(str(path))
    print(f"✅ Loaded {path.name}: {idx.ntotal} vectors, dim={idx.d}")
    return idx

def load_metadata(path, modality):
    if not path.exists():
        print(f"⚠️ Missing metadata: {path}")
        return []
    data = json.load(open(path))
    for c in data.get("chunks", []):
        c["modality"] = modality
    return data.get("chunks", [])

# -------------------------------------------------------------------
# MERGE PROCESS
# -------------------------------------------------------------------
def merge_indices(indexes):
    dims = {idx.d for idx in indexes if idx}
    if len(dims) != 1:
        raise ValueError(f"❌ Inconsistent dimensions across indexes: {dims}")

    d = dims.pop()
    unified = faiss.IndexFlatL2(d)
    for idx in indexes:
        if idx:
            arr = np.zeros((idx.ntotal, d), dtype="float32")
            idx.reconstruct_n(0, idx.ntotal, arr)
            unified.add(arr)
    return unified

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print("🔗 Building unified multimodal index...\n")

    indexes = [load_faiss(p) for p in INDEX_FILES.values()]
    metas = []
    for modality, path in META_FILES.items():
        metas.extend(load_metadata(path, modality))

    # Build unified FAISS
    unified = merge_indices(indexes)
    faiss.write_index(unified, str(OUT_FAISS))

    # Merge metadata
    unified_meta = {
        "built": datetime.utcnow().isoformat() + "Z",
        "count_total": len(metas),
        "by_modality": {k: load_faiss(p).ntotal if load_faiss(p) else 0 for k, p in INDEX_FILES.items()},
        "chunks": metas,
    }
    json.dump(unified_meta, open(OUT_META, "w"), indent=2)

    print("\n🎉 Unified FAISS index built successfully")
    print(f"🧠 Total vectors: {unified.ntotal}")
    print(f"💾 Saved → {OUT_FAISS}")
    print(f"🧾 Metadata → {OUT_META}")

if __name__ == "__main__":
    main()
