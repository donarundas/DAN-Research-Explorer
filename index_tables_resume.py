#!/usr/bin/env python3
import json, faiss, numpy as np
from openai import OpenAI
from tqdm import tqdm

ROOT = "/Users/donarundas/Projects/DAN/index"
INDEX_FILE = f"{ROOT}/index_tables.faiss"
META_FILE = f"{ROOT}/vector_metadata_tables.json"
MODEL = "text-embedding-3-small"

client = OpenAI()

# Load metadata + FAISS
meta_all = json.load(open(META_FILE))
idx = faiss.read_index(INDEX_FILE)
n_indexed = idx.ntotal
print(f"🔍 FAISS has {n_indexed}, metadata has {len(meta_all)}")

# Find unindexed entries
missing = meta_all[n_indexed:]
print(f"🧩 Missing: {len(missing)}")

# Re-embed and append
if missing:
    new_vecs = []
    for m in tqdm(missing):
        text = m.get("text", "")
        if not text.strip():
            continue
        emb = client.embeddings.create(model=MODEL, input=text)
        new_vecs.append(np.array(emb.data[0].embedding, dtype="float32"))
    if new_vecs:
        arr = np.vstack(new_vecs)
        idx.add(arr)
        faiss.write_index(idx, INDEX_FILE)
        print(f"✅ Added {len(new_vecs)} new vectors. Now total = {idx.ntotal}")
    else:
        print("⚠️ No valid vectors to add.")
else:
    print("✅ FAISS already up to date.")
