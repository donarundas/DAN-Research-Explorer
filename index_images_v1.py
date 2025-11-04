#!/usr/bin/env python3
"""
index_images_v1.py
------------------
Generates captions for figures using GPT-5 Vision,
then embeds captions with text-embedding-3-small.
"""

import os, json, faiss, base64, numpy as np
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime

ROOT = Path("/Users/donarundas/Projects/DAN")
IMG_DIR = ROOT / "DAN_Publications"
INDEX_DIR = ROOT / "index"
OUT_FAISS = INDEX_DIR / "index_images.faiss"
OUT_META = INDEX_DIR / "vector_metadata_images.json"

client = OpenAI()

def gen_caption(img_path):
                            try:
                                # Convert image to base64 string
                                with open(img_path, "rb") as f:
                                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
                        
                                # Send as base64 data URI
                                resp = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "You are a scientific image captioner. Write a concise, factual caption for this figure for academic retrieval."
                                        },
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": "Describe this scientific figure in one sentence."},
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/png;base64,{img_base64}"
                                                    }
                                                }
                                            ]
                                        }
                                    ],
                                    max_tokens=80,
                                )
                        
                                caption = resp.choices[0].message.content.strip()
                                return caption
                        
                            except Exception as e:
                                print(f"[WARN] caption failed {img_path}: {e}")
                                return ""
                
        

def embed_text(text):
    r = client.embeddings.create(model="text-embedding-3-small", input=text)
    return r.data[0].embedding

def main():
    imgs = list(IMG_DIR.rglob("*.[pj][pn]g"))
    print(f"🖼 Found {len(imgs)} images")
    meta, vecs = [], []

    for img in tqdm(imgs):
        cap = gen_caption(img)
        if not cap: continue
        emb = embed_text(cap)
        vecs.append(emb)
        meta.append({"src": str(img.relative_to(ROOT)), "caption": cap})

    arr = np.array(vecs, dtype="float32")
    idx = faiss.IndexFlatL2(arr.shape[1])
    idx.add(arr)
    faiss.write_index(idx, str(OUT_FAISS))
    json.dump({"built": datetime.utcnow().isoformat()+"Z","chunks":meta},
              open(OUT_META,"w"), indent=2)
    print(f"✅ Image captions embedded → {OUT_FAISS}")

if __name__ == "__main__":
    main()
