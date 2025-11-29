#!/usr/bin/env python3
"""
app_streamlit.py — DAN Research Explorer Dashboard
───────────────────────────────────────────────────
Streamlit frontend for hybrid retriever + GPT-5 summarization.
Focus: Markdown text (primary), Tables (context), Images (optional)
"""

import streamlit as st
import pandas as pd
import base64
from pathlib import Path
from hybrid_retriever import hybrid_search
from article_generator import generate_article

st.set_page_config(page_title="🌊 DAN Research Explorer", layout="wide")

st.title("🌊 DAN Research Explorer")
st.markdown("Explore Diving & Hyperbaric Medicine publications with AI-assisted summarization")

# ──────────────────────────────────────────────────────────────
# 1️⃣  Search Input
# ──────────────────────────────────────────────────────────────
query = st.text_input("🔍 Enter a research topic", placeholder="e.g. decompression sickness in diabetic divers")

if query:
    st.markdown("### 📋 Raw Retrieved Results")
    results = hybrid_search(query)

    # Split by modality
    text_results = [r for r in results if r["modality"] == "text"]
    table_results = [r for r in results if r["modality"] == "tables"]
    image_results = [r for r in results if r["modality"] == "images"]

    # ──────────────────────────────────────────────────────────────
    # 2️⃣  Display Raw Results
    # ──────────────────────────────────────────────────────────────
    for i, r in enumerate(results, 1):
        st.markdown(f"**{i}. [{r['modality'].upper()}]** — {r['preview'][:200]}...")
        st.caption(f"↳ {r['source']} | Score: {r['score']:.4f}")

    # ──────────────────────────────────────────────────────────────
    # 3️⃣  GPT-5 Article Summarization
    # ──────────────────────────────────────────────────────────────
    with st.spinner("Generating article summary with GPT-5..."):
        article = generate_article(query, text_results + table_results + image_results)

    st.markdown("## 🧾 Summary Article")
    if article:
        st.markdown(article)
    else:
        st.warning("No summary generated — check retriever output.")

    # ──────────────────────────────────────────────────────────────
    # 4️⃣  Related Tables
    # ──────────────────────────────────────────────────────────────
    if table_results:
        st.markdown("## 📊 Related Tables")
        for t in table_results[:5]:
            tpath = Path(t["source"])
            if tpath.exists() and tpath.suffix == ".csv":
                try:
                    df = pd.read_csv(tpath)
                    st.markdown(f"**{tpath.name}** — {len(df)} rows")
                    st.dataframe(df.head(10))
                except Exception as e:
                    st.warning(f"⚠️ Could not load {tpath.name}: {e}")
    else:
        st.info("No related tables found for this topic.")
    

    # ──────────────────────────────────────────────────────────────
    # 5️⃣  Related Figures
    # ──────────────────────────────────────────────────────────────
    if image_results:
        st.markdown("## 🖼 Related Figures")
        cols = st.columns(3)
        for i, img in enumerate(image_results[:6]):
            col = cols[i % 3]
            ipath = Path(img["source"])
            if ipath.exists() and ipath.is_file():
                try:
                    with open(ipath, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    caption = img.get("preview", ipath.stem)
                    col.image(f"data:image/png;base64,{b64}", caption=caption, use_container_width=True)
                except Exception as e:
                    col.caption(f"⚠️ Error displaying {ipath.name}: {e}")
    else:
        st.info("No relevant figures found.")
    

    # ──────────────────────────────────────────────────────────────
    # 6️⃣  Reference Section
    # ──────────────────────────────────────────────────────────────
    # 6️⃣  Reference Section
    st.markdown("## 🔖 References")
    
    refs = []
    for r in results:
        src = Path(r["source"])
        pub = (
            src.parts[-3]
            if len(src.parts) >= 3 and "DAN_Publications" in src.parts
            else src.stem
        )
        # Extract page number if present in filename
        page_hint = ""
        for token in src.stem.split("_"):
            if token.startswith("page"):
                try:
                    page_hint = f"(Page {int(token.replace('page', '').strip())})"
                    break
                except:
                    pass
        refs.append(f"- **{pub.replace('_', ' ')}** {page_hint}")
    
    st.markdown("\n".join(sorted(set(refs))) or "_No referenced sources found._")
    
else:
    st.info("Enter a query to begin exploring.")
