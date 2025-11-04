# /Users/donarundas/Projects/DAN/app_streamlit.py
import streamlit as st
from pathlib import Path
from article_generator import generate_article
from hybrid_retriever import hybrid_search
import pandas as pd
import base64

ROOT = Path("/Users/donarundas/Projects/DAN")

st.set_page_config(page_title="DAN Research Explorer", layout="wide")
st.title("🌊 DAN Research Explorer")

def encode_image(path):
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return None

def display_table(path, caption=None):
    try:
        df = pd.read_csv(path)
        st.dataframe(df.head(15), use_container_width=True)
        if caption:
            st.caption(caption)
    except Exception as e:
        st.warning(f"⚠️ Could not display table {path}: {e}")

query = st.text_input("🔎 Search diving or hyperbaric topic:", placeholder="e.g. decompression illness treatment")

if st.button("Generate Summary"):
    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    with st.spinner("🔍 Retrieving relevant DAN publications..."):
        results = hybrid_search(query, top_k=10)

    with st.spinner("🧠 Generating GPT-5 article... (check console for stream)"):
        article = generate_article(query, results)

    # --- Article Section ---
    st.markdown("## 🧾 Summary Article")
    st.markdown(article)

    # --- Related Tables & Figures ---
    st.markdown("## 📊 Related Tables & Figures")
    for r in results:
        src = Path(r["source"])
        caption = r.get("preview", "")
        page = r.get("page", "?")
        modality = r["modality"]

        if modality == "tables" and src.suffix == ".csv":
            st.markdown(f"**Table** — {caption} (Page {page})")
            display_table(ROOT / src, caption)

        elif modality == "images":
            img_path = ROOT / src
            img_data = encode_image(str(img_path))
            if img_data:
                st.markdown(f"**Figure** — {caption} (Page {page})")
                st.markdown(f"![{caption}]({img_data})", unsafe_allow_html=True)

    # --- References Section ---
    st.markdown("## 🔖 References")
    refs_seen = set()
    for r in results:
        src = Path(r["source"])
        pub_name = src.parent.name
        page = r.get("page", "?")
        if pub_name not in refs_seen:
            st.markdown(f"- **{pub_name}** (Page {page})")
            refs_seen.add(pub_name)
