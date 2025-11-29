#!/usr/bin/env python3
# dan-query-v1.py
# DAN RAG pipeline v1: accident-analysis-focused, FAISS-validated query expansion,
# structured multi-page report with linked bibliography.

import os, sys, io, json, base64, hashlib, textwrap, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import faiss
import zstandard as zstd
from openai import OpenAI

try:
    from rich.console import Console
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    RICH_AVAILABLE = False
    _console = None

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

CONFIG: Dict[str, Any] = {
    "faiss_index": "data/faiss_index_dan_v1/index.faiss",
    "idmap": "data/faiss_index_dan_v1/idmap.npy",
    "chunks": "data/ingestion_jsonl/dan_chunks_v1.jsonl.zst",
    "cache_dir": "data/.cache/dan_query_results",
    "model": "gpt-5-chat-latest",
    "fallback_model": "gpt-5-mini",
    "embedding_model": "text-embedding-3-large",
    "expansion_model": "gpt-4o",
    "top_k": 48,
    "max_docs": 32,
    "max_completion_tokens": 6000,
    "max_doc_chars": 2200,
}
ROOT_DIR = Path(__file__).resolve().parent

def _resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT_DIR / p

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def mkpath(p: str) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def println(msg: str) -> None:
    sys.stdout.write(msg + ("\n" if not msg.endswith("\n") else ""))
    sys.stdout.flush()

def load_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# ─────────────────────────────────────────────────────────────
# OpenAI helper
# ─────────────────────────────────────────────────────────────

def new_client() -> OpenAI:
    load_env()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

# ─────────────────────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────────────────────

CACHE_DIR = _resolve_path(CONFIG["cache_dir"])
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cache_key(query: str) -> Path:
    h = hashlib.sha1(query.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.json"

def cache_load(query: str) -> Optional[Dict[str, Any]]:
    p = cache_key(query)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def cache_save(query: str, result: Dict[str, Any]):
    p = cache_key(query)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False))

# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def read_jsonl_zst(path: str):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        with dctx.stream_reader(f) as r:
            for line in io.TextIOWrapper(r, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

def load_chunks() -> List[Dict[str, Any]]:
    path = _resolve_path(CONFIG["chunks"])
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks = []
    for obj in read_jsonl_zst(str(path)):
        meta = obj.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        obj["meta"] = meta

        # DAN-specific: normalise IDs and section-equivalent fields
        if "section" not in meta:
            meta.setdefault("section", meta.get("category") or "")

        pid = (
            meta.get("report_id")
            or meta.get("incident_id")
            or meta.get("parent_id")
            or meta.get("uuid")
            or ""
        )
        meta.setdefault("paper_id", pid)

        chunks.append(obj)
    return chunks

def load_faiss_index() -> faiss.Index:
    idx_path = _resolve_path(CONFIG["faiss_index"])
    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    return faiss.read_index(str(idx_path))

def load_idmap() -> np.ndarray:
    p = _resolve_path(CONFIG["idmap"])
    if not p.exists():
        raise FileNotFoundError(f"ID map not found: {p}")
    return np.load(str(p), allow_pickle=True)

# ─────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────

def embed_text(client: OpenAI, text: str) -> np.ndarray:
    out = client.embeddings.create(
        model=CONFIG["embedding_model"], input=text
    )
    v = np.array(out.data[0].embedding, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-9)

def faiss_search(index: faiss.Index, emb: np.ndarray, k: int):
    emb = emb.reshape(1, -1).astype("float32")
    dist, idx = index.search(emb, k)
    return dist[0], idx[0]

# ───────────────────────────────────────────────
# ID decoding for DAN corpus
# ───────────────────────────────────────────────

def decode_id(raw: Any) -> int:
    """
    DAN IDs look like:
        DAN::<report_id>::<section>::0000
    So we extract the final integer.
    """
    s = str(raw)
    if "::" in s:
        tail = s.rsplit("::", 1)[-1]
        try:
            return int(tail)
        except:
            pass

    # fallback
    try:
        return int(s)
    except:
        raise ValueError(f"Cannot decode DAN chunk id: {raw}")


def map_hits(hits, idmap, chunks):
    idxs, ids = [], []
    for h in hits:
        if h < 0:
            continue
        raw = idmap[h]
        try:
            ci = decode_id(raw)
            if 0 <= ci < len(chunks):
                idxs.append(ci)
                ids.append(str(raw))
        except Exception:
            continue
    return idxs, ids

# ─────────────────────────────────────────────────────────────
# Expanded query (DAN-specific)
# ─────────────────────────────────────────────────────────────

def build_expanded_query(client, original_query, seed_texts):
    if not seed_texts:
        return original_query

    snippets = []
    max_chars = 6000
    total = 0
    for t in seed_texts[:8]:
        t = (t or "")[:2000]
        if total + len(t) > max_chars:
            break
        snippets.append(t)
        total += len(t)
    context = "\n\n---\n\n".join(snippets)

    system_msg = (
        "You are a retrieval assistant working ONLY with the DAN accident/incident corpus.\n"
        "You will receive a query and DAN excerpts.\n"
        "Your job: expand the query into a more precise, technically detailed search query.\n"
        "Rules:\n"
        "- Use ONLY concepts present in the DAN excerpts.\n"
        "- Focus on: triggers, contributing factors, human factors, equipment issues, environment, procedures.\n"
        "- Do NOT answer the question.\n"
        "- Output exactly ONE expanded search query string."
    )

    user_msg = (
        f"Original query:\n{original_query}\n\n"
        f"DAN excerpts:\n{context}\n\n"
        "Produce one expanded technical search query grounded ONLY in these excerpts."
    )

    try:
        out = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=256,
        )
        expanded = out.choices[0].message.content.strip()
        if not expanded:
            return original_query
        return expanded
    except Exception:
        return original_query

# ─────────────────────────────────────────────────────────────
# Bibliography formatting
# ─────────────────────────────────────────────────────────────

def extract_biblio(docs):
    seen = set()
    lines = []

    for d in docs:
        meta = d.get("meta") or {}
        title = meta.get("title") or meta.get("header") or ""
        authors = meta.get("authors") or ""
        journal = meta.get("journal") or meta.get("source") or ""
        year = meta.get("year") or meta.get("dataset_year") or ""
        doi = meta.get("doi") or ""
        pdf_path = meta.get("pdf_path") or ""
        pdf_url = meta.get("pdf_url") or ""

        key = (title, authors, journal, year, doi, pdf_path)
        if key in seen:
            continue
        seen.add(key)

        parts = []
        if authors: parts.append(authors)
        if year: parts.append(f"({year})")
        if title: parts.append(title)
        if journal: parts.append(journal)
        if doi: parts.append(f"DOI: {doi}")

        loc = []
        if pdf_url: loc.append(f"URL: {pdf_url}")
        if pdf_path: loc.append(f"Local: {pdf_path}")
        if loc:
            parts.append("[" + "; ".join(loc) + "]")

        if parts:
            lines.append(" ".join(parts))

    return lines

# ─────────────────────────────────────────────────────────────
# Summary builder (DAN-oriented)
# ─────────────────────────────────────────────────────────────

def build_summary(client, query, docs):
    trimmed_docs = []
    total_chars = 0
    max_total = CONFIG["max_doc_chars"] * CONFIG["max_docs"]

    for d in docs:
        if total_chars >= max_total:
            break
        t = (d.get("text") or "")[: CONFIG["max_doc_chars"]]
        total_chars += len(t)

        meta = d.get("meta") or {}
        trimmed_docs.append({
            "text": t,
            "meta": {
                "title": meta.get("title") or meta.get("header") or "",
                "authors": meta.get("authors") or "",
                "journal": meta.get("journal") or meta.get("source") or "",
                "year": meta.get("year") or meta.get("dataset_year") or "",
                "doi": meta.get("doi") or "",
                "section": meta.get("section") or "",
                "paper_id": meta.get("paper_id") or "",
                "pdf_url": meta.get("pdf_url") or "",
                "pdf_path": meta.get("pdf_path") or "",
            },
        })

    biblio = extract_biblio(docs)

    system_msg = (
        "You are an expert in dive-incident analysis, working ONLY from the DAN corpus.\n"
        "You must produce a structured 2–3 page analytical report.\n"
        "STRICT RULES:\n"
        "• Use ONLY the provided docs (no external guidelines or speculation).\n"
        "• Focus on: contributing factors, human factors, equipment, physiology, environment.\n"
        "• Identify causal chains, recurrent patterns, and prevention lessons.\n"
        "• Required structure:\n"
        "  1. Incident question / context\n"
        "  2. Summary of DAN evidence\n"
        "  3. Event patterns and causal factors\n"
        "  4. Human factors and behavioural themes\n"
        "  5. Equipment-/environment-related factors\n"
        "  6. Limitations of the DAN evidence\n"
        "  7. Practical interpretation / preventive strategies\n"
        "• End with a section titled exactly 'Sources used:' followed by the bibliography_hint list."
    )

    payload = {
        "query": query,
        "docs": trimmed_docs,
        "bibliography_hint": biblio,
        "instructions": (
            "Use only the provided DAN docs. Attribute details to studies/reports "
            "using available metadata. Do not invent missing data."
        ),
    }

    out = client.chat.completions.create(
        model=CONFIG["model"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(payload)},
        ],
        max_completion_tokens=CONFIG["max_completion_tokens"],
    )

    return out.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────

def run_pipeline(query: str, force_refresh: bool = False) -> Dict[str, Any]:

    if not force_refresh:
        cached = cache_load(query)
        if cached:
            println("[CACHE] Loaded cached result")
            return cached

    client = new_client()

    println("[1] Loading DAN FAISS + chunks…")
    chunks = load_chunks()
    index = load_faiss_index()
    idmap = load_idmap()
    println(f"[INFO] Loaded {len(chunks)} DAN chunks")

    println("[2] Embedding original query…")
    q_emb = embed_text(client, query)

    println("[3] First FAISS search…")
    dist1, idx1 = faiss_search(index, q_emb, CONFIG["top_k"])
    d1_idx, d1_ids = map_hits(idx1, idmap, chunks)
    docs1 = [chunks[i] for i in d1_idx]
    seed = [d.get("text", "") for d in docs1]
    println(f"[3a] Hits: {len(d1_idx)}")

    println("[4] Expanded query…")
    exp = build_expanded_query(client, query, seed)
    println(f"[4a] Expanded: {exp}")

    println("[5] Embedding expanded query…")
    q2_emb = embed_text(client, exp)

    println("[6] Second FAISS search…")
    dist2, idx2 = faiss_search(index, q2_emb, CONFIG["top_k"])
    d2_idx, d2_ids = map_hits(idx2, idmap, chunks)

    combined = list(dict.fromkeys(list(zip(d1_idx, d1_ids)) + list(zip(d2_idx, d2_ids))))
    combined = combined[: CONFIG["max_docs"]]

    final_idx = [i for (i, _) in combined]
    final_ids = [cid for (_, cid) in combined]
    final_docs = [chunks[i] for i in final_idx]
    biblio = extract_biblio(final_docs) if final_docs else []

    if not final_docs:
        result = {
            "timestamp": utcnow(),
            "query": query,
            "expanded_query": exp,
            "docs_used": 0,
            "doc_ids": [],
            "summary": "No relevant documents found in the DAN corpus.",
            "bibliography": biblio,
        }
        cache_save(query, result)
        return result

    println("[7] Building summary…")
    summary = build_summary(client, query, final_docs)

    result = {
        "timestamp": utcnow(),
        "query": query,
        "expanded_query": exp,
        "docs_used": len(final_docs),
        "doc_ids": final_ids,
        "summary": summary,
        "bibliography": biblio,
    }
    if not force_refresh:
        cache_save(query, result)
    return result

# ─────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────

def format_report(result: Dict[str, Any]) -> str:
    cyan = "\033[96m"
    yellow = "\033[93m"
    red = "\033[91m"
    reset = "\033[0m"

    def wrap(text: str):
        lines = []
        for para in text.strip().split("\n\n"):
            lines.extend(textwrap.fill(para, width=96).splitlines())
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
        return lines

    lines = []
    lines.append(f"{cyan}DAN Query Result{reset}")
    lines.append(f"{yellow}Query:{reset} {result.get('query','')}")
    if result.get("expanded_query") != result.get("query"):
        lines.append(f"{yellow}Expanded:{reset} {result.get('expanded_query','')}")
    lines.append(f"{yellow}Docs used:{reset} {result.get('docs_used',0)}")
    if result.get("doc_ids"):
        lines.append(f"{yellow}Doc IDs:{reset} {', '.join(result['doc_ids'])}")
    lines.append("")
    lines.append(f"{yellow}Summary:{reset}")

    hl = re.compile(r"\b(risk|warning|hazard|fatal|panic|error|misread|equipment|failure)\b", re.IGNORECASE)
    summary = result.get("summary", "")
    if summary:
        for line in wrap(summary):
            if not line.strip():
                lines.append("")
                continue
            colored = hl.sub(lambda m: f"{red}{m.group(0)}{reset}", line)
            lines.append(colored)
    return "\n".join(lines)

def print_report(result):
    if RICH_AVAILABLE:
        _console.print("[bold cyan]DAN Query Result[/bold cyan]")
        _console.print(f"[yellow]Query:[/yellow] {result.get('query','')}")
        if result.get("expanded_query") != result.get("query"):
            _console.print(f"[yellow]Expanded:[/yellow] {result.get('expanded_query','')}")
        _console.print(f"[yellow]Docs used:[/yellow] {result.get('docs_used',0)}")
        if result.get("doc_ids"):
            _console.print(f"[yellow]Doc IDs:[/yellow] {', '.join(result['doc_ids'])}")
        _console.print("\n[bold cyan]Summary[/bold cyan]\n")
        summary = (result.get("summary") or "").strip()
        if summary:
            _console.print(Markdown(summary))
        else:
            _console.print("(no summary)")
    else:
        print(format_report(result))

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DAN RAG v1 query pipeline")
    parser.add_argument("-q", "--query", required=True, help="Incident / safety query for DAN corpus")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")

    args = parser.parse_args()
    out = run_pipeline(args.query)

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print_report(out)

if __name__ == "__main__":
    main()
