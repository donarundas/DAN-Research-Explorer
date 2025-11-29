# /Users/donarundas/Projects/DAN/article_generator.py
from openai import OpenAI
from datetime import datetime
from pathlib import Path  # ✅ this line fixes the NameError
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are a scientific summarizer for diving and hyperbaric research.
Write a clear, structured Markdown article using the provided snippets.
Guidelines:
- Use section headers (##) and short paragraphs.
- Where possible, cite sources in-text like (DAN 2005) or (DAN 2019).
- If a page number is provided in the source label, you may include it (e.g. p.32),
  but do NOT insist on page numbers if they are not available.
- At the end, add a short "References" section summarizing which DAN documents were used.
"""

def build_prompt(query, results):
    """
    Construct a prompt with meaningful snippets and source labels.
    """
    blocks = []
    for r in results:
        src = r.get("source", "Unknown source")
        page = r.get("page", None)
        snippet = r.get("preview", "").strip()

        # Derive publication folder name for context
        pub_name = Path(src).parent.name if "/" in src or "\\" in src else src
        if page:
            label = f"{pub_name} (p.{page})"
        else:
            label = pub_name

        block = f"### {label}\n{snippet}"
        blocks.append(block)

    context = "\n\n".join(blocks)

    return f"""
User query:
{query}

You are given context snippets from DAN publications:

{context}

Write a cohesive, well-structured Markdown article that:
- Answers the query in a scientific, neutral tone.
- Integrates insights from multiple sources.
- Uses inline citations like (DAN 2005) or (DAN 2019) when you clearly infer
  which publication the information came from.
- Ends with a "References" section listing each DAN publication that appears
  in the context (you may reuse the folder-style names as titles).
"""

def generate_article(query, results):
    """
    Streams GPT-5 output to console and returns full Markdown.
    """
    prompt = build_prompt(query, results)

    print("──────────────────────────────────────────────")
    print(f"🧠 Starting GPT-5 Article Generation for Query: {query}")
    print("──────────────────────────────────────────────\n")

    stream = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        max_completion_tokens=2000,
    )

    chunks = []
    for event in stream:
        if token := event.choices[0].delta.content:
            print(token, end="", flush=True)
            chunks.append(token)

    article = "".join(chunks)

    print("\n\n──────────────────────────────────────────────")
    print("✅ GPT-5 Article Generation Complete")
    print(f"🕒 {datetime.utcnow().isoformat()}Z")
    print("──────────────────────────────────────────────\n")

    return article
