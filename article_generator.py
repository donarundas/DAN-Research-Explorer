# /Users/donarundas/Projects/DAN/article_generator.py
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are a scientific summarizer for diving and hyperbaric research.
Write a clear, structured Markdown article using provided retrieved snippets.
Each factual statement must cite its source using the pattern (DAN <year>, p.<page>).
Include section headers (##) and bullet points where appropriate.
If tables or figures are referenced, label them as 'Table X' or 'Figure Y' and note the source.
"""

def build_prompt(query, results):
    """Builds combined context for GPT-5 with metadata and content snippets."""
    context_blocks = []
    for r in results[:10]:
        source = r.get("source", "Unknown Source")
        page = r.get("page", "?")
        snippet = r.get("preview", "").strip()
        context_blocks.append(f"### {source} (Page {page})\n{snippet}")
    context = "\n\n".join(context_blocks)

    return f"""
User query: {query}

You will synthesize information into a structured article that:
- Summarizes the findings from these DAN publications
- Mentions tables and figures by label if relevant
- Includes inline citations (DAN YEAR, p.PAGE)
- Ends with a References section listing each source

Retrieved context:
{context}
"""

def generate_article(query, results):
    """Streams GPT-5 output live to console and returns Markdown."""
    prompt = build_prompt(query, results)
    print(f"\n🧠 Generating GPT-5 article for query: {query}\n")

    stream = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        max_completion_tokens=2000,
    )

    collected = []
    for event in stream:
        if token := event.choices[0].delta.content:
            print(token, end="", flush=True)
            collected.append(token)

    article = "".join(collected)
    print("\n\n✅ Article generation complete.")
    print(f"🕒 {datetime.utcnow().isoformat()}Z\n")
    return article
