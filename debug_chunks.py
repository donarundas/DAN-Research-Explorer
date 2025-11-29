import zstandard as zstd, json, io

path = "data/ingestion_jsonl/dan_chunks_v1.jsonl.zst"
dctx = zstd.ZstdDecompressor()
with open(path, "rb") as f:
    with dctx.stream_reader(f) as r:
        text_stream = io.TextIOWrapper(r, encoding="utf-8")
        for i, line in enumerate(text_stream):
            if i >= 5:
                break
            o = json.loads(line)
            print("---- CHUNK", i)
            print("id:", o.get("id"))
            print("meta keys:", o.get("meta", {}).keys())
            print("section:", o.get("meta", {}).get("section"))
            print("paper_id:", o.get("meta", {}).get("paper_id"))
            print("text snippet:", (o.get("text","")[:200]).replace("\n"," "))
