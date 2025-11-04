import os
import pandas as pd
from pathlib import Path

# ======= CONFIG =======
ROOT = Path.cwd()

REPORT_FILE = "dan_sanity_report.csv"
CATEGORY_DEPTH = 2   # e.g. DAN/<Category>/<Files>
# ======================

records = []
for path, _, files in os.walk(ROOT):
    for f in files:
        file_path = Path(path) / f
        try:
            size_kb = file_path.stat().st_size / 1024
        except Exception:
            size_kb = 0
        rel_folder = file_path.relative_to(ROOT).parent
        category = "/".join(rel_folder.parts[:CATEGORY_DEPTH]) if rel_folder.parts else "root"
        records.append({
            "category": category,
            "folder": str(rel_folder),
            "file": f,
            "ext": file_path.suffix.lower(),
            "size_kb": round(size_kb, 2)
        })

df = pd.DataFrame(records)
if df.empty:
    print("❌ No files found under", ROOT)
    raise SystemExit

# === Summary by extension ===
summary_ext = (
    df.groupby("ext")
      .agg(files=("file", "count"), total_size_kb=("size_kb", "sum"))
      .sort_values("files", ascending=False)
)

# === Summary by category ===
summary_cat = (
    df.groupby("category")
      .agg(total_files=("file", "count"),
           total_size_mb=("size_kb", lambda x: round(x.sum()/1024, 2)))
      .sort_values("total_files", ascending=False)
)

# === Detect suspicious files ===
missing = df[df["size_kb"] < 1]

# === Save report ===
df.to_csv(REPORT_FILE, index=False)

# === Console output ===
print("\n📦 SUMMARY BY FILE TYPE")
print(summary_ext)
print("\n📁 SUMMARY BY CATEGORY")
print(summary_cat)

if not missing.empty:
    print("\n⚠️  EMPTY / CORRUPT FILES")
    print(missing[["folder", "file", "size_kb"]])
else:
    print("\n✅ No empty or near-empty files detected.")

print(f"\n🗒️  Full detailed report saved as {REPORT_FILE}")
