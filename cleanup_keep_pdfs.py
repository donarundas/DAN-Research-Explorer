from pathlib import Path
import shutil

ROOT = Path.cwd()
delete_dirs = ["extracted", "extracted_tables", "extracted_figures", "extracted_text"]
delete_files = ["metadata.json"]

count_dirs = 0
count_files = 0

for path in ROOT.rglob("*"):
    # remove any unwanted directories
    if path.is_dir() and any(key in path.name.lower() for key in delete_dirs):
        shutil.rmtree(path, ignore_errors=True)
        count_dirs += 1
    # remove unwanted files
    elif path.is_file() and path.name.lower() in delete_files:
        path.unlink()
        count_files += 1

print(f"✅ Removed {count_dirs} extracted folders and {count_files} metadata.json files under {ROOT}")
