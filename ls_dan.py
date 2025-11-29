#!/usr/bin/env python3
"""
List DAN folder structure, ignoring all Git and repo noise.
Prints only meaningful files (PDFs, docs, text, CSV, images).
"""

import os
from pathlib import Path

# CHANGE THIS if your folder has a different name
ROOT = Path("")

# Directories to ignore
IGNORE_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules",
    "venv",
    "env",
    "build",
    "dist",
}

# Files to ignore (extensions or exact names)
IGNORE_EXT = {
    ".gitignore",
    ".gitattributes",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".lock",
    ".cfg",
    ".ini",
    ".json",
}

# Allowed file extensions (adjust as needed)
ALLOWED_EXT = {
    ".pdf",
    ".csv",
    ".xls",
    ".xlsx",
    ".txt",
    ".doc",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
}

def should_ignore_dir(dirname: str) -> bool:
    return dirname.lower() in IGNORE_DIRS

def should_ignore_file(filename: str) -> bool:
    name = filename.lower()
    ext = Path(filename).suffix.lower()

    # ignore repo noise
    if name in IGNORE_EXT:
        return True

    # show only meaningful DAN files
    if ext not in ALLOWED_EXT:
        return True

    return False

def list_files(root: Path):
    for path, dirs, files in os.walk(root):

        # filter dirs
        dirs[:] = [d for d in dirs if not should_ignore_dir(d)]

        rel = Path(path).relative_to(root)
        folder_display = "/" if rel == Path(".") else str(rel)
        print(f"\n[DIR] {folder_display}")

        # filter files
        visible_files = [f for f in files if not should_ignore_file(f)]
        if not visible_files:
            print("  (no relevant files)")
            continue

        for f in visible_files:
            fp = Path(path) / f
            size_kb = fp.stat().st_size / 1024
            print(f"  - {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    if not ROOT.exists():
        print(f"Folder not found: {ROOT.resolve()}")
    else:
        list_files(ROOT)
