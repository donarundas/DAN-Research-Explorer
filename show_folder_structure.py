from pathlib import Path

def show_tree(root: Path, prefix: str = "", depth: int = 3):
    """Print a visual tree of folders up to a certain depth."""
    if depth < 0:
        return
    entries = sorted([p for p in root.iterdir() if p.is_dir()])
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name)
        extension = "    " if i == len(entries) - 1 else "│   "
        show_tree(entry, prefix + extension, depth - 1)

# === CONFIG ===
root_dir = Path.cwd()  # Change this path to your target folder
max_depth = 4                # Increase to show deeper levels
# ===============

print(f"\n📁 Folder tree for: {root_dir.resolve()}\n")
show_tree(root_dir, depth=max_depth)
