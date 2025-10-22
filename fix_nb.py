import sys
import shutil
import nbformat
from pathlib import Path
import json

def remove_widgets(meta):
    if isinstance(meta, dict) and 'widgets' in meta:
        del meta['widgets']

def fix_notebook(path):
    path = Path(path)
    # ensure data directory exists and save backup there
    backup_dir = Path("data")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / (path.name + ".bak")
    shutil.copy2(path, backup_path)

    with open(path, 'r') as f:
        notebook = json.load(f)

    if 'widgets' in notebook.get('metadata', {}):
        del notebook['metadata']['widgets']
    
    for cell in notebook.get('cells', []):
        if 'widgets' in cell.get('metadata', {}):
            del cell['metadata']['widgets']

    with open(path, 'w') as f:
        json.dump(notebook, f, indent=2)

    print(f"Removed metadata.widgets and saved backup as {backup_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 fix_nb.py <notebook.ipynb>")
        sys.exit(1)
    fix_notebook(sys.argv[1])