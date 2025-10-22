import sys
import shutil
import nbformat

def ensure_widgets_state(meta):
    if isinstance(meta, dict) and 'widgets' in meta and isinstance(meta['widgets'], dict):
        if 'state' not in meta['widgets']:
            meta['widgets']['state'] = {}

def fix_notebook(path):
    # create backup
    shutil.copy2(path, path + ".bak")
    nb = nbformat.read(path, as_version=nbformat.NO_CONVERT)
    # root metadata
    ensure_widgets_state(nb.get('metadata', {}))
    # cell metadata
    for cell in nb.get('cells', []):
        ensure_widgets_state(cell.get('metadata', {}))
    nbformat.write(nb, path)
    print(f"Fixed notebook and saved backup as {path}.bak")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 fix_notebook_widgets.py <notebook.ipynb>")
        sys.exit(1)
    fix_notebook(sys.argv[1])