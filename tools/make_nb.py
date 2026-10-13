"""Convert a percent-format .py file ("# %%" code cells, "# %% [markdown]" cells) to .ipynb.

Usage: python3 tools/make_nb.py kaggle/np-01-pilot/np-01-pilot.py
Writes the .ipynb next to the source file. Stdlib only.
"""
import json
import sys
from pathlib import Path


def parse(src: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        text = "\n".join(lines)
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in src.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main(path: str):
    p = Path(path)
    nb = {
        "cells": parse(p.read_text()),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = p.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main(sys.argv[1])
