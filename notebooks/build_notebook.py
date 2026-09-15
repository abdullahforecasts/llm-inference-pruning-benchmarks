"""Builds colab_sweep.ipynb from colab_sweep.py so the cell source stays diffable."""

import json
import os
import uuid

from colab_sweep import CELLS

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "colab_sweep.ipynb")


def make_cell(cell_type, source):
    lines = source.splitlines(keepends=True)
    cell = {"cell_type": cell_type, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main():
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [make_cell(t, s) for t, s in CELLS],
    }
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
