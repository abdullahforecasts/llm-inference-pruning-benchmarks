"""
Cell source for notebooks/colab_sweep.ipynb, kept here as a plain .py so it's
easy to diff/edit. Regenerate the notebook with:

    python notebooks/build_notebook.py
"""

CELLS = [
    ("markdown", """\
# KV-cache pruning benchmark sweep

Run on a Colab GPU runtime (Runtime > Change runtime type > T4 GPU).
Clones the repo, installs deps, runs the baseline-vs-pruned sweep for each
model/context length, then builds the figures.
"""),
    ("code", """\
!nvidia-smi
"""),
    ("code", """\
!git clone https://github.com/<your-username>/llm-inference-pruning-benchmarks.git
%cd llm-inference-pruning-benchmarks
!pip install -q -r requirements.txt
"""),
    ("markdown", """\
## Run the sweep

Repeat for each model key in `config.MODELS`. `--mode importance` uses the
attention-score-based eviction; swap to `--mode sliding_window` for the
recency-only baseline ablation.
"""),
    ("code", """\
!python benchmark.py \\
    --model qwen2.5-0.5b \\
    --seq-lens 256 512 1024 2048 \\
    --max-new-tokens 128 \\
    --window-size 512 \\
    --sink-tokens 4 \\
    --mode importance \\
    --device cuda \\
    --out results/benchmark.csv
"""),
    ("code", """\
!python benchmark.py \\
    --model llama3.2-1b \\
    --seq-lens 256 512 1024 2048 \\
    --max-new-tokens 128 \\
    --window-size 512 \\
    --sink-tokens 4 \\
    --mode importance \\
    --device cuda \\
    --out results/benchmark.csv
"""),
    ("markdown", "## Build figures"),
    ("code", """\
!python visualize.py --csv results/benchmark.csv --out-dir results/figures
"""),
    ("code", """\
from IPython.display import Image, display
display(Image("results/figures/memory_scaling.png"))
display(Image("results/figures/throughput_comparison.png"))
"""),
    ("code", """\
import pandas as pd
pd.read_csv("results/benchmark.csv")
"""),
]
