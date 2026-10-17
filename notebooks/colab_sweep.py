"""
Cell source for notebooks/colab_sweep.ipynb, kept here as a plain .py so it's
easy to diff/edit. Regenerate the notebook with:

    python notebooks/build_notebook.py
"""

CELLS = [
    ("markdown", """\
# KV-cache pruning benchmark sweep

Runs on a Colab T4, either in the browser (Runtime > Change runtime type > T4 GPU)
or from VS Code with the Google Colab extension (Select Kernel > Colab >
New Colab Server > GPU > T4 > Python 3).

Clones (or updates) the repo on the Colab machine, installs deps, runs the
baseline-vs-pruned sweep for each model, then builds the figures and summary.
"""),
    ("code", """\
!nvidia-smi
"""),
    ("code", """\
import os
REPO = "/content/llm-inference-pruning-benchmarks"
if not os.path.isdir(REPO):
    !git clone https://github.com/abdullahforecasts/llm-inference-pruning-benchmarks.git {REPO}
%cd {REPO}
!git pull
!git log --oneline -1
!pip install -q -r requirements.txt
"""),
    ("markdown", """\
Llama models are gated on Hugging Face: run the login cell once per server.
"""),
    ("code", """\
from huggingface_hub import login
login()
"""),
    ("markdown", """\
## Run the sweep

`RUN` names the results folder. Each setting gets 1 warm-up run and 3 timed
repeats (mean and std go into the CSV). `--mode importance` uses
attention-score-based eviction; `--mode sliding_window` is the recency-only
ablation.
"""),
    ("code", """\
RUN = "v1_t4"
OUT = f"results/{RUN}/benchmark.csv"
COMMON = "--seq-lens 256 512 1024 2048 --max-new-tokens 128 --window-size 512 --sink-tokens 4 --warmup 1 --repeats 3 --device cuda"
"""),
    ("code", """\
!python benchmark.py --model qwen2.5-0.5b --batch-size 1 --mode importance {COMMON} --out {OUT}
"""),
    ("code", """\
!python benchmark.py --model llama3.2-1b --batch-size 1 --mode importance {COMMON} --out {OUT}
"""),
    ("markdown", "## Build figures and summary"),
    ("code", """\
!python visualize.py --csv {OUT} --out-dir results/{RUN}/figures
"""),
    ("code", """\
from IPython.display import Image, display
display(Image(f"results/{RUN}/figures/memory_scaling.png"))
display(Image(f"results/{RUN}/figures/throughput_comparison.png"))
"""),
    ("markdown", """\
## Bring the results back to the local repo

The CSV is the only file needed locally: figures are rebuilt from it with
`python visualize.py --csv results/<RUN>/benchmark.csv --out-dir results/<RUN>/figures`.
Copy the printed CSV into `results/<RUN>/benchmark.csv` in the local repo.
"""),
    ("code", """\
!cat {OUT}
"""),
]
