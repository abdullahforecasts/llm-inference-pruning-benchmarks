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
repeats (mean and std go into the CSV). The pruned run prefills in 512-token
chunks and evicts after every chunk; the baseline prefills the whole prompt at
once and keeps the full cache.

Main sweep: importance-based eviction, window 512.

- batch 1: the baseline prefills the whole prompt in one pass. On a T4 that
  runs out of memory for long prompts; the run is recorded as `oom` in the CSV
  and the sweep continues.
- batch 8: the baseline also prefills in 512-token chunks
  (`--baseline-prefill-chunk 512`) but keeps the full cache, so both runs fit
  and the comparison isolates the effect of the bounded cache.

Results are appended to the CSV after every prompt length, and `--resume`
skips settings that are already in it, so an interrupted cell can simply be
run again.
"""),
    ("code", """\
RUN = "v2_t4"
OUT = f"results/{RUN}/benchmark.csv"
COMMON = "--max-new-tokens 128 --window-size 512 --sink-tokens 4 --prefill-chunk 512 --warmup 1 --repeats 3 --device cuda --resume"
"""),
    ("code", """\
!python benchmark.py --model qwen2.5-0.5b --batch-size 1 --seq-lens 1024 2048 4096 8192 --mode importance {COMMON} --out {OUT}
"""),
    ("code", """\
!python benchmark.py --model llama3.2-1b --batch-size 1 --seq-lens 1024 2048 4096 8192 --mode importance {COMMON} --out {OUT}
"""),
    ("code", """\
!python benchmark.py --model qwen2.5-0.5b --batch-size 8 --seq-lens 2048 4096 8192 --mode importance --baseline-prefill-chunk 512 {COMMON} --out {OUT}
"""),
    ("code", """\
!python benchmark.py --model llama3.2-1b --batch-size 8 --seq-lens 2048 4096 8192 --mode importance --baseline-prefill-chunk 512 {COMMON} --out {OUT}
"""),
    ("markdown", """\
### Ablations (llama3.2-1b)

- batch 8, recency-only eviction (`--mode sliding_window`) vs. importance
- batch 1, baseline that also prefills in 512-token chunks, to separate what
  chunked prefill alone saves from what the bounded cache saves
"""),
    ("code", """\
!python benchmark.py --model llama3.2-1b --batch-size 8 --seq-lens 2048 4096 8192 --mode sliding_window --baseline-prefill-chunk 512 {COMMON} --out results/{RUN}/ablation_sliding_window.csv
!python benchmark.py --model llama3.2-1b --batch-size 1 --seq-lens 1024 2048 4096 8192 --mode importance --baseline-prefill-chunk 512 {COMMON} --out results/{RUN}/ablation_chunked_baseline.csv
"""),
    ("markdown", """\
### Quality: WikiText-2 perplexity, full cache vs. pruned cache
"""),
    ("code", """\
!python quality.py --model qwen2.5-0.5b --contexts 2048 4096 --window-size 512 --eval-tokens 256 --out results/{RUN}/quality.csv
!python quality.py --model llama3.2-1b --contexts 2048 4096 --window-size 512 --eval-tokens 256 --out results/{RUN}/quality.csv
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

The CSVs are the only files needed locally: figures are rebuilt from it with
`python visualize.py --csv results/<RUN>/benchmark.csv --out-dir results/<RUN>/figures`.
Copy each printed CSV into the file of the same name under `results/<RUN>/` in the local repo.
"""),
    ("code", """\
!cat {OUT}
"""),
    ("code", """\
!cat results/{RUN}/ablation_sliding_window.csv
"""),
    ("code", """\
!cat results/{RUN}/ablation_chunked_baseline.csv
"""),
    ("code", """\
!cat results/{RUN}/quality.csv
"""),
]
