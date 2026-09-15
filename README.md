# llm-inference-pruning-benchmarks

Dynamic KV-cache truncation and attention-guided token pruning for open-source
Transformer LMs (Qwen, Llama), implemented as PyTorch forward hooks, with a
benchmarking harness to measure the memory/latency trade-off it buys.

## Why

KV-cache size grows linearly with context length, and it's usually the thing
that runs a GPU out of memory long before compute does. This repo hooks into
the attention layers of a stock HuggingFace causal LM and keeps the cache
bounded during generation, without modifying model weights or requiring
retraining.

## How it works

`hooks.py` attaches a `forward_hook` (via `register_forward_hook(..., with_kwargs=True)`)
to every `self_attn` submodule. Each hook runs right after that layer's KV
cache has been updated for the current step, and either:

- **sliding_window**: keeps a few "sink" tokens (the first tokens in the
  sequence act as an attention sink; dropping them destabilizes generation,
  see StreamingLLM) plus the most recently seen tokens, up to `window_size`.
- **importance**: same budget, but instead of pure recency it keeps the
  sink tokens plus whichever cached tokens have the highest attention weight
  (an EMA over generation steps). Needs `attn_implementation="eager"` so the
  attention module actually returns its weights.

Both operate directly on `cache.layers[layer_idx].keys` / `.values`
(transformers' `DynamicCache` API), so they work on any model whose attention
module exposes `layer_idx` and follows the same `[batch, heads, seq, head_dim]`
layout, which is Qwen2/2.5 and Llama-3.x as-is, and most other decoder-only
models with minimal changes.

One simplification worth calling out: because the cache is compacted, future
position ids are computed against the *compacted* cache length rather than
the true token count. This is the same trick sliding-window attention
implementations use: it keeps generation stable, but it means retained
tokens are attended to at a shifted relative position, which is part of the
quality/memory trade-off being measured here, not a bug to route around.

## Files

- `hooks.py`: the forward hooks and `KVCachePruner`.
- `benchmark.py`: manual prefill + decode loop, run once with hooks attached
  and once without, across a list of context lengths. Logs TTFT, decode
  throughput, and peak memory to `results/benchmark.csv`.
- `visualize.py`: loads the CSV, fits a linear memory-scaling trend per run
  with scikit-learn, and renders the comparison figures into
  `results/figures/`.
- `config.py`: model registry and default sweep settings.
- `notebooks/colab_sweep.ipynb`: the actual GPU sweep (T4 on Colab free
  tier). CPU can't give real VRAM numbers, so this is where the headline
  results come from.

## Running it

### Local setup (laptop, CPU)

```bash
bash setup.sh
source .venv/bin/activate
```

`setup.sh` creates `.venv`, installs the CPU-only torch build (the default
PyPI wheel pulls in the full CUDA toolkit, several GB, even with no GPU
present), then installs the rest of `requirements.txt`. It's just these
three commands, run in order, if you'd rather do it by hand:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On a GPU box (or Colab, see below) use `pip install torch` instead of the
`--index-url` line, so you get the CUDA build.

Sanity-check the hooks and pipeline without downloading any model weights:

```bash
python benchmark.py --smoke --seq-lens 128 256 --max-new-tokens 32 \
    --window-size 64 --mode importance --device cpu
python visualize.py
```

Verified working end to end on a CPU-only laptop (`setup.sh` -> smoke test ->
`visualize.py`, no errors, `results/benchmark.csv` and `results/figures/`
both populated). The numbers that run prints are not meaningful though: it's
a tiny random-weight model, and peak memory on CPU falls back to process RSS
(see `benchmark.py`'s `_peak_rss_mb`), which is a noisy, process-wide
high-water mark, not an isolated per-run measurement. It's there to catch
crashes and shape bugs before spending GPU time, not to show a trend.

Real sweep, on a GPU:

```bash
python benchmark.py --model qwen2.5-0.5b --seq-lens 256 512 1024 2048 \
    --max-new-tokens 128 --window-size 512 --mode importance --device cuda
python visualize.py
```

`--model` accepts any key from `config.MODELS` (Qwen2.5 and Llama-3.2
variants); add more there as needed. `--mode sliding_window` runs the
recency-only ablation for comparison against the importance-based eviction.

## Results

Headline numbers (peak VRAM reduction, throughput speedup) get filled in
here once the Colab sweep has run. See `results/benchmark.csv` and
`results/figures/` after running `notebooks/colab_sweep.ipynb`.
