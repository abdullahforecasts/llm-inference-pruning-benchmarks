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

Positions: once the cache is compacted its length no longer equals the
number of tokens seen, so `benchmark.py` passes the true absolute position of
every new token (`position_ids`) during decode. Retained tokens keep the
rotary positions they were cached with, so relative distances stay correct.

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

Real sweep, on a GPU (Colab T4, see `notebooks/colab_sweep.ipynb`):

```bash
python benchmark.py --model qwen2.5-0.5b --seq-lens 256 512 1024 2048 \
    --batch-size 1 --max-new-tokens 128 --window-size 512 --mode importance \
    --warmup 1 --repeats 3 --device cuda --out results/v1_t4/benchmark.csv
python visualize.py --csv results/v1_t4/benchmark.csv --out-dir results/v1_t4/figures
```

Each setting gets `--warmup` untimed runs and `--repeats` timed runs; the CSV
stores mean and std for TTFT, decode time, throughput, peak memory and final
KV-cache size. The baseline runs on SDPA attention; importance-mode pruning
switches to eager attention because it needs the attention weights.
`visualize.py` writes per-model figures and a `summary.csv` that only counts
settings where pruning actually evicted tokens.

`--model` accepts any key from `config.MODELS` (Qwen2.5 and Llama-3.2
variants); add more there as needed. `--mode sliding_window` runs the
recency-only ablation for comparison against the importance-based eviction.

## Results

### v0: initial T4 sweep (Colab, Sep 2026)

| model | peak VRAM reduction | decode throughput (pruned / baseline) |
|---|---|---|
| qwen2.5-0.5b | 0.6% | 0.82x |
| llama3.2-1b | 0.5% | 0.82x |

Batch 1, prompts of 256-2048 tokens, 128 new tokens, window 512, 4 sink tokens,
importance mode, fp16 on a T4. At this scale the KV cache is a small share of
peak memory and pruning applies after prefill; v1 adds chunked prefill with
eviction, amortized eviction, and longer-context / larger-batch runs.

Data, figures and Colab screenshots: [`results/v0_t4_initial/`](results/v0_t4_initial/).

The last repo got lost, so this is a new one and i am working on this to regain all previous results.
