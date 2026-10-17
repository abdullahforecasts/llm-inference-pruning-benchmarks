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
cache has been updated for the current forward pass and evicts entries once
the cache is over budget:

- **Prefill** is chunked (`--prefill-chunk`, default 512). After every chunk
  the cache is evicted back to `window_size`, so it never holds more than
  `window_size + chunk` tokens, and attention during prefill is computed over
  that bounded cache instead of the whole prompt. This is what lowers peak
  memory.
- **Decode** lets the cache grow to `window_size + evict_slack` and then evicts
  back to `window_size` in one go, so the top-k and the per-layer
  `index_select` run once every `evict_slack` steps instead of on every step.

What gets kept:

- **sliding_window**: a few "sink" tokens (the first tokens act as an
  attention sink; dropping them destabilizes generation, see StreamingLLM)
  plus the most recent tokens.
- **importance**: the sink tokens, the most recent `obs_window` tokens, and the
  older tokens that the last `obs_window` queries attend to most (an
  observation-window score in the style of SnapKV, max-pooled over neighbouring
  positions). The queries are captured with a hook on `q_proj` and rotated with
  the layer's own RoPE, so both runs stay on SDPA attention; no eager attention
  or full attention matrix is needed.

Both operate directly on `cache.layers[layer_idx].keys` / `.values`
(transformers' `DynamicCache` API), so they work on any model whose attention
module exposes `layer_idx`, `q_proj` and follows the same
`[batch, kv_heads, seq, head_dim]` layout: Qwen2/2.5 and Llama-3.x as-is.

Positions: once the cache is compacted its length no longer equals the
number of tokens seen, so `benchmark.py` passes the true absolute position of
every token (`position_ids`) during prefill and decode. Retained tokens keep
the rotary positions they were cached with, so relative distances stay
correct. The causal mask is built from the compacted cache length, so chunked
prefill on a pruned cache stays causal.

Baseline: the whole prompt in one forward pass and a full cache, SDPA
attention. Both runs compute logits only for the last prompt token. The
`--baseline-prefill-chunk` option makes the baseline chunk its prefill too, to
separate what chunking alone buys from what the bounded cache buys.

## Files

- `hooks.py`: the forward hooks and `KVCachePruner`.
- `benchmark.py`: chunked prefill + decode loop, run with and without the
  hooks across context lengths and batch sizes. Logs TTFT, decode throughput,
  peak memory, final KV-cache size and eviction count (mean and std over
  repeats).
- `quality.py`: WikiText-2 perplexity with the full cache vs. the pruned cache,
  to show what pruning costs in output quality.
- `visualize.py`: per-model figures (with a scikit-learn linear fit of memory
  scaling) and a `summary.csv` of memory reduction and throughput speedup.
- `config.py`: model registry and default sweep settings.
- `notebooks/colab_sweep.ipynb`: the GPU sweep (T4 on Colab free tier). CPU
  can't give real VRAM numbers, so this is where the headline results come
  from.

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
python benchmark.py --model llama3.2-1b --seq-lens 2048 4096 8192 \
    --batch-size 8 --max-new-tokens 128 --window-size 512 --mode importance \
    --prefill-chunk 512 --warmup 1 --repeats 3 --device cuda --out results/v2_t4/benchmark.csv
python quality.py --model llama3.2-1b --contexts 2048 4096 --window-size 512 \
    --eval-tokens 256 --out results/v2_t4/quality.csv
python visualize.py --csv results/v2_t4/benchmark.csv --out-dir results/v2_t4/figures
```

Each setting gets `--warmup` untimed runs and `--repeats` timed runs; the CSV
stores mean and std for TTFT, decode time, throughput, peak memory and final
KV-cache size.
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
