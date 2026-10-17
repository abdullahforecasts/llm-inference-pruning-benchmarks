"""
Baseline vs. pruned inference benchmark.

For each prompt length in --seq-lens, runs a manual prefill + decode loop with
no cache hooks (baseline) and with hooks.KVCachePruner attached (pruned), and
logs time-to-first-token, decode throughput, peak memory and final KV-cache
size for both. Each setting gets --warmup untimed runs followed by --repeats
timed runs; the CSV stores mean and std across the repeats.

Positions: once the cache is compacted, its length no longer equals the number
of tokens seen, so the decode loop passes the true absolute position of every
new token via `position_ids` instead of letting the model infer it from the
cache length.

Attention backend: the baseline uses SDPA. Importance-mode pruning needs the
attention weights, so the pruned run switches the model to eager attention;
sliding-window pruning stays on SDPA.

On CUDA, peak memory is torch.cuda.max_memory_allocated(). On CPU (laptop smoke
test) it falls back to process RSS, which is only good for catching crashes and
shape bugs, not for memory numbers.
"""

import argparse
import csv
import gc
import os
import resource
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from config import MODELS
from hooks import KVCachePruner, PruneConfig


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def build_prompt(tokenizer, seq_len, batch_size, device, vocab_size=None):
    if tokenizer is None:
        ids = torch.randint(0, vocab_size, (1, seq_len))
    else:
        text = "The history of machine learning research spans many decades. " * seq_len
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
    return ids.repeat(batch_size, 1).to(device)


def kv_cache_mb(cache):
    total = 0
    for layer in cache.layers:
        if getattr(layer, "keys", None) is not None:
            total += layer.keys.numel() * layer.keys.element_size()
            total += layer.values.numel() * layer.values.element_size()
    return total / 1e6


def set_attention(model, impl):
    if model.config._attn_implementation != impl:
        model.set_attn_implementation(impl)


def run_once(model, prompt_ids, max_new_tokens, device, prune_cfg=None):
    pruner = KVCachePruner(prune_cfg).attach(model) if prune_cfg is not None else None
    batch, prompt_len = prompt_ids.shape

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _sync(device)

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        t0 = time.perf_counter()
        positions = torch.arange(prompt_len, device=device).unsqueeze(0).expand(batch, -1)
        out = model(input_ids=prompt_ids, position_ids=positions, past_key_values=cache, use_cache=True)
        next_id = out.logits[:, -1:].argmax(-1)
        _sync(device)
        ttft = time.perf_counter() - t0

        cache = out.past_key_values
        pos = prompt_len
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            position_ids = torch.full((batch, 1), pos, device=device, dtype=torch.long)
            out = model(input_ids=next_id, position_ids=position_ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
            pos += 1
        _sync(device)
        decode_time = time.perf_counter() - t1

    generated = batch * (max_new_tokens - 1)
    metrics = {
        "ttft_s": ttft,
        "decode_time_s": decode_time,
        "throughput_tok_s": generated / decode_time if decode_time > 0 else float("nan"),
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else _peak_rss_mb(),
        "kv_cache_mb": kv_cache_mb(cache),
    }
    if pruner is not None:
        pruner.detach()
    del cache, out
    return metrics


def run_repeated(model, prompt_ids, args, prune_cfg):
    for _ in range(args.warmup):
        run_once(model, prompt_ids, args.max_new_tokens, args.device, prune_cfg)
    runs = [run_once(model, prompt_ids, args.max_new_tokens, args.device, prune_cfg)
            for _ in range(args.repeats)]
    agg = {}
    for key in runs[0]:
        values = [r[key] for r in runs]
        agg[key] = statistics.mean(values)
        agg[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return agg


def load_model(args):
    if args.model == "smoke":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=8192)
        model = Qwen2ForCausalLM(cfg).to(args.device).eval()
        return model, None, 1000
    model_id = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
    return model.to(args.device).eval(), tokenizer, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-0.5b", choices=list(MODELS) + ["smoke"])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--mode", default="importance", choices=["sliding_window", "importance"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/benchmark.csv")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny random-weight model, no download - for testing the pipeline")
    args = parser.parse_args()
    if args.smoke:
        args.model = "smoke"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model, tokenizer, vocab_size = load_model(args)
    prune_cfg = PruneConfig(window_size=args.window_size, sink_tokens=args.sink_tokens, mode=args.mode)
    pruned_attn = "eager" if args.mode == "importance" else "sdpa"

    rows = []
    for seq_len in args.seq_lens:
        prompt_ids = build_prompt(tokenizer, seq_len, args.batch_size, args.device, vocab_size)
        actual_len = prompt_ids.shape[1]

        set_attention(model, "sdpa")
        baseline = run_repeated(model, prompt_ids, args, prune_cfg=None)
        set_attention(model, pruned_attn)
        pruned = run_repeated(model, prompt_ids, args, prune_cfg=prune_cfg)
        set_attention(model, "sdpa")

        pruning_active = actual_len + args.max_new_tokens > args.window_size
        for label, metrics in [("baseline", baseline), ("pruned", pruned)]:
            rows.append({
                "model": args.model, "batch_size": args.batch_size, "seq_len": actual_len,
                "max_new_tokens": args.max_new_tokens, "run": label,
                "mode": args.mode if label == "pruned" else "none",
                "attn_impl": pruned_attn if label == "pruned" else "sdpa",
                "window_size": args.window_size if label == "pruned" else "",
                "pruning_active": pruning_active if label == "pruned" else "",
                "repeats": args.repeats, **metrics,
            })
        print(f"seq_len={actual_len} bs={args.batch_size} | "
              f"baseline tok/s={baseline['throughput_tok_s']:.1f} peak={baseline['peak_mem_mb']:.0f}MB "
              f"kv={baseline['kv_cache_mb']:.1f}MB | "
              f"pruned tok/s={pruned['throughput_tok_s']:.1f} peak={pruned['peak_mem_mb']:.0f}MB "
              f"kv={pruned['kv_cache_mb']:.1f}MB")

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
