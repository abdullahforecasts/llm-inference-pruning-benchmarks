"""
Baseline vs. pruned inference benchmark.

For each prompt length in --seq-lens, runs a manual prefill + decode loop twice
(once with no cache hooks, once with hooks.KVCachePruner attached) and logs peak
memory, time-to-first-token, and decode throughput for both. Results are appended
to a CSV that visualize.py consumes.

On CUDA, peak memory is torch.cuda.max_memory_allocated(). On CPU (e.g. a laptop
smoke test before the real run on a Colab GPU) it falls back to process RSS via
resource.getrusage, which is a much looser proxy - fine for sanity-checking the
hooks, not for the VRAM numbers themselves.
"""

import argparse
import csv
import os
import resource
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from config import MODELS, RunConfig
from hooks import KVCachePruner, PruneConfig


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build_prompt(tokenizer, seq_len, device, smoke=False, vocab_size=None):
    if smoke:
        ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        return ids
    text = "The history of machine learning research spans many decades. " * seq_len
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
    return ids.to(device)


def run_once(model, prompt_ids, max_new_tokens, device, prune_cfg=None):
    pruner = None
    if prune_cfg is not None:
        pruner = KVCachePruner(prune_cfg).attach(model)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
        if device == "cuda":
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        cache = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)

        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_id, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
        if device == "cuda":
            torch.cuda.synchronize()
        decode_time = time.perf_counter() - t1

    throughput = (max_new_tokens - 1) / decode_time if decode_time > 0 else float("nan")
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else _peak_rss_mb()

    if pruner is not None:
        pruner.detach()

    return {"ttft_s": ttft, "decode_time_s": decode_time,
            "throughput_tok_s": throughput, "peak_mem_mb": peak_mem_mb}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-0.5b", choices=list(MODELS) + ["smoke"])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--mode", default="importance", choices=["sliding_window", "importance"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/benchmark.csv")
    parser.add_argument("--smoke", action="store_true",
                         help="tiny random-weight model, no download - for testing the pipeline")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    attn_impl = "eager" if args.mode == "importance" else "sdpa"

    if args.smoke or args.model == "smoke":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                           max_position_embeddings=8192, attn_implementation=attn_impl)
        model = Qwen2ForCausalLM(cfg).to(args.device).eval()
        tokenizer = None
        vocab_size = 1000
    else:
        model_id = MODELS[args.model]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            attn_implementation=attn_impl,
        ).to(args.device).eval()
        vocab_size = None

    prune_cfg = PruneConfig(window_size=args.window_size, sink_tokens=args.sink_tokens, mode=args.mode)

    rows = []
    for seq_len in args.seq_lens:
        prompt_ids = build_prompt(tokenizer, seq_len, args.device,
                                   smoke=args.smoke or args.model == "smoke", vocab_size=vocab_size)

        baseline = run_once(model, prompt_ids, args.max_new_tokens, args.device, prune_cfg=None)
        pruned = run_once(model, prompt_ids, args.max_new_tokens, args.device, prune_cfg=prune_cfg)

        for label, metrics in [("baseline", baseline), ("pruned", pruned)]:
            rows.append({
                "model": args.model, "seq_len": seq_len, "run": label,
                "mode": args.mode if label == "pruned" else "none",
                "window_size": args.window_size if label == "pruned" else seq_len,
                **metrics,
            })
        print(f"seq_len={seq_len}  baseline={baseline}  pruned={pruned}")

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
