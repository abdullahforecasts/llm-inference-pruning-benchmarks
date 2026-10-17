"""
Quality check: perplexity with and without KV-cache pruning.

Takes one long WikiText-2 (test split) token stream, prefills `context` tokens,
then scores the next `--eval-tokens` tokens one step at a time (teacher forcing),
exactly like decoding but with the true next token fed back in. The baseline
keeps the full cache; the pruned run uses chunked prefill plus the same hooks as
benchmark.py. Lower perplexity is better; the gap between the two runs is the
quality cost of pruning at that window size.

    python quality.py --model llama3.2-1b --contexts 2048 4096 --window-size 512 \\
        --eval-tokens 256 --out results/v2_t4/quality.csv
"""

import argparse
import csv
import math
import os

import torch
from transformers import DynamicCache

from benchmark import load_model, prefill
from config import MODELS
from hooks import KVCachePruner, PruneConfig


def load_tokens(tokenizer, n_tokens, vocab_size, seed=0):
    if tokenizer is None:
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, vocab_size, (1, n_tokens), generator=g)
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids
    if ids.shape[1] < n_tokens:
        raise ValueError(f"need {n_tokens} tokens, dataset has {ids.shape[1]}")
    return ids[:, :n_tokens]


def perplexity(model, ids, context, eval_tokens, device, prune_cfg=None, prefill_chunk=0):
    pruner = KVCachePruner(prune_cfg).attach(model) if prune_cfg is not None else None
    ids = ids.to(device)
    nll = 0.0
    with torch.no_grad():
        cache = DynamicCache(config=model.config)
        out = prefill(model, ids[:, :context], cache, prefill_chunk, device)
        cache = out.past_key_values
        logits = out.logits[:, -1]
        for t in range(context, context + eval_tokens):
            target = ids[:, t]
            nll += torch.nn.functional.cross_entropy(logits.float(), target).item()
            pos = torch.full((1, 1), t, device=device, dtype=torch.long)
            out = model(input_ids=target[:, None], position_ids=pos, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1]
    if pruner is not None:
        pruner.detach()
    return math.exp(nll / eval_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-0.5b", choices=list(MODELS) + ["smoke"])
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--eval-tokens", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--obs-window", type=int, default=32)
    parser.add_argument("--evict-slack", type=int, default=32)
    parser.add_argument("--prefill-chunk", type=int, default=512)
    parser.add_argument("--modes", nargs="+", default=["importance", "sliding_window"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/quality.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.model = "smoke"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model, tokenizer, vocab_size = load_model(args)
    ids = load_tokens(tokenizer, max(args.contexts) + args.eval_tokens, vocab_size)

    rows = []
    for context in args.contexts:
        base = perplexity(model, ids, context, args.eval_tokens, args.device)
        row = {"model": args.model, "context": context, "eval_tokens": args.eval_tokens,
               "window_size": args.window_size, "ppl_baseline": round(base, 4)}
        for mode in args.modes:
            cfg = PruneConfig(window_size=args.window_size, sink_tokens=args.sink_tokens, mode=mode,
                              obs_window=args.obs_window, evict_slack=args.evict_slack)
            ppl = perplexity(model, ids, context, args.eval_tokens, args.device, cfg, args.prefill_chunk)
            row[f"ppl_{mode}"] = round(ppl, 4)
        rows.append(row)
        print(row)

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
