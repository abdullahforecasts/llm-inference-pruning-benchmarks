"""
Forward hooks that turn a stock HF causal LM into one with a bounded KV cache.

Two mechanisms, both operating directly on the model's `Cache` object after each
attention layer's forward pass (i.e. after `past_key_values.update()` has already
grown it for the current step):

  - sliding window: keep a handful of "sink" tokens (StreamingLLM-style) plus the
    most recent `window_size - sink_tokens` tokens, drop everything else.
  - importance pruning: same budget, but instead of pure recency we rank cached
    tokens by an EMA of the attention weight they receive and keep the sink tokens
    plus the highest-scoring survivors. Requires attn_implementation="eager" so
    attn_weights actually get computed and returned by the attention module.

Tested against Qwen2/Qwen2.5 and Llama-3 decoder layers, which both expose a
`layer_idx` attribute on `self_attn` and store cache tensors as
[batch, num_heads, seq_len, head_dim] via `cache.layers[layer_idx].keys/.values`.
Should work unmodified on any model that follows the same convention.
"""

from dataclasses import dataclass

import torch


@dataclass
class PruneConfig:
    window_size: int = 512      # hard cap on cached tokens per layer
    sink_tokens: int = 4        # always-kept prefix (attention sink)
    mode: str = "sliding_window"  # "sliding_window" | "importance"
    ema_alpha: float = 0.5      # importance score smoothing, only used in "importance" mode


def _find_attention_modules(model):
    found = []
    for name, module in model.named_modules():
        if name.endswith("self_attn") and hasattr(module, "layer_idx"):
            found.append((name, module))
    return found


class KVCachePruner:
    """Attaches/detaches truncation hooks and owns the per-layer importance buffers."""

    def __init__(self, config: PruneConfig):
        self.config = config
        self._handles = []
        self._importance = {}  # layer_idx -> 1D tensor aligned to that layer's cache

    def attach(self, model):
        self.detach()
        self._importance.clear()
        for _, module in _find_attention_modules(model):
            handle = module.register_forward_hook(self._make_hook(module.layer_idx), with_kwargs=True)
            self._handles.append(handle)
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self):
        self._importance.clear()

    def _make_hook(self, layer_idx):
        def hook(module, args, kwargs, output):
            cache = kwargs.get("past_key_values")
            if cache is None or layer_idx >= len(cache.layers):
                return output
            layer = cache.layers[layer_idx]
            if layer.keys is None:
                return output
            seq_len = layer.keys.shape[2]
            if seq_len <= self.config.window_size:
                return output

            device = layer.keys.device
            if self.config.mode == "importance":
                keep_idx = self._importance_keep_idx(layer_idx, output, seq_len, device)
            else:
                keep_idx = self._recency_keep_idx(seq_len, device)

            layer.keys = layer.keys.index_select(2, keep_idx)
            layer.values = layer.values.index_select(2, keep_idx)
            return output

        return hook

    def _recency_keep_idx(self, seq_len, device):
        sink = self.config.sink_tokens
        budget = self.config.window_size
        sink_idx = torch.arange(0, sink, device=device)
        recent_idx = torch.arange(seq_len - (budget - sink), seq_len, device=device)
        return torch.cat([sink_idx, recent_idx])

    def _importance_keep_idx(self, layer_idx, output, seq_len, device):
        sink = self.config.sink_tokens
        budget = self.config.window_size
        attn_weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None

        if attn_weights is not None:
            # mean over batch, heads, query positions -> one score per cached key
            score = attn_weights.detach().mean(dim=(0, 1, 2)).float()
            if score.shape[0] != seq_len:
                score = torch.nn.functional.pad(score, (0, seq_len - score.shape[0]))
        else:
            score = torch.zeros(seq_len, device=device)

        prev = self._importance.get(layer_idx)
        if prev is None or prev.shape[0] > seq_len:
            combined = score
        else:
            old_len = prev.shape[0]
            alpha = self.config.ema_alpha
            combined = score.clone()
            combined[:old_len] = alpha * prev + (1 - alpha) * score[:old_len]

        sink_idx = torch.arange(0, sink, device=device)
        rest_idx = torch.arange(sink, seq_len, device=device)
        rest_scores = combined[rest_idx]
        n_keep_rest = budget - sink
        top = torch.topk(rest_scores, k=min(n_keep_rest, rest_idx.shape[0])).indices
        keep_rest = rest_idx[top]
        keep_idx = torch.cat([sink_idx, keep_rest]).sort().values

        self._importance[layer_idx] = combined.index_select(0, keep_idx)
        return keep_idx
