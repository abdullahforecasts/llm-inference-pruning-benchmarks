"""
Forward hooks that turn a stock HF causal LM into one with a bounded KV cache.

Every attention module gets a forward hook that runs after the layer has
updated its cache for the current forward pass (a prefill chunk or one decode
step) and evicts entries once the cache is over budget:

  - prefill chunk (q_len > 1): evict down to `window_size` whenever the cache is
    larger than that, so with chunked prefill the cache never exceeds
    window_size + chunk_size.
  - decode step (q_len == 1): let the cache grow to window_size + evict_slack,
    then evict back to window_size in one go. Eviction (a top-k plus one
    index_select per layer) happens once every `evict_slack` steps instead of
    on every step.

What is kept:

  - sliding_window: the first `sink_tokens` tokens (StreamingLLM attention sink)
    plus the most recent tokens.
  - importance: the sink tokens, the most recent `obs_window` tokens, and the
    highest-scoring tokens in between. Scores are the attention the last
    `obs_window` queries pay to each older key (SnapKV-style observation window),
    summed over queries, averaged over heads and batch, and max-pooled over
    neighbouring positions. The queries are captured with a hook on `q_proj`
    and rotated with the layer's own RoPE, so the model can keep running on
    SDPA; no eager attention or full attention matrix is needed.

One selection is shared by all heads and all sequences in the batch (the
benchmark uses the same prompt for every sequence).

Tested against Qwen2/Qwen2.5 and Llama-3 attention modules, which expose
`layer_idx`, `q_proj`, `head_dim`, `scaling`, and store cache tensors as
[batch, kv_heads, seq_len, head_dim] in `cache.layers[layer_idx].keys/.values`.
"""

import importlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PruneConfig:
    window_size: int = 512    # cached tokens per layer after an eviction
    sink_tokens: int = 4      # always-kept prefix (attention sink)
    mode: str = "importance"  # "sliding_window" | "importance"
    obs_window: int = 32      # recent tokens that are always kept and whose queries score the rest
    evict_slack: int = 32     # decode steps between evictions
    pool_kernel: int = 5      # max-pool width applied to importance scores

    def __post_init__(self):
        if self.window_size <= self.sink_tokens + self.obs_window:
            raise ValueError("window_size must be larger than sink_tokens + obs_window")


def _find_attention_modules(model):
    return [(name, m) for name, m in model.named_modules()
            if name.endswith("self_attn") and hasattr(m, "layer_idx")]


def _rope_fn(module):
    return importlib.import_module(type(module).__module__).apply_rotary_pos_emb


class KVCachePruner:
    """Attaches/detaches the eviction hooks and owns the per-layer query buffers."""

    def __init__(self, config: PruneConfig):
        self.config = config
        self._handles = []
        self._q_raw = {}    # layer_idx -> q_proj output of the current forward (last obs tokens)
        self._q_buf = {}    # layer_idx -> rotated queries of the last obs_window tokens
        self.evictions = 0

    def attach(self, model):
        self.detach()
        self.reset()
        for _, module in _find_attention_modules(model):
            idx = module.layer_idx
            if self.config.mode == "importance":
                self._handles.append(module.q_proj.register_forward_hook(self._make_q_hook(idx)))
            self._handles.append(
                module.register_forward_hook(self._make_attn_hook(module, idx), with_kwargs=True))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self):
        self._q_raw.clear()
        self._q_buf.clear()
        self.evictions = 0

    # -- hooks ---------------------------------------------------------------

    def _make_q_hook(self, layer_idx):
        obs = self.config.obs_window

        def hook(module, args, output):
            self._q_raw[layer_idx] = output[:, -obs:].detach()

        return hook

    def _make_attn_hook(self, module, layer_idx):
        rope = _rope_fn(module) if self.config.mode == "importance" else None

        def hook(mod, args, kwargs, output):
            cache = kwargs.get("past_key_values")
            if cache is None or layer_idx >= len(cache.layers):
                return output
            layer = cache.layers[layer_idx]
            if layer.keys is None:
                return output

            if rope is not None:
                self._update_query_buffer(mod, layer_idx, kwargs, rope)

            hidden = kwargs.get("hidden_states", args[0] if args else None)
            q_len = hidden.shape[1]
            seq_len = layer.keys.shape[2]
            budget = self.config.window_size
            limit = budget if q_len > 1 else budget + self.config.evict_slack
            if seq_len <= limit:
                return output

            device = layer.keys.device
            if self.config.mode == "importance":
                keep_idx = self._importance_keep_idx(mod, layer_idx, layer.keys, seq_len, device)
            else:
                keep_idx = self._recency_keep_idx(seq_len, device)

            layer.keys = layer.keys.index_select(2, keep_idx)
            layer.values = layer.values.index_select(2, keep_idx)
            if layer_idx == 0:
                self.evictions += 1
            return output

        return hook

    # -- selection -----------------------------------------------------------

    def _update_query_buffer(self, mod, layer_idx, kwargs, rope):
        raw = self._q_raw.pop(layer_idx, None)
        pos_emb = kwargs.get("position_embeddings")
        if raw is None or pos_emb is None:
            return
        cos, sin = pos_emb
        b, n, _ = raw.shape
        q = raw.view(b, n, -1, mod.head_dim).transpose(1, 2)        # [b, Hq, n, d]
        q, _ = rope(q, q, cos[:, -n:], sin[:, -n:])
        prev = self._q_buf.get(layer_idx)
        q = q if prev is None else torch.cat([prev, q], dim=2)
        self._q_buf[layer_idx] = q[:, :, -self.config.obs_window:]

    def _recency_keep_idx(self, seq_len, device):
        sink, budget = self.config.sink_tokens, self.config.window_size
        return torch.cat([torch.arange(0, sink, device=device),
                          torch.arange(seq_len - (budget - sink), seq_len, device=device)])

    def _importance_keep_idx(self, mod, layer_idx, keys, seq_len, device):
        cfg = self.config
        sink, recent = cfg.sink_tokens, cfg.obs_window
        n_mid = cfg.window_size - sink - recent
        cand_end = seq_len - recent
        queries = self._q_buf.get(layer_idx)
        if queries is None:
            return self._recency_keep_idx(seq_len, device)

        b, hq, nq, d = queries.shape
        hkv = keys.shape[1]
        q = queries.reshape(b, hkv, (hq // hkv) * nq, d)              # group query heads by kv head
        k = keys[:, :, sink:cand_end]                                  # [b, Hkv, n_cand, d]
        logits = torch.matmul(q, k.transpose(-1, -2)).float() * mod.scaling
        score = logits.softmax(dim=-1).sum(dim=2).mean(dim=(0, 1))    # [n_cand]
        if cfg.pool_kernel > 1:
            score = F.max_pool1d(score[None, None], cfg.pool_kernel, stride=1,
                                 padding=cfg.pool_kernel // 2)[0, 0]

        mid = torch.topk(score, k=n_mid).indices.sort().values + sink
        return torch.cat([torch.arange(0, sink, device=device), mid,
                          torch.arange(cand_end, seq_len, device=device)])
