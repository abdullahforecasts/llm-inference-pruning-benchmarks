"""Shared settings for the benchmark and plotting scripts."""

from dataclasses import dataclass, field


MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

SEQ_LENGTHS = [256, 512, 1024, 2048]
MAX_NEW_TOKENS = 128

DEFAULT_WINDOW_SIZE = 512
DEFAULT_SINK_TOKENS = 4
DEFAULT_MODE = "importance"  # "sliding_window" | "importance"

RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"


@dataclass
class RunConfig:
    model_key: str = "qwen2.5-0.5b"
    seq_lengths: list = field(default_factory=lambda: list(SEQ_LENGTHS))
    max_new_tokens: int = MAX_NEW_TOKENS
    window_size: int = DEFAULT_WINDOW_SIZE
    sink_tokens: int = DEFAULT_SINK_TOKENS
    mode: str = DEFAULT_MODE
    device: str = "cuda"
    out_csv: str = "results/benchmark.csv"
