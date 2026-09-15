"""
Turns results/benchmark.csv into the figures used to back the memory/throughput
claims: peak VRAM vs. sequence length (baseline vs. pruned, with a linear fit on
each to show the scaling trend), and a throughput/TTFT comparison bar chart.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_theme(style="whitegrid")


def load(csv_path):
    df = pd.read_csv(csv_path)
    return df


def fit_scaling(df, run_label, y_col="peak_mem_mb"):
    sub = df[df["run"] == run_label].sort_values("seq_len")
    X = sub[["seq_len"]].values
    y = sub[y_col].values
    reg = LinearRegression().fit(X, y)
    return sub["seq_len"].values, y, reg


def plot_memory_scaling(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for run_label, color in [("baseline", "#d95f02"), ("pruned", "#1b9e77")]:
        x, y, reg = fit_scaling(df, run_label, "peak_mem_mb")
        ax.scatter(x, y, color=color, label=f"{run_label} (measured)")
        x_line = np.linspace(x.min(), x.max(), 50).reshape(-1, 1)
        ax.plot(x_line, reg.predict(x_line), color=color, linestyle="--",
                 label=f"{run_label} fit (slope={reg.coef_[0]:.3f} MB/tok)")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("peak GPU memory (MB)")
    ax.set_title("Peak memory vs. sequence length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_throughput(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    pivot_tp = df.pivot_table(index="seq_len", columns="run", values="throughput_tok_s")
    pivot_tp.plot(kind="bar", ax=axes[0], color=["#d95f02", "#1b9e77"])
    axes[0].set_ylabel("tokens/sec")
    axes[0].set_title("Decode throughput")

    pivot_ttft = df.pivot_table(index="seq_len", columns="run", values="ttft_s")
    pivot_ttft.plot(kind="bar", ax=axes[1], color=["#d95f02", "#1b9e77"])
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Time to first token")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize(df):
    agg = df.groupby("run")[["peak_mem_mb", "throughput_tok_s"]].mean()
    mem_reduction = 1 - agg.loc["pruned", "peak_mem_mb"] / agg.loc["baseline", "peak_mem_mb"]
    speedup = agg.loc["pruned", "throughput_tok_s"] / agg.loc["baseline", "throughput_tok_s"]
    print(agg)
    print(f"mean peak memory reduction: {mem_reduction * 100:.1f}%")
    print(f"mean throughput speedup: {speedup:.2f}x")
    return {"mem_reduction": mem_reduction, "speedup": speedup}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/benchmark.csv")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args.csv)

    plot_memory_scaling(df, os.path.join(args.out_dir, "memory_scaling.png"))
    plot_throughput(df, os.path.join(args.out_dir, "throughput_comparison.png"))
    summarize(df)


if __name__ == "__main__":
    main()
