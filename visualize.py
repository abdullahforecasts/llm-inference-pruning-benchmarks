"""
Turns a benchmark CSV into the figures and summary behind the memory/throughput
numbers:

  - memory_scaling.png: peak VRAM vs. sequence length per model (baseline vs.
    pruned), with a scikit-learn linear fit on each to show the scaling trend.
  - throughput_comparison.png: decode throughput and time-to-first-token per
    model and sequence length, with std error bars when repeats were recorded.
  - summary.csv: per model / batch size, mean peak-memory reduction and
    throughput speedup, computed only over settings where pruning actually
    evicted tokens (prompt + generated tokens > window).

Also reads the older v0 CSV format (no batch_size / std / pruning_active columns).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_theme(style="whitegrid")
COLORS = {"baseline": "#d95f02", "pruned": "#1b9e77"}
KEYS = ["model", "batch_size", "seq_len"]


def read_mixed_csv(csv_path):
    """Read a CSV whose later rows may carry one extra trailing column (`status`)
    that the header, written by an earlier run, does not have."""
    import csv
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], [r for r in rows[1:] if r]
    width = max(len(r) for r in body) if body else len(header)
    if width == len(header) + 1 and "status" not in header:
        header = header + ["status"]
    body = [r + [""] * (len(header) - len(r)) for r in body]
    df = pd.DataFrame(body, columns=header).replace(["", "nan", "NaN"], np.nan)
    for col in df.columns:
        if col != "status":
            conv = pd.to_numeric(df[col], errors="coerce")
            if conv.notna().sum() == df[col].notna().sum():
                df[col] = conv
    if "status" in df:
        df["status"] = df["status"].fillna("ok")
    return df


def load(csv_path):
    df = read_mixed_csv(csv_path)
    if "batch_size" not in df:
        df["batch_size"] = 1
    if "max_new_tokens" not in df:
        df["max_new_tokens"] = 128
    pruned = df[df["run"] == "pruned"].copy()
    if "pruning_active" not in df or pruned["pruning_active"].isna().all():
        pruned["pruning_active"] = (pruned["seq_len"] + pruned["max_new_tokens"]
                                    > pd.to_numeric(pruned["window_size"]))
    active = pruned.set_index(KEYS)["pruning_active"].astype(str).str.lower() == "true"
    df = df.drop(columns=["pruning_active"], errors="ignore")
    df = df.join(active.rename("pruning_active"), on=KEYS)
    df["pruning_active"] = df["pruning_active"].fillna(False).astype(bool)
    # if a setting was run more than once (appended CSV), keep the latest
    return df.drop_duplicates(subset=KEYS + ["run"], keep="last")


def groups(df):
    return [(m, b, g) for (m, b), g in df.groupby(["model", "batch_size"], sort=False)]


def plot_memory_scaling(df, out_path):
    gs = groups(df)
    fig, axes = plt.subplots(1, len(gs), figsize=(6.5 * len(gs), 5), squeeze=False)
    for ax, (model, bs, g) in zip(axes[0], gs):
        for run in ["baseline", "pruned"]:
            sub = g[g["run"] == run].dropna(subset=["peak_mem_mb"]).sort_values("seq_len")
            x, y = sub[["seq_len"]].values, sub["peak_mem_mb"].values
            ax.scatter(x, y, color=COLORS[run], label=f"{run} (measured)")
            if len(sub) >= 2:
                reg = LinearRegression().fit(x, y)
                xl = np.linspace(x.min(), x.max(), 50).reshape(-1, 1)
                ax.plot(xl, reg.predict(xl), color=COLORS[run], linestyle="--",
                        label=f"{run} fit ({reg.coef_[0]:.3f} MB/tok)")
        ax.set_title(f"{model}, batch {bs}")
        ax.set_xlabel("prompt length (tokens)")
        ax.set_ylabel("peak GPU memory (MB)")
        ax.legend(fontsize=8)
    fig.suptitle("Peak memory vs. sequence length")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _bars(ax, g, value, title, ylabel):
    piv = g.pivot_table(index="seq_len", columns="run", values=value)
    std_col = value + "_std"
    err = g.pivot_table(index="seq_len", columns="run", values=std_col) if std_col in g else None
    piv.plot(kind="bar", ax=ax, color=[COLORS[c] for c in piv.columns],
             yerr=err, capsize=3, rot=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("prompt length (tokens)")


def plot_throughput(df, out_path):
    gs = groups(df)
    fig, axes = plt.subplots(len(gs), 2, figsize=(12, 4.5 * len(gs)), squeeze=False)
    for row, (model, bs, g) in zip(axes, gs):
        _bars(row[0], g, "throughput_tok_s", f"{model}, batch {bs}: decode throughput", "tokens/sec")
        _bars(row[1], g, "ttft_s", f"{model}, batch {bs}: time to first token", "seconds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize(df):
    if "status" not in df:
        df = df.assign(status="ok")
    rows = []
    for model, bs, g in groups(df):
        g = g[g["pruning_active"]]
        if g.empty:
            continue
        status = g.pivot_table(index="seq_len", columns="run", values="status", aggfunc="first")
        oom = [str(s) for s in status.index if status.loc[s].get("baseline") == "oom"
               and status.loc[s].get("pruned") == "ok"]
        ok = g[g["status"] == "ok"]
        piv = ok.pivot_table(index="seq_len", columns="run",
                             values=["peak_mem_mb", "throughput_tok_s"]).dropna()
        row = {"model": model, "batch_size": bs,
               "seq_lens": " ".join(str(s) for s in piv.index),
               "baseline_oom_at": " ".join(oom)}
        if not piv.empty and "baseline" in piv["peak_mem_mb"]:
            mem_red = 1 - piv["peak_mem_mb"]["pruned"] / piv["peak_mem_mb"]["baseline"]
            speedup = piv["throughput_tok_s"]["pruned"] / piv["throughput_tok_s"]["baseline"]
            row.update({
                "mean_peak_mem_reduction_pct": round(100 * mem_red.mean(), 2),
                "max_peak_mem_reduction_pct": round(100 * mem_red.max(), 2),
                "mean_throughput_speedup_x": round(speedup.mean(), 3),
                "max_throughput_speedup_x": round(speedup.max(), 3),
            })
        rows.append(row)
    summary = pd.DataFrame(rows)
    print("summary (settings where pruning evicted tokens; baseline OOMs listed separately):")
    print(summary.to_string(index=False) if not summary.empty else "  none")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/benchmark.csv")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args.csv)
    plot_memory_scaling(df, os.path.join(args.out_dir, "memory_scaling.png"))
    plot_throughput(df, os.path.join(args.out_dir, "throughput_comparison.png"))
    summarize(df).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
