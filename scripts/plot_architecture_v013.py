from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_shape(shape: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in shape.lower().split("x"))  # type: ignore[return-value]


def parse_shape_k_n(shape: str) -> tuple[int, int]:
    _, k, n = parse_shape(shape)
    return k, n


def method_label(row: pd.Series) -> str:
    method = str(row["method"])
    if method == "rns":
        return f"RNS-q{int(row['bits'])} ({row['lut_variant']} LUT)"
    return method.upper()


def save_latency(matrix: pd.DataFrame, output_dir: Path) -> None:
    selected = matrix[
        (matrix["method"].isin(["fp32", "fp16", "native_int8"]))
        | (
            (matrix["method"] == "rns")
            & (matrix["policy"] == "large_primes")
            & (matrix["lut_variant"] == "two")
        )
    ].copy()
    selected["label"] = selected.apply(method_label, axis=1)
    summary = selected.groupby("label", sort=False)["core_vs_fp16"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    summary.plot(kind="bar", ax=ax)
    ax.axhline(1.0, linewidth=1.0, linestyle="--")
    ax.set_ylabel("Median core latency / FP16 latency")
    ax.set_xlabel("")
    ax.set_title("Architecture latency relative to FP16")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "latency_vs_fp16_v013.png", dpi=220)
    plt.close(fig)


def save_storage(matrix: pd.DataFrame, output_dir: Path) -> None:
    selected = matrix[
        (matrix["method"].isin(["fp32", "fp16", "native_int8"]))
        | (
            (matrix["method"] == "rns")
            & (matrix["policy"] == "large_primes")
            & (matrix["lut_variant"] == "two")
        )
    ].copy()
    selected["label"] = selected.apply(method_label, axis=1)
    selected["bytes_per_weight"] = [
        float(row.weight_bytes) / (parse_shape_k_n(row.shape)[0] * parse_shape_k_n(row.shape)[1])
        for row in selected.itertuples()
    ]
    summary = selected.groupby("label", sort=False)["bytes_per_weight"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    summary.plot(kind="bar", ax=ax)
    ax.set_ylabel("Stored bytes per weight, including scales")
    ax.set_xlabel("")
    ax.set_title("Static weight-storage cost")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "weight_storage_v013.png", dpi=220)
    plt.close(fig)


def save_lut_tradeoff(matrix: pd.DataFrame, output_dir: Path) -> None:
    rns = matrix[(matrix["method"] == "rns") & (matrix["policy"] == "large_primes")].copy()
    if rns.empty:
        return
    representative = max(rns["shape"].unique(), key=parse_shape)
    rns = rns[rns["shape"] == representative]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for bits, group in rns.groupby("bits"):
        ordered = group.sort_values("lut_bytes")
        ax.plot(ordered["lut_bytes"], ordered["core_p50_ms"], marker="o", label=f"q{int(bits)}")
        for row in ordered.itertuples():
            ax.annotate(str(row.lut_variant), (row.lut_bytes, row.core_p50_ms), fontsize=8)
    ax.set_xlabel("Compact LUT bytes")
    ax.set_ylabel("RNS core p50 latency, ms")
    ax.set_title(f"LUT latency-memory trade-off ({representative})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lut_tradeoff_v013.png", dpi=220)
    plt.close(fig)


def save_concurrency(concurrency: pd.DataFrame, output_dir: Path) -> None:
    if concurrency.empty:
        return
    selected = concurrency[concurrency["policy"] == "large_primes"].copy()
    representative = max(selected["shape"].unique(), key=parse_shape)
    selected = selected[(selected["shape"] == representative) & (selected["lut_variant"].isin(["none", "two"]))]
    selected["label"] = selected.apply(
        lambda row: f"q{int(row['bits'])}-{row['lut_variant']}", axis=1
    )
    selected = selected.sort_values(["bits", "lut_variant"])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(selected["label"], selected["core_contention_ratio"])
    ax.axhline(1.0, linewidth=1.0, linestyle="--")
    ax.set_ylabel("Four-stream wall latency / single-request latency")
    ax.set_xlabel("")
    ax.set_title(f"Shared-weight/shared-LUT contention ({representative})")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "concurrency_v013.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix_path = args.results_dir / "architecture_matrix_v013.csv"
    concurrency_path = args.results_dir / "concurrency_v013.csv"
    if not matrix_path.exists():
        raise SystemExit(f"missing {matrix_path}")
    matrix = pd.read_csv(matrix_path)
    concurrency = pd.read_csv(concurrency_path) if concurrency_path.exists() and concurrency_path.stat().st_size else pd.DataFrame()
    save_latency(matrix, args.results_dir)
    save_storage(matrix, args.results_dir)
    save_lut_tradeoff(matrix, args.results_dir)
    save_concurrency(concurrency, args.results_dir)
    print("Saved paper figures to", args.results_dir)


if __name__ == "__main__":
    main()
