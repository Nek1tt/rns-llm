from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_bar(frame: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path) -> None:
    if frame.empty:
        return
    figure, axis = plt.subplots(figsize=(max(8, 0.32 * len(frame)), 5))
    axis.bar(frame[x].astype(str), frame[y].astype(float))
    axis.axhline(1.0, linewidth=1.0)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=80)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/v0.14.2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v0.14.2/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = args.root / "matrix/matrix_aggregate_v014.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path)
        for shape, subset in matrix.groupby("shape"):
            slug = str(shape).replace("x", "_")
            save_bar(
                subset, "variant", "vs_fp16",
                f"Matrix E2E latency ratio, {shape}", "Latency / FP16",
                args.output_dir / f"matrix_latency_{slug}.png",
            )
            save_bar(
                subset, "variant", "weight_vs_fp16",
                f"Static weight ratio, {shape}", "Weight bytes / FP16",
                args.output_dir / f"matrix_weight_{slug}.png",
            )

    attention_path = args.root / "attention/attention_benchmark_v014.csv"
    if attention_path.exists():
        attention = pd.read_csv(attention_path)
        save_bar(
            attention, "variant", "vs_fp16",
            "Complete OPT self-attention latency", "Latency / FP16",
            args.output_dir / "attention_latency.png",
        )
        save_bar(
            attention, "variant", "weight_vs_fp16",
            "Attention projection storage", "Weight bytes / FP16",
            args.output_dir / "attention_weight.png",
        )

    ppl_path = args.root / "ppl/ppl_unified_v014.csv"
    if ppl_path.exists():
        ppl = pd.read_csv(ppl_path)
        ppl = ppl[ppl["relative_ppl_increase_percent"].notna()].copy()
        ppl["label"] = ppl["variant"].astype(str) + ":" + ppl["lut_policy"].astype(str)
        if not ppl.empty:
            figure, axis = plt.subplots(figsize=(max(8, 0.4 * len(ppl)), 5))
            axis.bar(ppl["label"], ppl["relative_ppl_increase_percent"])
            axis.axhline(5.0, linewidth=1.0)
            axis.set_ylabel("Relative PPL increase (%)")
            axis.set_title("PPL gate for actual CUDA attention projections")
            axis.tick_params(axis="x", rotation=80)
            figure.tight_layout()
            figure.savefig(args.output_dir / "ppl_gate.png", dpi=180)
            plt.close(figure)

    print(args.output_dir)


if __name__ == "__main__":
    main()
