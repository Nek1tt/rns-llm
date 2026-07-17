from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path)
    parser.add_argument("--moduli", type=Path)
    parser.add_argument("--lut", type=Path)
    parser.add_argument("--concurrency", type=Path)
    parser.add_argument("--attention", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/SUMMARY.md"))
    args = parser.parse_args()

    lines = ["# RNS Project Results Summary", ""]
    if args.main:
        data = load(args.main)
        t = data["timings"]
        lines += [
            "## Non-modular operations and end-to-end latency",
            "",
            f"- Moduli: `{data['moduli']}` ({data['channels']} channels)",
            f"- Old CRT p50: `{t['decode_crt_pytorch']['p50_ms']:.4f} ms`",
            f"- Garner CUDA p50: `{t['decode_garner_cuda']['p50_ms']:.4f} ms`",
            f"- Old cached end-to-end p50: `{t['end_to_end_old_cached_weight']['p50_ms']:.4f} ms`",
            f"- Fused LUT2 workspace p50: `{t['end_to_end_fused_lut2_workspace']['p50_ms']:.4f} ms`",
            f"- Exactness: `{data['correctness']}`",
            "",
        ]
    if args.moduli:
        data = load(args.moduli)
        lines += ["## Moduli set tradeoff", ""]
        for item in data["results"]:
            if "skipped" in item:
                lines.append(f"- {item['strategy']}: skipped ({item['skipped']})")
            else:
                lines.append(
                    f"- {item['strategy']}: {item['channels']} channels, "
                    f"p50 `{item['timing']['p50_ms']:.4f} ms`, "
                    f"encoded `{item['memory']['encoded_inputs_bytes']} bytes`"
                )
        lines.append("")
    if args.lut:
        data = load(args.lut)
        lines += ["## Table reuse", ""]
        lines.append(
            f"- Compact two-table memory saving: "
            f"`{100*data['table_memory']['saving_fraction']:.2f}%` versus full multiplication tables"
        )
        for key, value in data["single_request"].items():
            lines.append(f"- LUT channels {key}: p50 `{value['p50_ms']:.4f} ms`")
        lines.append("")
    if args.concurrency:
        data = load(args.concurrency)
        lines += ["## Concurrency", ""]
        for key, value in data["results"].items():
            lines.append(
                f"- {key} requests: p50 `{value['batch_p50_ms']:.4f} ms`, "
                f"throughput `{value['throughput_requests_per_second']:.2f} req/s`"
            )
        lines.append("")
    if args.attention:
        data = load(args.attention)
        lines += ["## Transformer/Self-Attention shapes", ""]
        for item in data["results"]:
            if "skipped" not in item:
                lines.append(
                    f"- {item['name']}: RNS `{item['rns_fused']['p50_ms']:.4f} ms`, "
                    f"FP16 `{item['torch_fp16']['p50_ms']:.4f} ms`"
                )
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
