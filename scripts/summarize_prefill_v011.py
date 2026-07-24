from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(value: float) -> str:
    return f"{value:.5f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.benchmark.read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("=" * 120)
    lines.append("RNS LLM v0.11 PREFILL-FIRST SUMMARY")
    lines.append(f"Device: {data.get('device')} | logical correction: q{data.get('logical_bits')}")
    lines.append("=" * 120)
    for layer in data["layers"]:
        lines.append("")
        lines.append(
            f"LAYER {layer['layer_name']} shape={layer['shape']} protected={layer['protected']['count']} "
            f"({100.0 * layer['protected']['ratio']:.4f}%)"
        )
        lines.append(
            "M     FP16 core  INT8 core  RNS serial  RNS parallel  FP16corr serial  "
            "RNS E2E best  A  B  C  RNS<=FP16corr"
        )
        for shape in layer["shapes"]:
            core = shape["core"]
            e2e = shape["end_to_end"]
            gates = shape["gates"]
            best_e2e = min(
                e2e["hybrid_rns_serial"]["p50_ms"],
                e2e["hybrid_rns_parallel"]["p50_ms"],
            )
            lines.append(
                f"{shape['m']:<5d} "
                f"{f(core['fp16']['p50_ms']):>10} "
                f"{f(core['native_int8']['p50_ms']):>10} "
                f"{f(core['hybrid_rns_serial']['p50_ms']):>11} "
                f"{f(core['hybrid_rns_parallel']['p50_ms']):>13} "
                f"{f(core['hybrid_fp16_serial']['p50_ms']):>15} "
                f"{f(best_e2e):>12} "
                f"{int(gates['A_native_int8_has_30pct_headroom'])}  "
                f"{int(gates['B_rns_correction_epilogue_within_15pct'])}  "
                f"{int(gates['C_hybrid_e2e_beats_fp16_by_10pct'])}  "
                f"{int(gates['rns_not_slower_than_fp16_correction'])}"
            )
    lines.append("")
    lines.append("=" * 120)
    lines.append("DECISION")
    lines.append(json.dumps(data["decision"], indent=2, ensure_ascii=False))
    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
