"""Run isolated default-vs-RNS nanoGPT generation benchmarks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_backend(args, backend: str, output_path: Path) -> dict:
    command = [
        sys.executable,
        "sample.py",
        f"--out_dir={args.out_dir}",
        f"--device={args.device}",
        f"--dtype={args.dtype}",
        f"--inference_backend={backend}",
        "--benchmark=True",
        f"--benchmark_warmup={args.warmup}",
        f"--benchmark_runs={args.runs}",
        f"--benchmark_output={output_path}",
        f"--max_new_tokens={args.max_new_tokens}",
        "--num_samples=1",
        "--print_samples=False",
        f"--rns_quant_bits={args.rns_quant_bits}",
        f"--rns_attention={args.rns_attention}",
        f"--rns_include_lm_head={args.rns_include_lm_head}",
    ]
    environment = os.environ.copy()
    source_path = str((ROOT.parent / "rns-llm" / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, environment.get("PYTHONPATH", "")) if value
    )
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    with output_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="out-shakespeare-char")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--rns-quant-bits", type=int, choices=[8, 12, 16], default=8)
    parser.add_argument(
        "--rns-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rns-include-lm-head",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        default="",
        help="Comparison JSON path (default: <out-dir>/inference_comparison.json)",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("tokens/runs must be positive and warmup must be non-negative")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_path = out_dir / "inference_torch.json"
    rns_path = out_dir / "inference_rns.json"

    torch_result = run_backend(args, "torch", torch_path)
    rns_result = run_backend(args, "rns", rns_path)
    torch_latency = torch_result["timing"]["latency_p50_ms"]
    rns_latency = rns_result["timing"]["latency_p50_ms"]
    torch_peak = torch_result["memory"]["gpu_peak_allocated_bytes"]
    rns_peak = rns_result["memory"]["gpu_peak_allocated_bytes"]
    prompt_tokens = torch_result["model"]["prompt_tokens"]
    torch_tokens = torch_result["output"]["token_ids"][prompt_tokens:]
    rns_tokens = rns_result["output"]["token_ids"][prompt_tokens:]
    compared = min(len(torch_tokens), len(rns_tokens))
    matches = sum(
        torch_tokens[index] == rns_tokens[index] for index in range(compared)
    )

    comparison = {
        "torch_result": str(torch_path),
        "rns_result": str(rns_path),
        "speed": {
            "torch_latency_p50_ms": torch_latency,
            "rns_latency_p50_ms": rns_latency,
            "rns_speedup_vs_torch": torch_latency / rns_latency,
            "rns_slowdown_vs_torch": rns_latency / torch_latency,
            "torch_tokens_per_second": torch_result["timing"]["tokens_per_second"],
            "rns_tokens_per_second": rns_result["timing"]["tokens_per_second"],
        },
        "memory": {
            "torch_gpu_peak_allocated_bytes": torch_peak,
            "rns_gpu_peak_allocated_bytes": rns_peak,
            "rns_minus_torch_bytes": rns_peak - torch_peak,
            "rns_over_torch_ratio": rns_peak / torch_peak if torch_peak else None,
            "torch_process_peak_rss_bytes": torch_result["memory"]["process_peak_rss_bytes"],
            "rns_process_peak_rss_bytes": rns_result["memory"]["process_peak_rss_bytes"],
        },
        "output_agreement": {
            "compared_generated_tokens": compared,
            "matching_positions": matches,
            "matching_fraction": matches / compared if compared else None,
            "note": "Sampling divergence is expected after quantized logits differ.",
        },
        "rns_backend_stats": rns_result["backend_stats"],
    }

    comparison_path = Path(args.output) if args.output else out_dir / "inference_comparison.json"
    if not comparison_path.is_absolute():
        comparison_path = ROOT / comparison_path
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)
    log_path = comparison_path.with_suffix(".log")
    lines = [
        f"torch_latency_p50_ms={torch_latency:.3f}",
        f"rns_latency_p50_ms={rns_latency:.3f}",
        f"rns_speedup_vs_torch={comparison['speed']['rns_speedup_vs_torch']:.6f}",
        f"rns_slowdown_vs_torch={comparison['speed']['rns_slowdown_vs_torch']:.6f}",
        f"torch_gpu_peak_allocated_bytes={torch_peak}",
        f"rns_gpu_peak_allocated_bytes={rns_peak}",
        f"rns_minus_torch_bytes={rns_peak - torch_peak}",
        f"generated_token_match_fraction={comparison['output_agreement']['matching_fraction']}",
        f"rns_backend_stats={comparison['rns_backend_stats']}",
        f"json={comparison_path}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"log={log_path}")


if __name__ == "__main__":
    main()
