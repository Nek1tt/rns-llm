from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run(command: list[str], env: dict[str, str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--nsys-repeats", type=int, default=100)
    parser.add_argument("--run-ncu", action="store_true")
    parser.add_argument("--gpu-metrics", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/v0.11/nsight"))
    args = parser.parse_args()

    env = os.environ.copy()
    env["OUTDIR"] = str(args.output_dir)
    env["GPU_METRICS"] = "1" if args.gpu_metrics else "0"

    nsys_methods = [
        ("fp16", "prepared"),
        ("native_int8", "prepared"),
        ("hybrid_rns_serial", "prepared"),
        ("hybrid_rns_parallel", "prepared"),
        ("hybrid_fp16_serial", "prepared"),
        ("hybrid_fp16_parallel", "prepared"),
        ("hybrid_rns_serial", "e2e"),
        ("hybrid_rns_parallel", "e2e"),
    ]
    for m in args.m:
        for method, stage in nsys_methods:
            run(
                [
                    "bash",
                    "scripts/profile_nsys_prefill_v011.sh",
                    str(args.audit),
                    str(args.layer_index),
                    method,
                    stage,
                    str(m),
                    str(args.nsys_repeats),
                ],
                env,
            )

    if args.run_ncu:
        ncu_targets = [
            "fp16",
            "native_int8",
            "preprocess",
            "main_int8",
            "rns_correction",
            "fp16_correction",
            "rns_fused_epilogue",
        ]
        # One medium/large prefill point is enough for detailed counter collection.
        ncu_m = args.m[-1]
        for target in ncu_targets:
            try:
                run(
                    [
                        "bash",
                        "scripts/profile_ncu_prefill_v011.sh",
                        str(args.audit),
                        str(args.layer_index),
                        target,
                        str(ncu_m),
                    ],
                    env,
                )
            except subprocess.CalledProcessError as exc:
                print(f"NCU target {target} failed with {exc.returncode}; continuing.", flush=True)


if __name__ == "__main__":
    main()
