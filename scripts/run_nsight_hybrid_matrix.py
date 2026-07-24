from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def run(command: list[str], env: dict[str, str] | None = None, check: bool = True) -> None:
    print("=" * 100, flush=True)
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=check, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--gpu-metrics", action="store_true")
    parser.add_argument("--ncu", action="store_true")
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    packs = [Path(item["pack_file"]) for item in audit["layer_decisions"] if item.get("pack_file")]
    if not packs:
        raise RuntimeError("No audit packs")
    pack = packs[0]
    env = dict(os.environ)
    env["GPU_METRICS"] = "1" if args.gpu_metrics else "0"

    cases = [
        ("fp16", "e2e"),
        ("native_int8", "e2e"),
        ("full_rns_q16", "prepared"),
        ("hybrid_int8_plus_fp16", "e2e"),
        ("hybrid_int8_plus_rns_q16", "e2e"),
        ("hybrid_int8_plus_rns_q16", "prepared"),
    ]
    for method, stage in cases:
        run([
            "bash", "scripts/profile_nsys_hybrid.sh", str(pack), method, stage,
            str(args.m), str(args.repeats),
        ], env=env)

    if args.ncu:
        for target in ("gemm", "epilogue"):
            run([
                "bash", "scripts/profile_ncu_hybrid.sh", str(pack),
                "hybrid_int8_plus_rns_q16", "prepared", str(args.m), target,
            ], check=False)


if __name__ == "__main__":
    main()
