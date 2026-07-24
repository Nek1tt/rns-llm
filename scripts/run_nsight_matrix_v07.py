from __future__ import annotations

import argparse
import os
import subprocess


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("=" * 100, flush=True)
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--gpu-metrics", action="store_true")
    args = parser.parse_args()

    env = dict(os.environ)
    env["GPU_METRICS"] = "1" if args.gpu_metrics else "0"

    cases: list[tuple[str, str, int, int, int, int]] = []
    cases.extend(
        [
            ("rns_v07", "e2e", 1, 768, 768, 300),
            ("rns_v07", "e2e", 16, 768, 768, 150),
            ("rns_v07", "e2e", 128, 768, 768, 100),
        ]
    )
    if args.prepared:
        cases.extend(
            [
                ("rns_v07", "prepared", 1, 768, 768, 300),
                ("rns_v07", "prepared", 128, 768, 768, 100),
            ]
        )
    if args.baselines:
        cases.extend(
            [
                ("fp16", "e2e", 1, 768, 768, 300),
                ("native_int8", "e2e", 1, 768, 768, 300),
                ("rns_v06", "e2e", 1, 768, 768, 300),
                ("fp16", "e2e", 128, 768, 768, 100),
                ("native_int8", "e2e", 128, 768, 768, 100),
            ]
        )
    if args.large:
        cases.extend(
            [
                ("rns_v07", "e2e", 1, 4096, 4096, 50),
                ("rns_v07", "prepared", 1, 4096, 4096, 50),
                ("fp16", "e2e", 1, 4096, 4096, 50),
                ("native_int8", "e2e", 1, 4096, 4096, 50),
            ]
        )

    seen = set()
    for case in cases:
        if case in seen:
            continue
        seen.add(case)
        backend, stage, m, k, n, repeats = case
        run(
            [
                "bash",
                "scripts/profile_nsys_v07.sh",
                backend,
                stage,
                str(m),
                str(k),
                str(n),
                str(repeats),
            ],
            env=env,
        )


if __name__ == "__main__":
    main()
