from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.reports:
        payload = json.loads(path.read_text())
        print("=" * 100)
        print(path)
        print("GPU:", payload["hardware"]["device"])
        print("Correctness tests:", payload["correctness_testing"]["enabled"])
        for result in payload["results"]:
            shape = result["shape"]
            timings = result["timings"]
            v07 = timings["rns_v07_direct_fp16_4ch"]["p50_ms"]
            fp16 = timings["fp16_linear"]["p50_ms"]
            native = timings["native_int8_direct_fp16"]["p50_ms"]
            print(
                f"M={shape['m']} K={shape['k']} N={shape['n']} | "
                f"v0.7={v07:.6f} ms | FP16={fp16:.6f} ms | "
                f"native={native:.6f} ms | "
                f"gap FP16={v07/fp16:.3f}x | gap native={v07/native:.3f}x"
            )
            graphs = result.get("graph_timings", {})
            if graphs:
                gv07 = graphs.get("graph_rns_v07_direct_fp16_4ch", {}).get("p50_ms")
                gfp16 = graphs.get("graph_fp16_linear", {}).get("p50_ms")
                if gv07 is not None and gfp16 is not None:
                    print(
                        f"  CUDA Graph: v0.7={gv07:.6f} ms | "
                        f"FP16={gfp16:.6f} ms | gap={gv07/gfp16:.3f}x"
                    )


if __name__ == "__main__":
    main()
