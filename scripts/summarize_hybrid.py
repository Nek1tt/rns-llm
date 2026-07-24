from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value):
    return "-" if value is None else f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("benchmark", type=Path, nargs="?")
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    print("=" * 100)
    print("MODEL AUDIT")
    print("Decision:", audit["decision"])
    print("Model:", audit["model"])
    print("Gate:", audit["gate"])
    for layer in audit["layer_decisions"]:
        plan = layer.get("evaluation", {}).get("selected_plan")
        stats = layer.get("statistics", {})
        print("\n", layer["layer_name"], "PASS" if layer.get("passed") else "STOP")
        print("shape:", layer.get("shape"), "reasons:", layer.get("reasons"), "warnings:", layer.get("warnings"))
        print(
            "row_outlier_rate=", fmt(stats.get("row_outlier_rate")),
            "top1_energy=", fmt(stats.get("top1_energy_ratio")),
            "top1_jaccard=", fmt(stats.get("top1_jaccard_mean")),
            "selected_cross_split_jaccard=", fmt(stats.get("selected_map_cross_split_jaccard")),
        )
        print(
            "heldout_event_recall=", fmt(stats.get("heldout_outlier_event_recall")),
            "heldout_energy=", fmt(stats.get("heldout_protected_energy_ratio")),
            "heldout_selected_overlap=", fmt(stats.get("heldout_selected_k_overlap_mean")),
        )
        if plan:
            print(
                "protected=", fmt(plan.get("protected_ratio")),
                "padded=", fmt(plan.get("protected_ratio_after_padding")),
                "q16 ideal/FP16=", fmt(plan.get("ideal_compute_ratio_vs_fp16", {}).get("16")),
                "error reduction=", fmt(plan.get("native_int8_error_reduction")),
            )

    if args.benchmark is None or not args.benchmark.exists():
        return
    result = json.loads(args.benchmark.read_text(encoding="utf-8"))
    print("\n" + "=" * 100)
    print("PERFORMANCE AND ACCURACY")
    for layer in result.get("layers", []):
        print("\nLAYER", layer["layer_name"], layer["shape"], "protected", layer["protected_ratio"])
        for shape in layer["shapes"]:
            print(f"  M={shape['m']}")
            print("    method                               e2e p50 ms   core p50 ms   rel-L2       cosine")
            for method in shape["methods"]:
                if "error" in method:
                    print(f"    {method['name']:<36} ERROR {method['error']}")
                    continue
                print(
                    f"    {method['name']:<36} "
                    f"{method['end_to_end']['p50_ms']:>10.5f}   "
                    f"{method['prepared_core']['p50_ms']:>11.5f}   "
                    f"{method['accuracy']['relative_l2']:>10.4g}   "
                    f"{method['accuracy']['cosine_similarity']:>10.6f}"
                )
    print("\nPROTOTYPE GATE")
    print(json.dumps(result.get("prototype_gate"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
