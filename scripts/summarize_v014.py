from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def finite_numbers(values):
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def best(values):
    valid = finite_numbers(values)
    return min(valid) if valid else None


def lut_savings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        policy = str(row.get("lut_policy", "n/a"))
        if policy not in {"none", "one", "two", "all"}:
            continue
        key = (
            str(row.get("shape")),
            str(row.get("architecture")),
            int(row.get("logical_bits") or 0),
        )
        grouped[key][policy] = row
    result = []
    for (shape, architecture, bits), policies in grouped.items():
        all_row = policies.get("all")
        if not all_row:
            continue
        all_bytes = int(all_row.get("lut_allocated_bytes") or 0)
        for policy in ("one", "two"):
            row = policies.get(policy)
            if not row or all_bytes <= 0:
                continue
            active = int(row.get("lut_allocated_bytes") or 0)
            result.append({
                "shape": shape,
                "architecture": architecture,
                "logical_bits": bits,
                "policy": policy,
                "all_lut_bytes": all_bytes,
                "policy_lut_bytes": active,
                "lut_memory_saving_percent": 100.0 * (1.0 - active / all_bytes),
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/v0.14.2"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/v0.14.2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v0.14.2/summary"))
    args = parser.parse_args()

    matrix = load(args.root / "matrix/matrix_benchmark_v014.json")
    attention = load(args.root / "attention/attention_benchmark_v014.json")
    ppl = load(args.root / "ppl/ppl_unified_v014.json")
    preflight = load(args.root / "preflight_v014.json")

    matrix_rows = [] if matrix is None else matrix.get("aggregate", matrix.get("rows", []))
    attention_rows = [] if attention is None else attention.get("rows", [])
    ppl_rows = [] if ppl is None else ppl.get("results", [])

    full_matrix = [row for row in matrix_rows if row.get("architecture") == "full_rns"]
    hybrid_matrix = [row for row in matrix_rows if str(row.get("architecture", "")).startswith("hybrid")]
    non_fp16_attention = [row for row in attention_rows if row.get("variant") != "fp16"]
    ppl_success = [row for row in ppl_rows if row.get("status") in {"PASS", "FAIL"} and row.get("variant") != "fp16"]
    ppl_pass = [row for row in ppl_success if row.get("ppl_gate_pass") is True]

    nsight_ncu_manifests = sorted(args.reports_root.glob("ncu/*_manifest.json"))
    nsight_nsys_manifests = sorted(args.reports_root.glob("nsys/*_manifest.json"))
    nsight_ncu_json = sorted(args.reports_root.glob("ncu/*_raw_full.json"))
    nsight_nsys_sqlite = sorted(args.reports_root.glob("nsys/*.sqlite"))

    savings = lut_savings(matrix_rows)
    ppl_gate = [
        {
            key: row.get(key)
            for key in (
                "variant", "lut_policy", "status", "ppl",
                "relative_ppl_increase_percent", "ppl_gate_pass",
            )
        }
        for row in ppl_rows
    ]

    payload: dict[str, Any] = {
        "version": "0.14.2",
        "available": {
            "preflight": preflight is not None,
            "matrix": matrix is not None,
            "attention": attention is not None,
            "ppl": ppl is not None,
            "ncu_json_reports": len(nsight_ncu_json),
            "nsys_sqlite_reports": len(nsight_nsys_sqlite),
        },
        "matrix": {
            "best_full_rns_e2e_vs_fp16": best(row.get("vs_fp16") for row in full_matrix),
            "best_hybrid_e2e_vs_fp16": best(row.get("vs_fp16") for row in hybrid_matrix),
            "best_full_rns_weight_vs_fp16": best(row.get("weight_vs_fp16") for row in full_matrix),
            "best_hybrid_weight_vs_fp16": best(row.get("weight_vs_fp16") for row in hybrid_matrix),
            "rows": len(matrix_rows),
            "errors": [] if matrix is None else matrix.get("errors", []),
        },
        "attention": {
            "best_non_fp16_vs_fp16": best(row.get("vs_fp16") for row in non_fp16_attention),
            "best_non_fp16_relative_l2": best(row.get("relative_l2") for row in non_fp16_attention),
            "rows": len(attention_rows),
            "errors": [] if attention is None else attention.get("errors", []),
        },
        "lut_memory_savings": savings,
        "ppl": {
            "baseline_ppl": None if ppl is None else ppl.get("baseline_ppl"),
            "successful_variants": len(ppl_success),
            "gate_pass_count": len(ppl_pass),
            "best_relative_ppl_increase_percent": best(
                row.get("relative_ppl_increase_percent") for row in ppl_success
            ),
            "best_model_memory_vs_fp16": best(
                row.get("memory_allocated_vs_fp16") for row in ppl_success
            ),
            "best_peak_memory_vs_fp16": best(
                row.get("peak_memory_vs_fp16") for row in ppl_success
            ),
            "results": ppl_gate,
        },
        "nsight": {
            "ncu_manifests": [str(path) for path in nsight_ncu_manifests],
            "nsys_manifests": [str(path) for path in nsight_nsys_manifests],
            "ncu_json_reports": [str(path) for path in nsight_ncu_json],
            "nsys_sqlite_reports": [str(path) for path in nsight_nsys_sqlite],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "unified_summary_v014.json").write_text(
        json.dumps(payload, indent=2)
    )

    requirement_rows = [
        (
            "1. Moduli <= 8 bits and dynamic-range bound",
            "MEASURED" if matrix is not None else "MISSING RUN",
            "Full-RNS plans include INT8/16/32 and large-prime/school-small studies.",
        ),
        (
            "1. Speed/memory tradeoff",
            "MEASURED" if matrix is not None else "MISSING RUN",
            "Unified matrix JSON reports latency, channels, weights, LUT and workspace.",
        ),
        (
            "2. Full-RNS matrix multiplication",
            "MEASURED" if full_matrix else "MISSING RUN",
            "Actual CUDA residue GEMM and Garner paths.",
        ),
        (
            "2. Hybrid matrix multiplication",
            "MEASURED" if hybrid_matrix else "MISSING RUN",
            "INT8 main path plus FP16/RNS protected correction.",
        ),
        (
            "2/3. Complete attention block",
            "MEASURED" if attention is not None else "MISSING RUN",
            "QKV/out projection replaced; QK, mask, softmax and AV remain native and are included in latency.",
        ),
        (
            "4. LUT none/one/two/all and reuse",
            "MEASURED" if savings else "MISSING RUN",
            "Actual allocated LUT bytes and four-stream sharing are recorded.",
        ),
        (
            "5. Four concurrent requests",
            "MEASURED" if matrix is not None and attention is not None else "MISSING RUN",
            "Four CUDA streams; throughput speedup and contention are recorded.",
        ),
        (
            "5. PPL increase <5%",
            "MEASURED" if ppl is not None else "MISSING RUN",
            "Actual CUDA attention-projection variants on WikiText-2.",
        ),
        (
            "Nsight Compute kernel reports",
            f"{len(nsight_ncu_json)} JSON / {len(nsight_ncu_manifests)} manifests",
            "All metric rows from the selected article-essential sections, details and .ncu-rep files.",
        ),
        (
            "Nsight Systems SQL reports",
            f"{len(nsight_nsys_sqlite)} SQLite / {len(nsight_nsys_manifests)} manifests",
            "Full SQLite databases, SQL queries/schema and derived JSON.",
        ),
    ]
    markdown = [
        "# RNS LLM v0.14.2 requirement status",
        "",
        "| Requirement | Status | Evidence |",
        "|---|---|---|",
    ]
    markdown += [f"| {name} | {status} | {evidence} |" for name, status, evidence in requirement_rows]
    markdown += [
        "",
        "## Key automatic summaries",
        "",
        f"- Best full-RNS matrix E2E / FP16: {payload['matrix']['best_full_rns_e2e_vs_fp16']}",
        f"- Best hybrid matrix E2E / FP16: {payload['matrix']['best_hybrid_e2e_vs_fp16']}",
        f"- Best full-RNS weight / FP16: {payload['matrix']['best_full_rns_weight_vs_fp16']}",
        f"- Best hybrid weight / FP16: {payload['matrix']['best_hybrid_weight_vs_fp16']}",
        f"- Best non-FP16 attention / FP16: {payload['attention']['best_non_fp16_vs_fp16']}",
        f"- PPL variants passing <5% gate: {payload['ppl']['gate_pass_count']}",
        f"- Best full-model allocated memory / FP16 during PPL: {payload['ppl']['best_model_memory_vs_fp16']}",
        "",
        "The values are populated only from files produced by the notebook; no missing result is inferred.",
    ]
    (args.output_dir / "requirements_status_v014.md").write_text(
        "\n".join(markdown) + "\n"
    )

    def fmt(value: Any) -> str:
        return "--" if value is None else f"{float(value):.3f}"

    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\begin{tabular}{lr}", r"\toprule",
        r"Metric & Result " + r"\\", r"\midrule",
        f"Best full-RNS matrix E2E / FP16 & {fmt(payload['matrix']['best_full_rns_e2e_vs_fp16'])} " + r"\\",
        f"Best hybrid matrix E2E / FP16 & {fmt(payload['matrix']['best_hybrid_e2e_vs_fp16'])} " + r"\\",
        f"Best full-RNS weight / FP16 & {fmt(payload['matrix']['best_full_rns_weight_vs_fp16'])} " + r"\\",
        f"Best hybrid weight / FP16 & {fmt(payload['matrix']['best_hybrid_weight_vs_fp16'])} " + r"\\",
        f"Best attention E2E / FP16 & {fmt(payload['attention']['best_non_fp16_vs_fp16'])} " + r"\\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Unified v0.14.2 summary.}", r"\end{table}",
    ]
    (args.output_dir / "paper_summary_v014.tex").write_text("\n".join(tex) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
