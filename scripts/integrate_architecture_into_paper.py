from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FILES = (
    "architecture_results_table.tex",
    "lut_memory_table.tex",
    "architecture_result_macros.tex",
    "architecture_results_paragraph.tex",
)

OPTIONAL_FIGURES = (
    "latency_vs_fp16_v013.png",
    "weight_storage_v013.png",
    "lut_tradeoff_v013.png",
    "concurrency_v013.png",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    args = parser.parse_args()
    generated = args.paper_dir / "generated" / "v013"
    generated.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        source = args.results_dir / name
        if not source.exists():
            raise SystemExit(f"missing generated asset: {source}")
        shutil.copy2(source, generated / name)
    for name in OPTIONAL_FIGURES:
        source = args.results_dir / name
        if source.exists():
            shutil.copy2(source, generated / name)
    snippet = args.paper_dir / "generated" / "v013_architecture_snippet.tex"
    snippet.write_text(
        "% Include macros in the preamble:\n"
        "\\input{generated/v013/architecture_result_macros.tex}\n\n"
        "% Include in the Results section:\n"
        "\\input{generated/v013/architecture_results_paragraph.tex}\n"
        "\\input{generated/v013/architecture_results_table.tex}\n"
        "\\input{generated/v013/lut_memory_table.tex}\n"
    )
    print("Copied v0.13 paper assets to", generated)
    print("Integration snippet:", snippet)


if __name__ == "__main__":
    main()
