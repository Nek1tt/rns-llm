#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil
from pathlib import Path

REQUIRED = ("ppl_result_macros.tex", "ppl_results_table.tex", "ppl_results_paragraph.tex")

def main() -> None:
    ap = argparse.ArgumentParser(description="Copy v0.12 PPL paper artifacts into the Overleaf project")
    ap.add_argument("results", type=Path, help="results directory or extracted results ZIP root")
    ap.add_argument("paper", type=Path, help="Overleaf project root")
    args = ap.parse_args()
    candidates = [args.results / "paper", args.results]
    source = next((p for p in candidates if all((p/n).exists() for n in REQUIRED)), None)
    if source is None:
        raise SystemExit(f"Could not find {REQUIRED} under {args.results}")
    target = args.paper / "generated"
    target.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        shutil.copy2(source / name, target / name)
        print(f"copied {source/name} -> {target/name}")

if __name__ == "__main__":
    main()
