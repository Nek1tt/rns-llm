#!/usr/bin/env python3
"""Fail-fast validation for the installed v0.11.3 CUDA package.

This script deliberately imports only the wheel installed in site-packages.  It
removes a repository ``src`` directory from sys.path and purges stale modules,
which prevents a source-tree package from shadowing compiled extension modules.
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import py_compile
import sys


def _is_under(path: str | os.PathLike[str], parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def import_installed(repo_root: Path):
    src_root = (repo_root / "src").resolve()
    cleaned = []
    for entry in sys.path:
        resolved = Path(entry or os.getcwd()).resolve()
        if resolved == src_root:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned

    for name in list(sys.modules):
        if name == "rns_llm" or name.startswith("rns_llm."):
            del sys.modules[name]
    importlib.invalidate_caches()

    import torch  # preload libtorch DSOs
    pkg = importlib.import_module("rns_llm")
    prefill = importlib.import_module("rns_llm._PREFILL")
    hybrid = importlib.import_module("rns_llm._HYBRID")

    for module in (pkg, prefill, hybrid):
        file = Path(module.__file__).resolve()
        if _is_under(file, src_root):
            raise RuntimeError(f"source-tree shadowing detected: {file}")
    return torch, pkg, prefill, hybrid


def validate_api(prefill, hybrid) -> None:
    required_prefill = {
        "LtInt8Plan", "LtFp16Plan", "quantize_weight_masked_out",
        "quantize_weight_all_out", "encode_protected_weight_out",
        "quantize_rows_out", "fused_hybrid_preprocess_out",
        "cast_fp32_to_fp16_out", "add_bias_out", "dequant_epilogue_out",
        "rns_rankk_correction_out", "fp16_rankk_correction_out",
        "merge_epilogue_out", "rns_fused_epilogue_out",
        "fp16_fused_epilogue_out",
    }
    required_hybrid = {
        "encode_activation_fp32_out", "encode_weight_fp32_out",
        "quantize_activation_int8_out", "quantize_weight_int8_out",
        "native_mm_dequant_fp32_out", "rns_mm_dequant_fp32_out",
    }
    missing_p = sorted(required_prefill - set(dir(prefill)))
    missing_h = sorted(required_hybrid - set(dir(hybrid)))
    if missing_p or missing_h:
        raise RuntimeError(f"extension API mismatch: PREFILL={missing_p}, HYBRID={missing_h}")


def compile_python(repo_root: Path) -> None:
    paths = [repo_root / "src", repo_root / "scripts", repo_root / "benchmarks"]
    failures = []
    for base in paths:
        for path in base.rglob("*.py"):
            try:
                py_compile.compile(str(path), doraise=True)
            except Exception as exc:  # pragma: no cover - diagnostics
                failures.append(f"{path}: {exc}")
    if failures:
        raise RuntimeError("Python syntax failures:\n" + "\n".join(failures))


def cuda_smoke(torch, prefill) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    m, k, n = 16, 256, 256
    workspace_bytes = 8 * 1024 * 1024
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    a8 = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="cuda")
    b8 = torch.randint(-8, 8, (k, n), dtype=torch.int8, device="cuda")
    c32 = torch.empty((m, n), dtype=torch.int32, device="cuda")
    p8 = prefill.LtInt8Plan(m, k, n, workspace_bytes)
    p8.run(a8, b8, c32, workspace)

    a16, b16 = a8.half(), b8.half()
    cf = torch.empty((m, n), dtype=torch.float32, device="cuda")
    p16 = prefill.LtFp16Plan(m, k, n, workspace_bytes)
    p16.run(a16, b16, cf, workspace)
    torch.cuda.synchronize()
    ref_i32 = (a8.cpu().int() @ b8.cpu().int()).to(device="cuda")
    if not torch.equal(c32, ref_i32):
        diff = int((c32 - ref_i32).abs().max().item())
        raise RuntimeError(f"INT8 cuBLASLt smoke mismatch, max diff={diff}")
    if not torch.isfinite(cf).all():
        raise RuntimeError("FP16 cuBLASLt smoke produced non-finite output")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--skip-cuda-smoke", action="store_true")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    compile_python(repo)
    torch, pkg, prefill, hybrid = import_installed(repo)
    validate_api(prefill, hybrid)
    if not args.skip_cuda_smoke:
        cuda_smoke(torch, prefill)
    print("PREFLIGHT_OK")
    print("package:", pkg.__file__)
    print("prefill:", prefill.__file__)
    print("hybrid:", hybrid.__file__)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)


if __name__ == "__main__":
    main()
