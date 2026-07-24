from __future__ import annotations

import os
from setuptools import find_packages, setup


def should_build_cuda() -> bool:
    mode = os.environ.get("RNS_LLM_BUILD_CUDA", "auto").lower()
    if mode in {"0", "false", "no"}:
        return False
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        if mode in {"1", "true", "yes"}:
            raise RuntimeError("PyTorch must be installed before building CUDA extensions")
        return False
    available = CUDA_HOME is not None
    if mode in {"1", "true", "yes"} and not available:
        raise RuntimeError("RNS_LLM_BUILD_CUDA=1 but CUDA toolkit/nvcc was not found")
    return available


ext_modules = []
cmdclass = {}
if should_build_cuda():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    common_cxx = ["-O3", "-std=c++17"]
    common_nvcc = ["-O3", "-std=c++17", "-lineinfo"]
    ext_modules = [
        CUDAExtension(
            name="rns_llm._C",
            sources=["csrc/bindings.cpp", "csrc/rns_cuda.cu"],
            libraries=["cublas"],
            extra_compile_args={
                "cxx": common_cxx,
                "nvcc": [*common_nvcc, "--use_fast_math"],
            },
        ),
        CUDAExtension(
            name="rns_llm._V07",
            sources=["csrc/v07_extension.cu"],
            libraries=["cublas"],
            extra_compile_args={"cxx": common_cxx, "nvcc": common_nvcc},
        ),
        CUDAExtension(
            name="rns_llm._HYBRID",
            sources=["csrc/hybrid_rns_extension.cu"],
            libraries=["cublas"],
            extra_compile_args={"cxx": common_cxx, "nvcc": common_nvcc},
        ),
        CUDAExtension(
            name="rns_llm._PREFILL",
            sources=["csrc/v011_prefill_extension.cu"],
            libraries=["cublas", "cublasLt"],
            extra_compile_args={
                "cxx": common_cxx,
                "nvcc": [*common_nvcc, "--ptxas-options=-v"],
            },
        ),
        CUDAExtension(
            name="rns_llm._ARCH",
            sources=["csrc/v013_architecture_extension.cu"],
            libraries=["cublas"],
            extra_compile_args={
                "cxx": common_cxx,
                "nvcc": [*common_nvcc, "--ptxas-options=-v"],
            },
        ),
    ]
    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True)}


setup(
    name="rns-llm",
    version="0.14.2",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
