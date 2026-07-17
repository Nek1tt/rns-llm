from __future__ import annotations

import numpy as np
import torch

from rns_llm.backends import CudaRNSBackend
from rns_llm.reference import choose_moduli_for_dot


def exact(a, b):
    return a.cpu().numpy().astype(np.int64) @ b.cpu().numpy().astype(np.int64)


def check_exact(backend, a, b, moduli, kernel):
    actual = backend.matmul_wide(a, b, moduli, kernel=kernel)
    np.testing.assert_array_equal(actual.cpu().numpy(), exact(a, b))
    print(f"PASS centered kernel={kernel}")


def check_fused(backend, a, b, moduli, lut_channels):
    prepared = backend.prepare_weight(b, moduli)
    if prepared.kernel != "cublas":
        print("SKIP fused: shape not cuBLAS compatible")
        return
    workspace = backend.create_workspace(
        device=a.device,
        channels=len(moduli),
        m=a.shape[0],
        n=b.shape[1],
    )
    actual = backend.matmul_prepared_fused(
        a, prepared, lut_channels=lut_channels, workspace=workspace
    )
    np.testing.assert_array_equal(actual.cpu().numpy(), exact(a, b))
    print(f"PASS fused Garner lut_channels={lut_channels}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    torch.manual_seed(7)
    backend = CudaRNSBackend()

    a8 = torch.randint(-127, 128, (32, 64), dtype=torch.int8, device="cuda")
    b8 = torch.randint(-127, 128, (64, 32), dtype=torch.int8, device="cuda")
    mods8 = choose_moduli_for_dot(64, 127, 127, strategy="dense_coprime")

    check_exact(backend, a8, b8, mods8, "scalar")
    if torch.cuda.get_device_capability() >= (6, 1):
        check_exact(backend, a8, b8, mods8, "dp4a")
        check_exact(backend, a8, b8, mods8, "dp4a_safe")
        if backend.cublas_compatible(k=64, n=32, moduli=mods8, device=a8.device):
            check_exact(backend, a8, b8, mods8, "cublas")
    check_exact(backend, a8, b8, mods8, "auto")
    for lut_channels in (0, 1, 2):
        check_fused(backend, a8, b8, mods8, lut_channels)

    max12 = 2047
    a12 = torch.randint(-max12, max12 + 1, (16, 64), dtype=torch.int16, device="cuda")
    b12 = torch.randint(-max12, max12 + 1, (64, 16), dtype=torch.int16, device="cuda")
    mods12 = choose_moduli_for_dot(64, max12, max12, strategy="dense_coprime")
    check_exact(backend, a12, b12, mods12, "auto")
    for lut_channels in (0, 1, 2):
        check_fused(backend, a12, b12, mods12, lut_channels)
    print(f"PASS wide-int12 channels={len(mods12)} moduli={mods12}")

    # Continuous batching: one merged M dimension must exactly match separate calls.
    prepared8 = backend.prepare_weight(b8, mods8)
    requests = [a8[:1], a8[1:4], a8[4:8], a8[8:10]]
    batch_workspace = backend.create_request_batch_workspace(
        device=a8.device,
        dtype=a8.dtype,
        rows_per_request=[x.shape[0] for x in requests],
        k=a8.shape[1],
        channels=len(mods8),
        n=b8.shape[1],
    )
    merged_outputs = backend.matmul_prepared_fused_requests(
        requests, prepared8, workspace=batch_workspace, clone_outputs=True
    )
    for request, output in zip(requests, merged_outputs):
        np.testing.assert_array_equal(output.cpu().numpy(), exact(request, b8))
    print("PASS continuous batching exactness")

    # Adaptive prefix: selection is accepted only after a strict bound check.
    adaptive_weight = backend.prepare_weight_adaptive(b8, mods8, min_channels=3)
    adaptive_output, adaptive_meta = backend.matmul_prepared_adaptive_fused(
        a8, adaptive_weight, return_metadata=True
    )
    np.testing.assert_array_equal(adaptive_output.cpu().numpy(), exact(a8, b8))
    assert adaptive_meta["bound"] <= adaptive_meta["capacity"]
    print(f"PASS adaptive channels={adaptive_meta['channels']} bound={adaptive_meta['bound']}")


if __name__ == "__main__":
    main()
