import numpy as np
import pytest
import torch

from rns_llm.backends import CudaRNSBackend, cuda_extension_available
from rns_llm.reference import choose_moduli_for_dot


pytestmark = pytest.mark.cuda


def require_cuda():
    if not torch.cuda.is_available() or not cuda_extension_available():
        pytest.skip("CUDA extension/GPU is unavailable")


def exact_reference(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return a.cpu().numpy().astype(np.int64) @ b.cpu().numpy().astype(np.int64)


@pytest.mark.parametrize("kernel", ["scalar", "dp4a", "dp4a_safe", "auto"])
def test_centered_cuda_int8(kernel):
    require_cuda()
    if kernel.startswith("dp4a") and torch.cuda.get_device_capability() < (6, 1):
        pytest.skip("DP4A requires SM 6.1+")

    backend = CudaRNSBackend()
    a = torch.randint(-127, 128, (17, 64), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 128, (64, 20), dtype=torch.int8, device="cuda")
    moduli = choose_moduli_for_dot(64, 127, 127)

    actual = backend.matmul_wide(a, b, moduli, kernel=kernel)
    np.testing.assert_array_equal(actual.cpu().numpy(), exact_reference(a, b))


def test_cublas_centered_cuda_int8():
    require_cuda()
    backend = CudaRNSBackend()
    a = torch.randint(-127, 128, (32, 64), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 128, (64, 32), dtype=torch.int8, device="cuda")
    moduli = choose_moduli_for_dot(64, 127, 127)
    if not backend.cublas_compatible(k=64, n=32, moduli=moduli, device=a.device):
        pytest.skip("cuBLAS INT8 constraints are not satisfied")

    actual = backend.matmul_wide(a, b, moduli, kernel="cublas")
    np.testing.assert_array_equal(actual.cpu().numpy(), exact_reference(a, b))


def test_wide_int12_exactness():
    require_cuda()
    backend = CudaRNSBackend()
    max_abs = 2047
    a = torch.randint(-max_abs, max_abs + 1, (16, 64), dtype=torch.int16, device="cuda")
    b = torch.randint(-max_abs, max_abs + 1, (64, 16), dtype=torch.int16, device="cuda")
    moduli = choose_moduli_for_dot(64, max_abs, max_abs)

    actual = backend.matmul_wide(a, b, moduli, kernel="auto")
    np.testing.assert_array_equal(actual.cpu().numpy(), exact_reference(a, b))


def test_prepared_weight_matches_uncached():
    require_cuda()
    backend = CudaRNSBackend()
    a = torch.randint(-127, 128, (24, 64), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 128, (64, 32), dtype=torch.int8, device="cuda")
    moduli = choose_moduli_for_dot(64, 127, 127)
    prepared = backend.prepare_weight(b, moduli)

    actual = backend.matmul_prepared(a, prepared, kernel="auto")
    expected = backend.matmul_wide(a, b, moduli, kernel="auto")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

@pytest.mark.parametrize("lut_channels", [0, 1, 2])
def test_fused_garner_matches_exact_int12(lut_channels):
    require_cuda()
    backend = CudaRNSBackend()
    max_abs = 2047
    a = torch.randint(-max_abs, max_abs + 1, (24, 64), dtype=torch.int16, device="cuda")
    b = torch.randint(-max_abs, max_abs + 1, (64, 32), dtype=torch.int16, device="cuda")
    moduli = choose_moduli_for_dot(64, max_abs, max_abs, strategy="dense_coprime")
    prepared = backend.prepare_weight(b, moduli)
    if prepared.kernel != "cublas":
        pytest.skip("shape is not cuBLAS compatible")

    workspace = backend.create_workspace(
        device=a.device,
        channels=len(moduli),
        m=a.shape[0],
        n=b.shape[1],
    )
    actual = backend.matmul_prepared_fused(
        a, prepared, lut_channels=lut_channels, workspace=workspace
    )
    np.testing.assert_array_equal(actual.cpu().numpy(), exact_reference(a, b))


def test_garner_decoder_matches_old_crt_decoder():
    require_cuda()
    backend = CudaRNSBackend()
    a = torch.randint(-127, 128, (32, 64), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 128, (64, 32), dtype=torch.int8, device="cuda")
    moduli = choose_moduli_for_dot(64, 127, 127, strategy="dense_coprime")
    a_rns = backend.encode_centered(a, moduli)
    b_rns = backend.encode_centered(b, moduli)
    residues = backend.matmul_centered_residues(a_rns, b_rns, moduli, kernel="cublas")
    old = backend.decode(residues, moduli)
    new = backend.decode_garner(residues, moduli)
    torch.testing.assert_close(new, old, rtol=0, atol=0)


def test_continuous_batch_matches_independent_requests():
    require_cuda()
    backend = CudaRNSBackend()
    moduli = choose_moduli_for_dot(64, 127, 127, strategy="dense_coprime")
    weight = torch.randint(-127, 128, (64, 32), dtype=torch.int8, device="cuda")
    prepared = backend.prepare_weight(weight, moduli)
    requests = [
        torch.randint(-127, 128, (rows, 64), dtype=torch.int8, device="cuda")
        for rows in (1, 3, 5, 2)
    ]
    batch_workspace = backend.create_request_batch_workspace(
        device=weight.device,
        dtype=torch.int8,
        rows_per_request=[x.shape[0] for x in requests],
        k=64,
        channels=len(moduli),
        n=32,
    )
    merged = backend.matmul_prepared_fused_requests(
        requests,
        prepared,
        workspace=batch_workspace,
        clone_outputs=True,
    )
    for request, actual in zip(requests, merged):
        expected = backend.matmul_prepared_fused(request, prepared)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(actual.cpu().numpy(), exact_reference(request, weight))


def test_adaptive_prefix_is_exact_and_falls_back_when_needed():
    require_cuda()
    backend = CudaRNSBackend()
    moduli = choose_moduli_for_dot(64, 127, 127, strategy="dense_coprime")
    weight = torch.randint(-40, 41, (64, 32), dtype=torch.int8, device="cuda")
    adaptive = backend.prepare_weight_adaptive(weight, moduli, min_channels=3)

    # Low-L1 activations should often fit a smaller prefix.
    a_low = torch.randint(-8, 9, (16, 64), dtype=torch.int8, device="cuda")
    actual_low, meta_low = backend.matmul_prepared_adaptive_fused(
        a_low, adaptive, return_metadata=True
    )
    np.testing.assert_array_equal(actual_low.cpu().numpy(), exact_reference(a_low, weight))
    assert meta_low["bound"] <= meta_low["capacity"]

    # Full-range activation remains exact; selection may use more channels.
    a_high = torch.randint(-127, 128, (16, 64), dtype=torch.int8, device="cuda")
    actual_high, meta_high = backend.matmul_prepared_adaptive_fused(
        a_high, adaptive, return_metadata=True
    )
    np.testing.assert_array_equal(actual_high.cpu().numpy(), exact_reference(a_high, weight))
    assert meta_high["bound"] <= meta_high["capacity"]
