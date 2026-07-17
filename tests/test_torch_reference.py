import torch

from rns_llm.backends import TorchReferenceBackend


def test_torch_reference_backend_cpu():
    backend = TorchReferenceBackend()
    moduli = (251, 241, 239, 233)
    a = torch.randint(-127, 128, (9, 21), dtype=torch.int8)
    b = torch.randint(-127, 128, (21, 5), dtype=torch.int8)
    actual = backend.matmul_int8(a, b, moduli)
    expected = a.to(torch.int64) @ b.to(torch.int64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
