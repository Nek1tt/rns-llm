import numpy as np
from rns_llm.backends import NumPyReferenceBackend, RNSMatmulBackend


def test_protocol():
    assert isinstance(NumPyReferenceBackend(), RNSMatmulBackend)


def test_reference_backend():
    backend = NumPyReferenceBackend()
    a = np.array([[1,-2],[3,4]], dtype=np.int64); b = np.array([[5,6],[-7,8]], dtype=np.int64)
    np.testing.assert_array_equal(backend.matmul(a,b,(3,5,7,11,13)), a @ b)
