import numpy as np
from rns_llm.rns.matmul import rns_matmul


def test_rns_matmul_matches_integer_matmul():
    rng = np.random.default_rng(42)
    a = rng.integers(-8, 9, size=(16,32), dtype=np.int64)
    b = rng.integers(-8, 9, size=(32,12), dtype=np.int64)
    mods = (3,5,7,11,13,17,19)
    np.testing.assert_array_equal(rns_matmul(a,b,mods), a @ b)


def test_residue_shape():
    a = np.array([[1,2],[3,4]], dtype=np.int64); b = np.array([[5,6],[7,8]], dtype=np.int64)
    mods = (3,5,7)
    assert rns_matmul(a,b,mods,decode_result=False).shape == (3,2,2)
