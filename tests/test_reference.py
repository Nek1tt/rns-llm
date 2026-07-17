import numpy as np

from rns_llm.reference import (
    decode_numpy,
    encode_numpy,
    matmul_residues_numpy,
    rns_matmul_numpy,
)


def test_signed_roundtrip():
    moduli = (251, 241, 239, 233)
    values = np.arange(-10000, 10001, dtype=np.int64)
    np.testing.assert_array_equal(decode_numpy(encode_numpy(values, moduli), moduli), values)


def test_residue_matmul():
    rng = np.random.default_rng(4)
    moduli = (251, 241, 239, 233)
    a = rng.integers(-127, 128, (13, 29), dtype=np.int16).astype(np.int8)
    b = rng.integers(-127, 128, (29, 17), dtype=np.int16).astype(np.int8)
    a_rns = encode_numpy(a, moduli)
    b_rns = encode_numpy(b, moduli)
    output = matmul_residues_numpy(a_rns, b_rns, moduli)
    decoded = decode_numpy(output, moduli)
    np.testing.assert_array_equal(decoded, a.astype(np.int64) @ b.astype(np.int64))


def test_end_to_end_matmul():
    rng = np.random.default_rng(5)
    moduli = (251, 241, 239, 233)
    a = rng.integers(-32, 33, (7, 19), dtype=np.int16).astype(np.int8)
    b = rng.integers(-32, 33, (19, 11), dtype=np.int16).astype(np.int8)
    np.testing.assert_array_equal(
        rns_matmul_numpy(a, b, moduli),
        a.astype(np.int64) @ b.astype(np.int64),
    )


def test_choose_moduli_for_12_bit_dot():
    from math import prod
    from rns_llm.reference import choose_moduli_for_dot

    moduli = choose_moduli_for_dot(768, 2047, 2047)
    assert prod(moduli) > 2 * 768 * 2047 * 2047
    assert all(modulus <= 255 for modulus in moduli)


def test_garner_matches_crt_for_centered_residues():
    from rns_llm.reference import decode_garner_numpy, encode_centered_numpy

    rng = np.random.default_rng(123)
    values = rng.integers(-1_000_000, 1_000_001, size=(17, 13), dtype=np.int64)
    moduli = (255, 253, 251, 247)
    residues = encode_centered_numpy(values, moduli)
    actual = decode_garner_numpy(residues, moduli)
    np.testing.assert_array_equal(actual, values)


def test_dense_moduli_are_pairwise_coprime_and_cover_range():
    from math import prod
    from rns_llm.reference import choose_moduli_for_dot

    moduli = choose_moduli_for_dot(768, 127, 127, strategy="dense_coprime")
    assert prod(moduli) > 2 * 768 * 127 * 127
    assert max(moduli) <= 255


def test_minimal_prefix_channels_is_strictly_safe():
    from math import prod

    from rns_llm.adaptive import minimal_prefix_channels, signed_capacity

    moduli = (255, 253, 251, 247)
    bound = 5_000_000
    channels = minimal_prefix_channels(moduli, bound, min_channels=3)
    prefix = moduli[:channels]
    assert 2 * bound < prod(prefix)
    assert signed_capacity(prefix) >= bound
    if channels > 3:
        assert 2 * bound >= prod(moduli[: channels - 1])


def test_safe_l1_bound_dominates_exact_dot_products():
    from rns_llm.adaptive import safe_l1_dot_bound

    rng = np.random.default_rng(77)
    a = rng.integers(-40, 41, size=(9, 31), dtype=np.int64)
    b = rng.integers(-50, 51, size=(31, 7), dtype=np.int64)
    exact = a @ b
    bound = safe_l1_dot_bound(
        int(np.abs(a).sum(axis=1).max()),
        int(np.abs(b).max()),
    )
    assert int(np.abs(exact).max()) <= bound
