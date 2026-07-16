import numpy as np
import pytest
from rns_llm.rns.arithmetic import add_residues, decode, encode, multiply_residues, validate_moduli


def test_signed_roundtrip():
    mods = (3,5,7,11)
    values = np.arange(-500, 501, dtype=np.int64)
    np.testing.assert_array_equal(decode(encode(values, mods), mods), values)


def test_addition():
    mods = (3,5,7,11)
    a = np.arange(-20, 21, dtype=np.int64); b = np.arange(20, -21, -1, dtype=np.int64)
    result = decode(add_residues(encode(a, mods), encode(b, mods), mods), mods)
    np.testing.assert_array_equal(result, a + b)


def test_multiplication():
    mods = (3,5,7,11)
    a = np.arange(-20, 21, dtype=np.int64); b = np.full_like(a, 3)
    result = decode(multiply_residues(encode(a, mods), encode(b, mods), mods), mods)
    np.testing.assert_array_equal(result, a * b)


def test_non_coprime_rejected():
    with pytest.raises(ValueError): validate_moduli((3,6,7))
