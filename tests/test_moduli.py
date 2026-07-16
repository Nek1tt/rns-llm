from rns_llm.rns.moduli import choose_moduli, modulus_product, required_modulus_product


def test_choose_moduli_reaches_range():
    mods = choose_moduli(128, 8, 8)
    assert modulus_product(mods) >= required_modulus_product(128,8,8)
    assert max(mods) <= 255
