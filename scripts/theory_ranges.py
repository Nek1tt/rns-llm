from math import prod

from rns_llm.reference import choose_moduli_for_dot, dot_product_bound


def show(bits: int, k: int) -> None:
    max_abs = (1 << (bits - 1)) - 1
    moduli = choose_moduli_for_dot(k, max_abs, max_abs)
    bound = dot_product_bound(k, max_abs, max_abs)
    print(
        f"bits={bits:2d} K={k:4d} channels={len(moduli)} "
        f"bound={bound:,} product={prod(moduli):,} moduli={moduli}"
    )


if __name__ == "__main__":
    for bits in (8, 12, 16):
        show(bits, 768)
