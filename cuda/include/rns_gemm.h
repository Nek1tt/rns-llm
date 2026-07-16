#pragma once
#include <cstddef>
#include <cstdint>

// OWNER: CUDA/performance.
// Row-major layout: A [R,M,K], B [R,K,N], C [R,M,N], moduli [R].
void rns_gemm(
    const std::uint8_t* a,
    const std::uint8_t* b,
    std::uint8_t* c,
    const std::uint16_t* moduli,
    std::size_t residue_channels,
    std::size_t m,
    std::size_t k,
    std::size_t n
);
