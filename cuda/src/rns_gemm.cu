#include "rns_gemm.h"
#include <stdexcept>

void rns_gemm(
    const std::uint8_t* a, const std::uint8_t* b, std::uint8_t* c,
    const std::uint16_t* moduli, std::size_t residue_channels,
    std::size_t m, std::size_t k, std::size_t n
) {
    (void)a; (void)b; (void)c; (void)moduli;
    (void)residue_channels; (void)m; (void)k; (void)n;
    // TODO(CUDA owner): one modulus -> multi-channel -> grouped/Tensor Core experiment.
    throw std::runtime_error("rns_gemm CUDA implementation is not ready");
}
