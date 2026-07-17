#include <torch/extension.h>

#include <cstdint>

torch::Tensor rns_encode_int8_cuda(
    torch::Tensor values,
    torch::Tensor moduli);

torch::Tensor rns_encode_centered_cuda(
    torch::Tensor values,
    torch::Tensor moduli);

torch::Tensor rns_matmul_residues_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor moduli,
    torch::Tensor reciprocals,
    std::int64_t kernel_id);

torch::Tensor rns_matmul_centered_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor moduli,
    std::int64_t kernel_id);

torch::Tensor rns_decode_garner_cuda(
    torch::Tensor residues,
    torch::Tensor moduli,
    torch::Tensor reciprocals,
    torch::Tensor pairwise_inverses,
    std::int64_t modulus_product);

torch::Tensor rns_matmul_centered_fused_out_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor moduli,
    torch::Tensor reciprocals,
    torch::Tensor pairwise_inverses,
    torch::Tensor compact_lut,
    std::int64_t modulus_product,
    std::int64_t lut_channels,
    torch::Tensor accumulators,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "encode_int8",
        &rns_encode_int8_cuda,
        "Encode signed int8 tensor into canonical uint8 RNS residue planes (CUDA)");

    module.def(
        "encode_centered",
        &rns_encode_centered_cuda,
        "Encode int8/int16/int32 tensor into centered signed-int8 RNS planes (CUDA)");

    module.def(
        "matmul_residues",
        &rns_matmul_residues_cuda,
        "Canonical uint8 RNS residue matrix multiplication (CUDA)");

    module.def(
        "matmul_centered_residues",
        &rns_matmul_centered_cuda,
        "Centered signed-int8 RNS GEMM: scalar, DP4A or cuBLAS backend (CUDA)");

    module.def(
        "decode_garner",
        &rns_decode_garner_cuda,
        "Single-kernel mixed-radix/Garner reconstruction for centered residues (CUDA)");

    module.def(
        "matmul_centered_fused_out",
        &rns_matmul_centered_fused_out_cuda,
        "cuBLAS INT8 GEMM plus fused modulo/Garner reconstruction into preallocated output (CUDA)");
}
