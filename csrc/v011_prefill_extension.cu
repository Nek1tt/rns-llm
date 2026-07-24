#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

#define CUBLASLT_CHECK(expr)                                                        \
    do {                                                                            \
        cublasStatus_t _status = (expr);                                             \
        TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS,                                \
                    "cuBLASLt call failed with status ", static_cast<int>(_status), \
                    " at ", __FILE__, ":", __LINE__);                             \
    } while (0)

inline void check_cuda_contiguous(const torch::Tensor& t, const char* name, int dim) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == dim, name, " must have ", dim, " dimensions");
}

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ float block_max(float value, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        float result = lane < ((blockDim.x + 31) >> 5) ? shared[lane] : 0.0f;
        result = warp_max(result);
        if (lane == 0) {
            shared[0] = result;
        }
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ int64_t quantize_float(float value, float scale, int64_t qmax) {
    if (!(scale > 0.0f)) {
        return 0;
    }
    int64_t q = static_cast<int64_t>(llrintf(value / scale));
    q = q > qmax ? qmax : q;
    q = q < -qmax ? -qmax : q;
    return q;
}

__device__ __forceinline__ int centered_residue(int64_t value, int modulus) {
    int64_t r = value % modulus;
    const int half = modulus >> 1;
    if (r > half) {
        r -= modulus;
    } else if (r < -half) {
        r += modulus;
    }
    return static_cast<int>(r);
}

__global__ void quantize_weight_masked_kernel(
    const float* __restrict__ weight_nk,
    const float* __restrict__ scales_n,
    const uint8_t* __restrict__ protected_mask_k,
    int8_t* __restrict__ weight_kn,
    int n,
    int k) {
    const int64_t total = static_cast<int64_t>(n) * k;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int row_n = static_cast<int>(idx / k);
    const int col_k = static_cast<int>(idx - static_cast<int64_t>(row_n) * k);
    int8_t q = 0;
    if (!protected_mask_k[col_k]) {
        q = static_cast<int8_t>(quantize_float(weight_nk[idx], scales_n[row_n], 127));
    }
    weight_kn[static_cast<int64_t>(col_k) * n + row_n] = q;
}

__global__ void quantize_weight_all_kernel(
    const float* __restrict__ weight_nk,
    const float* __restrict__ scales_n,
    int8_t* __restrict__ weight_kn,
    int n,
    int k) {
    const int64_t total = static_cast<int64_t>(n) * k;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int row_n = static_cast<int>(idx / k);
    const int col_k = static_cast<int>(idx - static_cast<int64_t>(row_n) * k);
    const int8_t q = static_cast<int8_t>(quantize_float(weight_nk[idx], scales_n[row_n], 127));
    weight_kn[static_cast<int64_t>(col_k) * n + row_n] = q;
}

__global__ void encode_protected_weight_kernel(
    const float* __restrict__ weight_np,
    const float* __restrict__ scales_n,
    const int32_t* __restrict__ moduli_c,
    int8_t* __restrict__ residues_cnp,
    int n,
    int p,
    int p_padded,
    int channels,
    int64_t qmax) {
    const int64_t total = static_cast<int64_t>(n) * p_padded;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int row_n = static_cast<int>(idx / p_padded);
    const int col_p = static_cast<int>(idx - static_cast<int64_t>(row_n) * p_padded);
    int64_t q = 0;
    if (col_p < p) {
        q = quantize_float(weight_np[static_cast<int64_t>(row_n) * p + col_p], scales_n[row_n], qmax);
    }
    for (int c = 0; c < channels; ++c) {
        residues_cnp[(static_cast<int64_t>(c) * n + row_n) * p_padded + col_p] =
            static_cast<int8_t>(centered_residue(q, moduli_c[c]));
    }
}

__global__ void quantize_rows_kernel(
    const float* __restrict__ input_mk,
    int8_t* __restrict__ quantized_mk,
    float* __restrict__ scales_m,
    int m,
    int k) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }
    float local_max = 0.0f;
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input_mk[static_cast<int64_t>(row) * k + col]));
    }
    const float max_value = block_max(local_max, shared);
    const float scale = fmaxf(max_value / 127.0f, 1.17549435e-38f);
    if (threadIdx.x == 0) {
        scales_m[row] = scale;
    }
    __syncthreads();
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        const int64_t idx = static_cast<int64_t>(row) * k + col;
        quantized_mk[idx] = static_cast<int8_t>(quantize_float(input_mk[idx], scale, 127));
    }
}

__global__ void fused_hybrid_preprocess_kernel(
    const float* __restrict__ input_mk,
    const uint8_t* __restrict__ protected_mask_k,
    const int32_t* __restrict__ protected_indices_p,
    const int32_t* __restrict__ moduli_c,
    int8_t* __restrict__ main_quantized_mk,
    float* __restrict__ main_scales_m,
    __half* __restrict__ protected_half_mp,
    int8_t* __restrict__ protected_residues_cmp,
    float* __restrict__ protected_scales_m,
    int m,
    int k,
    int p,
    int p_padded,
    int channels,
    int64_t protected_qmax) {
    extern __shared__ float shared[];
    float* safe_shared = shared;
    float* protected_shared = shared + 32;
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }

    float safe_local_max = 0.0f;
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        if (!protected_mask_k[col]) {
            safe_local_max = fmaxf(
                safe_local_max,
                fabsf(input_mk[static_cast<int64_t>(row) * k + col]));
        }
    }
    const float safe_max = block_max(safe_local_max, safe_shared);

    float protected_local_max = 0.0f;
    for (int i = threadIdx.x; i < p; i += blockDim.x) {
        const int col = protected_indices_p[i];
        protected_local_max = fmaxf(
            protected_local_max,
            fabsf(input_mk[static_cast<int64_t>(row) * k + col]));
    }
    const float protected_max = block_max(protected_local_max, protected_shared);

    const float safe_scale = fmaxf(safe_max / 127.0f, 1.17549435e-38f);
    const float protected_scale = fmaxf(
        protected_max / static_cast<float>(protected_qmax), 1.17549435e-38f);
    if (threadIdx.x == 0) {
        main_scales_m[row] = safe_scale;
        protected_scales_m[row] = protected_scale;
    }
    __syncthreads();

    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        const int64_t idx = static_cast<int64_t>(row) * k + col;
        main_quantized_mk[idx] = protected_mask_k[col]
            ? static_cast<int8_t>(0)
            : static_cast<int8_t>(quantize_float(input_mk[idx], safe_scale, 127));
    }

    for (int i = threadIdx.x; i < p_padded; i += blockDim.x) {
        float value = 0.0f;
        int64_t q = 0;
        if (i < p) {
            const int col = protected_indices_p[i];
            value = input_mk[static_cast<int64_t>(row) * k + col];
            q = quantize_float(value, protected_scale, protected_qmax);
            protected_half_mp[static_cast<int64_t>(row) * p_padded + i] = __float2half(value);
        } else {
            protected_half_mp[static_cast<int64_t>(row) * p_padded + i] = __float2half(0.0f);
        }
        for (int c = 0; c < channels; ++c) {
            protected_residues_cmp[(static_cast<int64_t>(c) * m + row) * p_padded + i] =
                static_cast<int8_t>(centered_residue(q, moduli_c[c]));
        }
    }
}


__global__ void fused_hybrid_preprocess_fp16_kernel(
    const float* __restrict__ input_mk,
    const uint8_t* __restrict__ protected_mask_k,
    const int32_t* __restrict__ protected_indices_p,
    int8_t* __restrict__ main_quantized_mk,
    float* __restrict__ main_scales_m,
    __half* __restrict__ protected_half_mp,
    int m,
    int k,
    int p,
    int p_padded) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    if (row >= m) return;

    float safe_local_max = 0.0f;
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        if (!protected_mask_k[col]) {
            safe_local_max = fmaxf(
                safe_local_max,
                fabsf(input_mk[static_cast<int64_t>(row) * k + col]));
        }
    }
    const float safe_max = block_max(safe_local_max, shared);
    const float safe_scale = fmaxf(safe_max / 127.0f, 1.17549435e-38f);
    if (threadIdx.x == 0) main_scales_m[row] = safe_scale;
    __syncthreads();

    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        const int64_t index = static_cast<int64_t>(row) * k + col;
        main_quantized_mk[index] = protected_mask_k[col]
            ? static_cast<int8_t>(0)
            : static_cast<int8_t>(quantize_float(input_mk[index], safe_scale, 127));
    }
    for (int i = threadIdx.x; i < p_padded; i += blockDim.x) {
        float value = 0.0f;
        if (i < p) {
            const int col = protected_indices_p[i];
            value = input_mk[static_cast<int64_t>(row) * k + col];
        }
        protected_half_mp[static_cast<int64_t>(row) * p_padded + i] = __float2half(value);
    }
}


__device__ __forceinline__ int canonical_mod_i32_lut_v014(
    int value,
    int modulus,
    const int16_t* __restrict__ table,
    int channel) {
    const uint32_t magnitude = value < 0
        ? static_cast<uint32_t>(-static_cast<int64_t>(value))
        : static_cast<uint32_t>(value);
    const int64_t base = static_cast<int64_t>(channel) * 4 * 256;
    int residue = 0;
#pragma unroll
    for (int byte_position = 0; byte_position < 4; ++byte_position) {
        const uint32_t byte_value = (magnitude >> (8 * byte_position)) & 0xFFU;
        residue += static_cast<int>(
            table[base + byte_position * 256 + byte_value]);
    }
#pragma unroll
    for (int correction = 0; correction < 4; ++correction) {
        if (residue >= modulus) residue -= modulus;
    }
    if (value < 0 && residue != 0) residue = modulus - residue;
    return residue;
}

__device__ __forceinline__ int positive_mod(int value, int modulus) {
    int r = value % modulus;
    return r < 0 ? r + modulus : r;
}

struct U128V014 {
    uint64_t lo;
    uint64_t hi;
};

__device__ __forceinline__ U128V014 u128_from_u64_v014(uint64_t value) {
    return U128V014{value, 0ULL};
}

__device__ __forceinline__ U128V014 u128_add_v014(U128V014 a, U128V014 b) {
    U128V014 out;
    out.lo = a.lo + b.lo;
    const uint64_t carry = out.lo < a.lo ? 1ULL : 0ULL;
    out.hi = a.hi + b.hi + carry;
    return out;
}

__device__ __forceinline__ U128V014 u128_sub_v014(U128V014 a, U128V014 b) {
    U128V014 out;
    const uint64_t borrow = a.lo < b.lo ? 1ULL : 0ULL;
    out.lo = a.lo - b.lo;
    out.hi = a.hi - b.hi - borrow;
    return out;
}

__device__ __forceinline__ int u128_compare_v014(U128V014 a, U128V014 b) {
    if (a.hi < b.hi) return -1;
    if (a.hi > b.hi) return 1;
    if (a.lo < b.lo) return -1;
    if (a.lo > b.lo) return 1;
    return 0;
}

__device__ __forceinline__ U128V014 u128_shift_right_one_v014(U128V014 value) {
    return U128V014{
        (value.lo >> 1) | (value.hi << 63),
        value.hi >> 1,
    };
}

__device__ __forceinline__ U128V014 u128_mul_small_v014(
    U128V014 value,
    uint32_t factor) {
    const uint64_t f = static_cast<uint64_t>(factor);
    const uint64_t lo = value.lo * f;
    const uint64_t carry = __umul64hi(value.lo, f);
    const uint64_t hi = value.hi * f + carry;
    return U128V014{lo, hi};
}

__device__ __forceinline__ uint32_t u128_mod_small_v014(
    U128V014 value,
    uint32_t modulus) {
    const uint32_t limbs[4] = {
        static_cast<uint32_t>(value.hi >> 32),
        static_cast<uint32_t>(value.hi),
        static_cast<uint32_t>(value.lo >> 32),
        static_cast<uint32_t>(value.lo),
    };
    uint64_t remainder = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        remainder = ((remainder << 32) + limbs[i]) % modulus;
    }
    return static_cast<uint32_t>(remainder);
}

__device__ __forceinline__ double u128_to_double_v014(U128V014 value) {
    constexpr double kTwo64 = 18446744073709551616.0;
    return static_cast<double>(value.hi) * kTwo64 + static_cast<double>(value.lo);
}

__device__ __forceinline__ double garner_signed_u128_v014(
    const int* residues,
    const int32_t* moduli,
    const int32_t* prefix_inverses,
    int channels) {
    U128V014 x = u128_from_u64_v014(
        static_cast<uint64_t>(positive_mod(residues[0], moduli[0])));
    U128V014 prefix = u128_from_u64_v014(static_cast<uint64_t>(moduli[0]));
    for (int c = 1; c < channels; ++c) {
        const uint32_t modulus = static_cast<uint32_t>(moduli[c]);
        const uint32_t target = static_cast<uint32_t>(positive_mod(residues[c], moduli[c]));
        const uint32_t x_mod = u128_mod_small_v014(x, modulus);
        int32_t delta = static_cast<int32_t>(target) - static_cast<int32_t>(x_mod);
        delta %= static_cast<int32_t>(modulus);
        if (delta < 0) delta += static_cast<int32_t>(modulus);
        const uint32_t digit = static_cast<uint32_t>(
            (static_cast<uint64_t>(static_cast<uint32_t>(delta)) *
             static_cast<uint64_t>(static_cast<uint32_t>(prefix_inverses[c]))) %
            static_cast<uint64_t>(modulus));
        x = u128_add_v014(x, u128_mul_small_v014(prefix, digit));
        prefix = u128_mul_small_v014(prefix, modulus);
    }
    const U128V014 half = u128_shift_right_one_v014(prefix);
    if (u128_compare_v014(x, half) > 0) {
        return -u128_to_double_v014(u128_sub_v014(prefix, x));
    }
    return u128_to_double_v014(x);
}

__global__ void rns_rankk_correction_kernel(
    const int8_t* __restrict__ activation_cmp,
    const int8_t* __restrict__ weight_cnp,
    const int32_t* __restrict__ moduli_c,
    const int32_t* __restrict__ prefix_inverses_c,
    const int16_t* __restrict__ compact_lut,
    const float* __restrict__ activation_scales_m,
    const float* __restrict__ weight_scales_n,
    float* __restrict__ correction_mn,
    int m,
    int n,
    int p_padded,
    int channels,
    int lut_channels) {
    const int64_t total = static_cast<int64_t>(m) * n;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<int64_t>(row) * n);
    int residues[10] = {0};
    #pragma unroll
    for (int c = 0; c < 10; ++c) {
        if (c >= channels) {
            break;
        }
        int acc = 0;
        const int8_t* a_bytes = activation_cmp + (static_cast<int64_t>(c) * m + row) * p_padded;
        const int8_t* w_bytes = weight_cnp + (static_cast<int64_t>(c) * n + col) * p_padded;
        const int* a4 = reinterpret_cast<const int*>(a_bytes);
        const int* w4 = reinterpret_cast<const int*>(w_bytes);
        for (int pack = 0; pack < p_padded / 4; ++pack) {
            acc = __dp4a(a4[pack], w4[pack], acc);
        }
        residues[c] = (c < lut_channels)
            ? canonical_mod_i32_lut_v014(acc, moduli_c[c], compact_lut, c)
            : positive_mod(acc, moduli_c[c]);
    }
    const double reconstructed = garner_signed_u128_v014(
        residues, moduli_c, prefix_inverses_c, channels);
    correction_mn[index] = static_cast<float>(
        reconstructed *
        static_cast<double>(activation_scales_m[row]) *
        static_cast<double>(weight_scales_n[col]));
}

__global__ void fp16_rankk_correction_kernel(
    const __half* __restrict__ activation_mp,
    const __half* __restrict__ weight_np,
    float* __restrict__ correction_mn,
    int m,
    int n,
    int p_padded) {
    const int64_t total = static_cast<int64_t>(m) * n;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<int64_t>(row) * n);
    const __half* a = activation_mp + static_cast<int64_t>(row) * p_padded;
    const __half* w = weight_np + static_cast<int64_t>(col) * p_padded;
    const __half2* a2 = reinterpret_cast<const __half2*>(a);
    const __half2* w2 = reinterpret_cast<const __half2*>(w);
    float acc = 0.0f;
    for (int pack = 0; pack < p_padded / 2; ++pack) {
        const float2 av = __half22float2(a2[pack]);
        const float2 wv = __half22float2(w2[pack]);
        acc = fmaf(av.x, wv.x, acc);
        acc = fmaf(av.y, wv.y, acc);
    }
    correction_mn[index] = acc;
}

__global__ void cast_fp32_to_fp16_kernel(
    const float* __restrict__ input,
    __half* __restrict__ output,
    int64_t total) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < total) {
        output[index] = __float2half(input[index]);
    }
}

__global__ void add_bias_kernel(
    float* __restrict__ matrix,
    const float* __restrict__ bias,
    int64_t total,
    int n) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < total) {
        matrix[index] += bias[index % n];
    }
}

__global__ void merge_epilogue_kernel(
    const int32_t* __restrict__ main_acc_mn,
    const float* __restrict__ main_activation_scales_m,
    const float* __restrict__ main_weight_scales_n,
    const float* __restrict__ correction_mn,
    const float* __restrict__ bias_n,
    float* __restrict__ output_mn,
    int64_t total,
    int n,
    bool has_bias) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<int64_t>(row) * n);
    float value = static_cast<float>(
        static_cast<double>(main_acc_mn[index]) *
        static_cast<double>(main_activation_scales_m[row]) *
        static_cast<double>(main_weight_scales_n[col]));
    if (correction_mn != nullptr) {
        value += correction_mn[index];
    }
    if (has_bias) {
        value += bias_n[col];
    }
    output_mn[index] = value;
}

__global__ void rns_fused_epilogue_kernel(
    const int32_t* __restrict__ main_acc_mn,
    const float* __restrict__ main_activation_scales_m,
    const float* __restrict__ main_weight_scales_n,
    const int8_t* __restrict__ activation_cmp,
    const int8_t* __restrict__ weight_cnp,
    const int32_t* __restrict__ moduli_c,
    const int32_t* __restrict__ prefix_inverses_c,
    const int16_t* __restrict__ compact_lut,
    const float* __restrict__ protected_activation_scales_m,
    const float* __restrict__ protected_weight_scales_n,
    const float* __restrict__ bias_n,
    float* __restrict__ output_mn,
    int m,
    int n,
    int p_padded,
    int channels,
    int lut_channels,
    bool has_bias) {
    const int64_t total = static_cast<int64_t>(m) * n;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<int64_t>(row) * n);
    int residues[10] = {0};
    #pragma unroll
    for (int c = 0; c < 10; ++c) {
        if (c >= channels) {
            break;
        }
        int acc = 0;
        const int8_t* a_bytes = activation_cmp + (static_cast<int64_t>(c) * m + row) * p_padded;
        const int8_t* w_bytes = weight_cnp + (static_cast<int64_t>(c) * n + col) * p_padded;
        const int* a4 = reinterpret_cast<const int*>(a_bytes);
        const int* w4 = reinterpret_cast<const int*>(w_bytes);
        for (int pack = 0; pack < p_padded / 4; ++pack) {
            acc = __dp4a(a4[pack], w4[pack], acc);
        }
        residues[c] = (c < lut_channels)
            ? canonical_mod_i32_lut_v014(acc, moduli_c[c], compact_lut, c)
            : positive_mod(acc, moduli_c[c]);
    }
    const double reconstructed = garner_signed_u128_v014(
        residues, moduli_c, prefix_inverses_c, channels);
    float value = static_cast<float>(
        static_cast<double>(main_acc_mn[index]) *
        static_cast<double>(main_activation_scales_m[row]) *
        static_cast<double>(main_weight_scales_n[col]));
    value += static_cast<float>(
        reconstructed *
        static_cast<double>(protected_activation_scales_m[row]) *
        static_cast<double>(protected_weight_scales_n[col]));
    if (has_bias) {
        value += bias_n[col];
    }
    output_mn[index] = value;
}

__global__ void fp16_fused_epilogue_kernel(
    const int32_t* __restrict__ main_acc_mn,
    const float* __restrict__ main_activation_scales_m,
    const float* __restrict__ main_weight_scales_n,
    const __half* __restrict__ activation_mp,
    const __half* __restrict__ weight_np,
    const float* __restrict__ bias_n,
    float* __restrict__ output_mn,
    int m,
    int n,
    int p_padded,
    bool has_bias) {
    const int64_t total = static_cast<int64_t>(m) * n;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<int64_t>(row) * n);
    const __half* a = activation_mp + static_cast<int64_t>(row) * p_padded;
    const __half* w = weight_np + static_cast<int64_t>(col) * p_padded;
    const __half2* a2 = reinterpret_cast<const __half2*>(a);
    const __half2* w2 = reinterpret_cast<const __half2*>(w);
    float correction = 0.0f;
    for (int pack = 0; pack < p_padded / 2; ++pack) {
        const float2 av = __half22float2(a2[pack]);
        const float2 wv = __half22float2(w2[pack]);
        correction = fmaf(av.x, wv.x, correction);
        correction = fmaf(av.y, wv.y, correction);
    }
    float value = static_cast<float>(
        static_cast<double>(main_acc_mn[index]) *
        static_cast<double>(main_activation_scales_m[row]) *
        static_cast<double>(main_weight_scales_n[col]));
    value += correction;
    if (has_bias) {
        value += bias_n[col];
    }
    output_mn[index] = value;
}

class LtPlanBase {
public:
    LtPlanBase(int m, int k, int n, size_t workspace_bytes)
        : m_(m), k_(k), n_(n), workspace_bytes_(workspace_bytes) {
        TORCH_CHECK(m > 0 && k > 0 && n > 0, "M,K,N must be positive");
        CUBLASLT_CHECK(cublasLtCreate(&handle_));
    }

    virtual ~LtPlanBase() {
        if (a_layout_) cublasLtMatrixLayoutDestroy(a_layout_);
        if (b_layout_) cublasLtMatrixLayoutDestroy(b_layout_);
        if (c_layout_) cublasLtMatrixLayoutDestroy(c_layout_);
        if (matmul_desc_) cublasLtMatmulDescDestroy(matmul_desc_);
        if (preference_) cublasLtMatmulPreferenceDestroy(preference_);
        if (handle_) cublasLtDestroy(handle_);
    }

    size_t workspace_bytes() const { return workspace_bytes_; }
    int m() const { return m_; }
    int k() const { return k_; }
    int n() const { return n_; }

protected:
    void set_row_major(cublasLtMatrixLayout_t layout) {
        cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    }

    void finish_algorithm_selection() {
        CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference_));
        CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes_, sizeof(workspace_bytes_)));
        cublasLtMatmulHeuristicResult_t result{};
        int returned = 0;
        CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
            handle_, matmul_desc_, a_layout_, b_layout_, c_layout_, c_layout_,
            preference_, 1, &result, &returned));
        TORCH_CHECK(returned > 0, "cuBLASLt did not find a compatible algorithm");
        algo_ = result.algo;
        selected_workspace_bytes_ = std::min(workspace_bytes_, result.workspaceSize);
    }

    cublasLtHandle_t handle_ = nullptr;
    cublasLtMatmulDesc_t matmul_desc_ = nullptr;
    cublasLtMatrixLayout_t a_layout_ = nullptr;
    cublasLtMatrixLayout_t b_layout_ = nullptr;
    cublasLtMatrixLayout_t c_layout_ = nullptr;
    cublasLtMatmulPreference_t preference_ = nullptr;
    cublasLtMatmulAlgo_t algo_{};
    int m_;
    int k_;
    int n_;
    size_t workspace_bytes_;
    size_t selected_workspace_bytes_ = 0;
};

class LtInt8Plan final : public LtPlanBase {
public:
    LtInt8Plan(int m, int k, int n, size_t workspace_bytes)
        : LtPlanBase(m, k, n, workspace_bytes) {
        TORCH_CHECK(k % 4 == 0 && n % 4 == 0,
                    "INT8 Tensor Core plan requires K and N multiples of 4");
        CUBLASLT_CHECK(cublasLtMatmulDescCreate(
            &matmul_desc_, CUBLAS_COMPUTE_32I, CUDA_R_32I));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_layout_, CUDA_R_8I, m, k, k));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_layout_, CUDA_R_8I, k, n, n));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_layout_, CUDA_R_32I, m, n, n));
        set_row_major(a_layout_);
        set_row_major(b_layout_);
        set_row_major(c_layout_);
        finish_algorithm_selection();
    }

    torch::Tensor run(
        torch::Tensor a,
        torch::Tensor b,
        torch::Tensor c,
        torch::Tensor workspace) {
        check_cuda_contiguous(a, "a", 2);
        check_cuda_contiguous(b, "b", 2);
        check_cuda_contiguous(c, "c", 2);
        check_cuda_contiguous(workspace, "workspace", 1);
        TORCH_CHECK(a.scalar_type() == torch::kInt8 && b.scalar_type() == torch::kInt8,
                    "a and b must be int8");
        TORCH_CHECK(c.scalar_type() == torch::kInt32, "c must be int32");
        TORCH_CHECK(workspace.scalar_type() == torch::kUInt8, "workspace must be uint8");
        TORCH_CHECK(a.size(0) == m_ && a.size(1) == k_, "a shape mismatch");
        TORCH_CHECK(b.size(0) == k_ && b.size(1) == n_, "b shape mismatch");
        TORCH_CHECK(c.size(0) == m_ && c.size(1) == n_, "c shape mismatch");
        TORCH_CHECK(static_cast<size_t>(workspace.numel()) >= selected_workspace_bytes_,
                    "workspace is too small");
        c10::cuda::CUDAGuard guard(a.device());
        const int32_t alpha = 1;
        const int32_t beta = 0;
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        CUBLASLT_CHECK(cublasLtMatmul(
            handle_, matmul_desc_, &alpha,
            a.data_ptr<int8_t>(), a_layout_,
            b.data_ptr<int8_t>(), b_layout_,
            &beta,
            c.data_ptr<int32_t>(), c_layout_,
            c.data_ptr<int32_t>(), c_layout_,
            &algo_, workspace.data_ptr<uint8_t>(), selected_workspace_bytes_, stream));
        return c;
    }
};

class LtFp16Plan final : public LtPlanBase {
public:
    LtFp16Plan(int m, int k, int n, size_t workspace_bytes)
        : LtPlanBase(m, k, n, workspace_bytes) {
        CUBLASLT_CHECK(cublasLtMatmulDescCreate(
            &matmul_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_layout_, CUDA_R_16F, m, k, k));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_layout_, CUDA_R_16F, k, n, n));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_layout_, CUDA_R_32F, m, n, n));
        set_row_major(a_layout_);
        set_row_major(b_layout_);
        set_row_major(c_layout_);
        finish_algorithm_selection();
    }

    torch::Tensor run(
        torch::Tensor a,
        torch::Tensor b,
        torch::Tensor c,
        torch::Tensor workspace) {
        check_cuda_contiguous(a, "a", 2);
        check_cuda_contiguous(b, "b", 2);
        check_cuda_contiguous(c, "c", 2);
        check_cuda_contiguous(workspace, "workspace", 1);
        TORCH_CHECK(a.scalar_type() == torch::kFloat16 && b.scalar_type() == torch::kFloat16,
                    "a and b must be float16");
        TORCH_CHECK(c.scalar_type() == torch::kFloat32, "c must be float32");
        TORCH_CHECK(workspace.scalar_type() == torch::kUInt8, "workspace must be uint8");
        TORCH_CHECK(a.size(0) == m_ && a.size(1) == k_, "a shape mismatch");
        TORCH_CHECK(b.size(0) == k_ && b.size(1) == n_, "b shape mismatch");
        TORCH_CHECK(c.size(0) == m_ && c.size(1) == n_, "c shape mismatch");
        TORCH_CHECK(static_cast<size_t>(workspace.numel()) >= selected_workspace_bytes_,
                    "workspace is too small");
        c10::cuda::CUDAGuard guard(a.device());
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        CUBLASLT_CHECK(cublasLtMatmul(
            handle_, matmul_desc_, &alpha,
            a.data_ptr<at::Half>(), a_layout_,
            b.data_ptr<at::Half>(), b_layout_,
            &beta,
            c.data_ptr<float>(), c_layout_,
            c.data_ptr<float>(), c_layout_,
            &algo_, workspace.data_ptr<uint8_t>(), selected_workspace_bytes_, stream));
        return c;
    }
};

void launch_1d(int64_t total, int& blocks, int& threads) {
    threads = 256;
    blocks = static_cast<int>((total + threads - 1) / threads);
}

torch::Tensor quantize_weight_masked_out(
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor protected_mask,
    torch::Tensor output) {
    check_cuda_contiguous(weight, "weight", 2);
    check_cuda_contiguous(scales, "scales", 1);
    check_cuda_contiguous(protected_mask, "protected_mask", 1);
    check_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(protected_mask.scalar_type() == torch::kUInt8, "protected_mask must be uint8");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8");
    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(scales.numel() == n && protected_mask.numel() == k, "metadata shape mismatch");
    TORCH_CHECK(output.size(0) == k && output.size(1) == n, "output must be [K,N]");
    c10::cuda::CUDAGuard guard(weight.device());
    int blocks, threads;
    launch_1d(weight.numel(), blocks, threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_weight_masked_kernel<<<blocks, threads, 0, stream>>>(
        weight.data_ptr<float>(), scales.data_ptr<float>(), protected_mask.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(), n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor quantize_weight_all_out(
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor output) {
    check_cuda_contiguous(weight, "weight", 2);
    check_cuda_contiguous(scales, "scales", 1);
    check_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8");
    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(scales.numel() == n, "scale shape mismatch");
    TORCH_CHECK(output.size(0) == k && output.size(1) == n, "output must be [K,N]");
    c10::cuda::CUDAGuard guard(weight.device());
    int blocks, threads;
    launch_1d(weight.numel(), blocks, threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_weight_all_kernel<<<blocks, threads, 0, stream>>>(
        weight.data_ptr<float>(), scales.data_ptr<float>(), output.data_ptr<int8_t>(), n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor encode_protected_weight_out(
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor moduli,
    int64_t quant_max,
    torch::Tensor output) {
    check_cuda_contiguous(weight, "weight", 2);
    check_cuda_contiguous(scales, "scales", 1);
    check_cuda_contiguous(moduli, "moduli", 1);
    check_cuda_contiguous(output, "output", 3);
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32, "moduli must be int32");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8");
    const int n = static_cast<int>(weight.size(0));
    const int p = static_cast<int>(weight.size(1));
    const int channels = static_cast<int>(moduli.numel());
    const int p_padded = static_cast<int>(output.size(2));
    TORCH_CHECK(output.size(0) == channels && output.size(1) == n,
                "output must be [C,N,Ppad]");
    TORCH_CHECK(p_padded >= p && p_padded % 4 == 0, "Ppad must cover P and be multiple of 4");
    TORCH_CHECK(channels >= 2 && channels <= 10, "optimized path supports 2..10 channels");
    c10::cuda::CUDAGuard guard(weight.device());
    int blocks, threads;
    launch_1d(static_cast<int64_t>(n) * p_padded, blocks, threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    encode_protected_weight_kernel<<<blocks, threads, 0, stream>>>(
        weight.data_ptr<float>(), scales.data_ptr<float>(), moduli.data_ptr<int32_t>(),
        output.data_ptr<int8_t>(), n, p, p_padded, channels, quant_max);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor quantize_rows_out(
    torch::Tensor input,
    torch::Tensor quantized,
    torch::Tensor scales) {
    check_cuda_contiguous(input, "input", 2);
    check_cuda_contiguous(quantized, "quantized", 2);
    check_cuda_contiguous(scales, "scales", 1);
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(quantized.scalar_type() == torch::kInt8, "quantized must be int8");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    const int m = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    TORCH_CHECK(quantized.sizes() == input.sizes() && scales.numel() == m,
                "output shape mismatch");
    c10::cuda::CUDAGuard guard(input.device());
    const int threads = 256;
    const int shared = 32 * sizeof(float);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_rows_kernel<<<m, threads, shared, stream>>>(
        input.data_ptr<float>(), quantized.data_ptr<int8_t>(), scales.data_ptr<float>(), m, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return quantized;
}

std::vector<torch::Tensor> fused_hybrid_preprocess_out(
    torch::Tensor input,
    torch::Tensor protected_mask,
    torch::Tensor protected_indices,
    torch::Tensor moduli,
    int64_t protected_qmax,
    torch::Tensor main_quantized,
    torch::Tensor main_scales,
    torch::Tensor protected_half,
    torch::Tensor protected_residues,
    torch::Tensor protected_scales) {
    check_cuda_contiguous(input, "input", 2);
    check_cuda_contiguous(protected_mask, "protected_mask", 1);
    check_cuda_contiguous(protected_indices, "protected_indices", 1);
    check_cuda_contiguous(moduli, "moduli", 1);
    check_cuda_contiguous(main_quantized, "main_quantized", 2);
    check_cuda_contiguous(main_scales, "main_scales", 1);
    check_cuda_contiguous(protected_half, "protected_half", 2);
    check_cuda_contiguous(protected_residues, "protected_residues", 3);
    check_cuda_contiguous(protected_scales, "protected_scales", 1);
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(protected_mask.scalar_type() == torch::kUInt8, "protected_mask must be uint8");
    TORCH_CHECK(protected_indices.scalar_type() == torch::kInt32, "protected_indices must be int32");
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32, "moduli must be int32");
    TORCH_CHECK(main_quantized.scalar_type() == torch::kInt8, "main_quantized must be int8");
    TORCH_CHECK(main_scales.scalar_type() == torch::kFloat32, "main_scales must be float32");
    TORCH_CHECK(protected_half.scalar_type() == torch::kFloat16, "protected_half must be float16");
    TORCH_CHECK(protected_residues.scalar_type() == torch::kInt8,
                "protected_residues must be int8");
    TORCH_CHECK(protected_scales.scalar_type() == torch::kFloat32,
                "protected_scales must be float32");
    const int m = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    const int p = static_cast<int>(protected_indices.numel());
    const int channels = static_cast<int>(moduli.numel());
    const int p_padded = static_cast<int>(protected_half.size(1));
    TORCH_CHECK(protected_mask.numel() == k, "mask length mismatch");
    TORCH_CHECK(main_quantized.sizes() == input.sizes() && main_scales.numel() == m,
                "main output shape mismatch");
    TORCH_CHECK(protected_half.size(0) == m && p_padded >= p && p_padded % 4 == 0,
                "protected_half shape mismatch");
    TORCH_CHECK(protected_residues.size(0) == channels &&
                protected_residues.size(1) == m && protected_residues.size(2) == p_padded,
                "protected residue shape mismatch");
    TORCH_CHECK(protected_scales.numel() == m, "protected scale shape mismatch");
    TORCH_CHECK(channels >= 2 && channels <= 10, "optimized path supports 2..10 channels");
    c10::cuda::CUDAGuard guard(input.device());
    const int threads = 256;
    const int shared = 64 * sizeof(float);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_hybrid_preprocess_kernel<<<m, threads, shared, stream>>>(
        input.data_ptr<float>(), protected_mask.data_ptr<uint8_t>(),
        protected_indices.data_ptr<int32_t>(), moduli.data_ptr<int32_t>(),
        main_quantized.data_ptr<int8_t>(), main_scales.data_ptr<float>(),
        reinterpret_cast<__half*>(protected_half.data_ptr<at::Half>()),
        protected_residues.data_ptr<int8_t>(), protected_scales.data_ptr<float>(),
        m, k, p, p_padded, channels, protected_qmax);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {main_quantized, main_scales, protected_half, protected_residues, protected_scales};
}


std::vector<torch::Tensor> fused_hybrid_preprocess_fp16_out(
    torch::Tensor input,
    torch::Tensor protected_mask,
    torch::Tensor protected_indices,
    torch::Tensor main_quantized,
    torch::Tensor main_scales,
    torch::Tensor protected_half) {
    check_cuda_contiguous(input, "input", 2);
    check_cuda_contiguous(protected_mask, "protected_mask", 1);
    check_cuda_contiguous(protected_indices, "protected_indices", 1);
    check_cuda_contiguous(main_quantized, "main_quantized", 2);
    check_cuda_contiguous(main_scales, "main_scales", 1);
    check_cuda_contiguous(protected_half, "protected_half", 2);
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(protected_mask.scalar_type() == torch::kUInt8, "protected_mask must be uint8");
    TORCH_CHECK(protected_indices.scalar_type() == torch::kInt32, "protected_indices must be int32");
    TORCH_CHECK(main_quantized.scalar_type() == torch::kInt8, "main_quantized must be int8");
    TORCH_CHECK(main_scales.scalar_type() == torch::kFloat32, "main_scales must be float32");
    TORCH_CHECK(protected_half.scalar_type() == torch::kFloat16, "protected_half must be float16");
    const int m = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    const int p = static_cast<int>(protected_indices.numel());
    const int p_padded = static_cast<int>(protected_half.size(1));
    TORCH_CHECK(protected_mask.numel() == k, "mask length mismatch");
    TORCH_CHECK(main_quantized.sizes() == input.sizes() && main_scales.numel() == m,
                "main output shape mismatch");
    TORCH_CHECK(protected_half.size(0) == m && p_padded >= p && p_padded % 4 == 0,
                "protected_half shape mismatch");
    c10::cuda::CUDAGuard guard(input.device());
    const int threads = 256;
    const int shared = 32 * sizeof(float);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_hybrid_preprocess_fp16_kernel<<<m, threads, shared, stream>>>(
        input.data_ptr<float>(), protected_mask.data_ptr<uint8_t>(),
        protected_indices.data_ptr<int32_t>(), main_quantized.data_ptr<int8_t>(),
        main_scales.data_ptr<float>(),
        reinterpret_cast<__half*>(protected_half.data_ptr<at::Half>()),
        m, k, p, p_padded);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {main_quantized, main_scales, protected_half};
}

torch::Tensor cast_fp32_to_fp16_out(torch::Tensor input, torch::Tensor output) {
    check_cuda_contiguous(input, "input", 2);
    check_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat16, "output must be float16");
    TORCH_CHECK(input.sizes() == output.sizes(), "shape mismatch");
    c10::cuda::CUDAGuard guard(input.device());
    int blocks, threads;
    launch_1d(input.numel(), blocks, threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cast_fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(), reinterpret_cast<__half*>(output.data_ptr<at::Half>()), input.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor add_bias_out(torch::Tensor matrix, torch::Tensor bias) {
    check_cuda_contiguous(matrix, "matrix", 2);
    check_cuda_contiguous(bias, "bias", 1);
    TORCH_CHECK(matrix.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32,
                "matrix and bias must be float32");
    const int n = static_cast<int>(matrix.size(1));
    TORCH_CHECK(bias.numel() == n, "bias shape mismatch");
    c10::cuda::CUDAGuard guard(matrix.device());
    int blocks, threads;
    launch_1d(matrix.numel(), blocks, threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    add_bias_kernel<<<blocks, threads, 0, stream>>>(
        matrix.data_ptr<float>(), bias.data_ptr<float>(), matrix.numel(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return matrix;
}

torch::Tensor dequant_epilogue_out(
    torch::Tensor main_acc,
    torch::Tensor main_activation_scales,
    torch::Tensor main_weight_scales,
    torch::Tensor bias,
    torch::Tensor output) {
    check_cuda_contiguous(main_acc, "main_acc", 2);
    check_cuda_contiguous(main_activation_scales, "main_activation_scales", 1);
    check_cuda_contiguous(main_weight_scales, "main_weight_scales", 1);
    check_cuda_contiguous(bias, "bias", 1);
    check_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(main_acc.scalar_type() == torch::kInt32, "main_acc must be int32");
    TORCH_CHECK(main_activation_scales.scalar_type() == torch::kFloat32 &&
                main_weight_scales.scalar_type() == torch::kFloat32,
                "scales must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32 && output.scalar_type() == torch::kFloat32,
                "bias/output must be float32");
    const int m = static_cast<int>(main_acc.size(0));
    const int n = static_cast<int>(main_acc.size(1));
    TORCH_CHECK(main_activation_scales.numel() == m && main_weight_scales.numel() == n,
                "scale shape mismatch");
    TORCH_CHECK(bias.numel() == 0 || bias.numel() == n, "bias shape mismatch");
    TORCH_CHECK(output.sizes() == main_acc.sizes(), "output shape mismatch");
    c10::cuda::CUDAGuard guard(main_acc.device());
    int blocks, threads;
    launch_1d(main_acc.numel(), blocks, threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    merge_epilogue_kernel<<<blocks, threads, 0, stream>>>(
        main_acc.data_ptr<int32_t>(), main_activation_scales.data_ptr<float>(),
        main_weight_scales.data_ptr<float>(), nullptr,
        bias.numel() ? bias.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        main_acc.numel(), n, bias.numel() == n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor rns_rankk_correction_out(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor moduli,
    torch::Tensor prefix_inverses,
    torch::Tensor compact_lut,
    int64_t lut_channels,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    torch::Tensor output) {
    check_cuda_contiguous(activation, "activation", 3);
    check_cuda_contiguous(weight, "weight", 3);
    check_cuda_contiguous(moduli, "moduli", 1);
    check_cuda_contiguous(prefix_inverses, "prefix_inverses", 1);
    check_cuda_contiguous(compact_lut, "compact_lut", 3);
    check_cuda_contiguous(activation_scales, "activation_scales", 1);
    check_cuda_contiguous(weight_scales, "weight_scales", 1);
    check_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(activation.scalar_type() == torch::kInt8 && weight.scalar_type() == torch::kInt8,
                "RNS operands must be int8");
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32 &&
                prefix_inverses.scalar_type() == torch::kInt32, "constants must be int32");
    TORCH_CHECK(compact_lut.scalar_type() == torch::kInt16, "compact_lut must be int16");
    TORCH_CHECK(activation_scales.scalar_type() == torch::kFloat32 &&
                weight_scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
    const int channels = static_cast<int>(activation.size(0));
    const int m = static_cast<int>(activation.size(1));
    const int p_padded = static_cast<int>(activation.size(2));
    const int n = static_cast<int>(weight.size(1));
    TORCH_CHECK(weight.size(0) == channels && weight.size(2) == p_padded,
                "weight must be [C,N,Ppad]");
    TORCH_CHECK(moduli.numel() == channels && prefix_inverses.numel() == channels,
                "constant length mismatch");
    TORCH_CHECK(activation_scales.numel() == m && weight_scales.numel() == n,
                "scale shape mismatch");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
    TORCH_CHECK(channels >= 2 && channels <= 10, "optimized path supports 2..10 channels");
    TORCH_CHECK(lut_channels >= 0 && lut_channels <= channels, "invalid lut_channels");
    TORCH_CHECK(compact_lut.size(0) == lut_channels && compact_lut.size(1) == 4 &&
                compact_lut.size(2) == 256, "compact_lut must be [lut_channels,4,256]");
    c10::cuda::CUDAGuard guard(activation.device());
    const int64_t total = static_cast<int64_t>(m) * n;
    int blocks, threads;
    launch_1d(total, blocks, threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    rns_rankk_correction_kernel<<<blocks, threads, 0, stream>>>(
        activation.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
        moduli.data_ptr<int32_t>(), prefix_inverses.data_ptr<int32_t>(),
        compact_lut.data_ptr<int16_t>(), activation_scales.data_ptr<float>(),
        weight_scales.data_ptr<float>(), output.data_ptr<float>(),
        m, n, p_padded, channels, static_cast<int>(lut_channels));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor fp16_rankk_correction_out(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor output) {
    check_cuda_contiguous(activation, "activation", 2);
    check_cuda_contiguous(weight, "weight", 2);
    check_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(activation.scalar_type() == torch::kFloat16 &&
                weight.scalar_type() == torch::kFloat16, "operands must be float16");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
    const int m = static_cast<int>(activation.size(0));
    const int p_padded = static_cast<int>(activation.size(1));
    const int n = static_cast<int>(weight.size(0));
    TORCH_CHECK(weight.size(1) == p_padded, "P mismatch");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
    c10::cuda::CUDAGuard guard(activation.device());
    const int64_t total = static_cast<int64_t>(m) * n;
    int blocks, threads;
    launch_1d(total, blocks, threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fp16_rankk_correction_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(activation.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        output.data_ptr<float>(), m, n, p_padded);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor merge_epilogue_out(
    torch::Tensor main_acc,
    torch::Tensor main_activation_scales,
    torch::Tensor main_weight_scales,
    torch::Tensor correction,
    torch::Tensor bias,
    torch::Tensor output) {
    check_cuda_contiguous(main_acc, "main_acc", 2);
    check_cuda_contiguous(main_activation_scales, "main_activation_scales", 1);
    check_cuda_contiguous(main_weight_scales, "main_weight_scales", 1);
    check_cuda_contiguous(correction, "correction", 2);
    check_cuda_contiguous(bias, "bias", 1);
    check_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(main_acc.scalar_type() == torch::kInt32, "main_acc must be int32");
    TORCH_CHECK(main_activation_scales.scalar_type() == torch::kFloat32 &&
                main_weight_scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(correction.scalar_type() == torch::kFloat32, "correction must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
    const int m = static_cast<int>(main_acc.size(0));
    const int n = static_cast<int>(main_acc.size(1));
    TORCH_CHECK(correction.sizes() == main_acc.sizes() && output.sizes() == main_acc.sizes(),
                "matrix shape mismatch");
    TORCH_CHECK(main_activation_scales.numel() == m && main_weight_scales.numel() == n,
                "scale shape mismatch");
    TORCH_CHECK(bias.numel() == 0 || bias.numel() == n, "bias shape mismatch");
    c10::cuda::CUDAGuard guard(main_acc.device());
    const int64_t total = static_cast<int64_t>(m) * n;
    int blocks, threads;
    launch_1d(total, blocks, threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    merge_epilogue_kernel<<<blocks, threads, 0, stream>>>(
        main_acc.data_ptr<int32_t>(), main_activation_scales.data_ptr<float>(),
        main_weight_scales.data_ptr<float>(), correction.data_ptr<float>(),
        bias.numel() ? bias.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        total, n, bias.numel() == n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor rns_fused_epilogue_out(
    torch::Tensor main_acc,
    torch::Tensor main_activation_scales,
    torch::Tensor main_weight_scales,
    torch::Tensor protected_activation,
    torch::Tensor protected_weight,
    torch::Tensor moduli,
    torch::Tensor prefix_inverses,
    torch::Tensor compact_lut,
    int64_t lut_channels,
    torch::Tensor protected_activation_scales,
    torch::Tensor protected_weight_scales,
    torch::Tensor bias,
    torch::Tensor output) {
    check_cuda_contiguous(main_acc, "main_acc", 2);
    check_cuda_contiguous(main_activation_scales, "main_activation_scales", 1);
    check_cuda_contiguous(main_weight_scales, "main_weight_scales", 1);
    check_cuda_contiguous(protected_activation, "protected_activation", 3);
    check_cuda_contiguous(protected_weight, "protected_weight", 3);
    check_cuda_contiguous(moduli, "moduli", 1);
    check_cuda_contiguous(prefix_inverses, "prefix_inverses", 1);
    check_cuda_contiguous(compact_lut, "compact_lut", 3);
    check_cuda_contiguous(protected_activation_scales, "protected_activation_scales", 1);
    check_cuda_contiguous(protected_weight_scales, "protected_weight_scales", 1);
    check_cuda_contiguous(bias, "bias", 1);
    check_cuda_contiguous(output, "output", 2);
    const int m = static_cast<int>(main_acc.size(0));
    const int n = static_cast<int>(main_acc.size(1));
    const int channels = static_cast<int>(protected_activation.size(0));
    const int p_padded = static_cast<int>(protected_activation.size(2));
    TORCH_CHECK(main_acc.scalar_type() == torch::kInt32, "main_acc must be int32");
    TORCH_CHECK(main_activation_scales.scalar_type() == torch::kFloat32 &&
                main_weight_scales.scalar_type() == torch::kFloat32 &&
                protected_activation_scales.scalar_type() == torch::kFloat32 &&
                protected_weight_scales.scalar_type() == torch::kFloat32,
                "scales must be float32");
    TORCH_CHECK(protected_activation.scalar_type() == torch::kInt8 &&
                protected_weight.scalar_type() == torch::kInt8, "RNS operands must be int8");
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32 &&
                prefix_inverses.scalar_type() == torch::kInt32, "constants must be int32");
    TORCH_CHECK(compact_lut.scalar_type() == torch::kInt16, "compact_lut must be int16");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32 && output.scalar_type() == torch::kFloat32,
                "bias/output must be float32");
    TORCH_CHECK(protected_activation.size(1) == m &&
                protected_weight.size(0) == channels && protected_weight.size(1) == n &&
                protected_weight.size(2) == p_padded, "protected shape mismatch");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
    TORCH_CHECK(channels >= 2 && channels <= 10, "optimized path supports 2..10 channels");
    TORCH_CHECK(lut_channels >= 0 && lut_channels <= channels, "invalid lut_channels");
    TORCH_CHECK(compact_lut.size(0) == lut_channels && compact_lut.size(1) == 4 &&
                compact_lut.size(2) == 256, "compact_lut must be [lut_channels,4,256]");
    c10::cuda::CUDAGuard guard(main_acc.device());
    const int64_t total = static_cast<int64_t>(m) * n;
    int blocks, threads;
    launch_1d(total, blocks, threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    rns_fused_epilogue_kernel<<<blocks, threads, 0, stream>>>(
        main_acc.data_ptr<int32_t>(), main_activation_scales.data_ptr<float>(),
        main_weight_scales.data_ptr<float>(), protected_activation.data_ptr<int8_t>(),
        protected_weight.data_ptr<int8_t>(), moduli.data_ptr<int32_t>(),
        prefix_inverses.data_ptr<int32_t>(), compact_lut.data_ptr<int16_t>(),
        protected_activation_scales.data_ptr<float>(),
        protected_weight_scales.data_ptr<float>(), bias.numel() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), m, n, p_padded, channels,
        static_cast<int>(lut_channels), bias.numel() == n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor fp16_fused_epilogue_out(
    torch::Tensor main_acc,
    torch::Tensor main_activation_scales,
    torch::Tensor main_weight_scales,
    torch::Tensor protected_activation,
    torch::Tensor protected_weight,
    torch::Tensor bias,
    torch::Tensor output) {
    check_cuda_contiguous(main_acc, "main_acc", 2);
    check_cuda_contiguous(main_activation_scales, "main_activation_scales", 1);
    check_cuda_contiguous(main_weight_scales, "main_weight_scales", 1);
    check_cuda_contiguous(protected_activation, "protected_activation", 2);
    check_cuda_contiguous(protected_weight, "protected_weight", 2);
    check_cuda_contiguous(bias, "bias", 1);
    check_cuda_contiguous(output, "output", 2);
    const int m = static_cast<int>(main_acc.size(0));
    const int n = static_cast<int>(main_acc.size(1));
    const int p_padded = static_cast<int>(protected_activation.size(1));
    TORCH_CHECK(main_acc.scalar_type() == torch::kInt32, "main_acc must be int32");
    TORCH_CHECK(main_activation_scales.scalar_type() == torch::kFloat32 &&
                main_weight_scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(protected_activation.scalar_type() == torch::kFloat16 &&
                protected_weight.scalar_type() == torch::kFloat16, "protected operands must be fp16");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32 && output.scalar_type() == torch::kFloat32,
                "bias/output must be float32");
    TORCH_CHECK(protected_weight.size(0) == n && protected_weight.size(1) == p_padded,
                "protected weight must be [N,Ppad]");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
    c10::cuda::CUDAGuard guard(main_acc.device());
    const int64_t total = static_cast<int64_t>(m) * n;
    int blocks, threads;
    launch_1d(total, blocks, threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fp16_fused_epilogue_kernel<<<blocks, threads, 0, stream>>>(
        main_acc.data_ptr<int32_t>(), main_activation_scales.data_ptr<float>(),
        main_weight_scales.data_ptr<float>(),
        reinterpret_cast<const __half*>(protected_activation.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(protected_weight.data_ptr<at::Half>()),
        bias.numel() ? bias.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        m, n, p_padded, bias.numel() == n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<LtInt8Plan, std::shared_ptr<LtInt8Plan>>(m, "LtInt8Plan")
        .def(py::init<int, int, int, size_t>())
        .def("run", &LtInt8Plan::run)
        .def_property_readonly("workspace_bytes", &LtInt8Plan::workspace_bytes)
        .def_property_readonly("m", &LtInt8Plan::m)
        .def_property_readonly("k", &LtInt8Plan::k)
        .def_property_readonly("n", &LtInt8Plan::n);
    py::class_<LtFp16Plan, std::shared_ptr<LtFp16Plan>>(m, "LtFp16Plan")
        .def(py::init<int, int, int, size_t>())
        .def("run", &LtFp16Plan::run)
        .def_property_readonly("workspace_bytes", &LtFp16Plan::workspace_bytes)
        .def_property_readonly("m", &LtFp16Plan::m)
        .def_property_readonly("k", &LtFp16Plan::k)
        .def_property_readonly("n", &LtFp16Plan::n);
    m.def("quantize_weight_masked_out", &quantize_weight_masked_out);
    m.def("quantize_weight_all_out", &quantize_weight_all_out);
    m.def("encode_protected_weight_out", &encode_protected_weight_out);
    m.def("quantize_rows_out", &quantize_rows_out);
    m.def("fused_hybrid_preprocess_out", &fused_hybrid_preprocess_out);
    m.def("fused_hybrid_preprocess_fp16_out", &fused_hybrid_preprocess_fp16_out);
    m.def("cast_fp32_to_fp16_out", &cast_fp32_to_fp16_out);
    m.def("add_bias_out", &add_bias_out);
    m.def("dequant_epilogue_out", &dequant_epilogue_out);
    m.def("rns_rankk_correction_out", &rns_rankk_correction_out);
    m.def("fp16_rankk_correction_out", &fp16_rankk_correction_out);
    m.def("merge_epilogue_out", &merge_epilogue_out);
    m.def("rns_fused_epilogue_out", &rns_fused_epilogue_out);
    m.def("fp16_fused_epilogue_out", &fp16_fused_epilogue_out);
}
