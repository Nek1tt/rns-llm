#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kMaxChannels = 20;
constexpr int kThreads = 256;

inline void check_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name,
    int rank) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == rank, name, " has unexpected rank");
}

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
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

__device__ __forceinline__ std::int64_t quantize_float_to_i64(
    float value,
    float scale,
    std::int64_t qmax) {
    const double scaled = static_cast<double>(value) / static_cast<double>(scale);
    double rounded = nearbyint(scaled);
    const double hi = static_cast<double>(qmax);
    if (rounded > hi) {
        rounded = hi;
    } else if (rounded < -hi) {
        rounded = -hi;
    }
    return static_cast<std::int64_t>(rounded);
}

__device__ __forceinline__ std::int8_t centered_residue_i64(
    std::int64_t value,
    int modulus) {
    std::int64_t r = value % static_cast<std::int64_t>(modulus);
    const int half = modulus / 2;
    if (r > half) {
        r -= modulus;
    } else if (r < -half) {
        r += modulus;
    }
    return static_cast<std::int8_t>(r);
}

__global__ void quantize_rows_int8_kernel(
    const float* __restrict__ input,
    std::int8_t* __restrict__ quantized,
    float* __restrict__ scales,
    int rows,
    int cols) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float local_max = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[static_cast<std::int64_t>(row) * cols + col]));
    }
    const float max_value = block_max(local_max, shared);
    const float scale = fmaxf(max_value / 127.0f, std::numeric_limits<float>::min());
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    __syncthreads();
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const std::int64_t index = static_cast<std::int64_t>(row) * cols + col;
        quantized[index] = static_cast<std::int8_t>(
            quantize_float_to_i64(input[index], scale, 127));
    }
}

__global__ void quantize_encode_rows_kernel(
    const float* __restrict__ input,
    const std::int32_t* __restrict__ moduli,
    std::int8_t* __restrict__ residues,
    float* __restrict__ scales,
    int rows,
    int cols,
    int channels,
    std::int64_t qmax) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float local_max = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[static_cast<std::int64_t>(row) * cols + col]));
    }
    const float max_value = block_max(local_max, shared);
    const float scale = fmaxf(
        max_value / static_cast<float>(static_cast<double>(qmax)),
        std::numeric_limits<float>::min());
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    __syncthreads();

    const std::int64_t plane = static_cast<std::int64_t>(rows) * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const std::int64_t input_index = static_cast<std::int64_t>(row) * cols + col;
        const std::int64_t q = quantize_float_to_i64(input[input_index], scale, qmax);
        for (int channel = 0; channel < channels; ++channel) {
            residues[static_cast<std::int64_t>(channel) * plane + input_index] =
                centered_residue_i64(q, moduli[channel]);
        }
    }
}

__device__ __forceinline__ std::uint32_t fast_mod_u32(
    std::uint32_t value,
    std::uint32_t modulus,
    std::uint32_t reciprocal) {
    if ((modulus & (modulus - 1U)) == 0U) {
        return value & (modulus - 1U);
    }
    const std::uint32_t quotient = __umulhi(value, reciprocal);
    std::uint32_t remainder = value - quotient * modulus;
    if (remainder >= modulus) {
        remainder -= modulus;
    }
    if (remainder >= modulus) {
        remainder -= modulus;
    }
    return remainder;
}

__device__ __forceinline__ std::int32_t canonical_mod_i32_barrett(
    std::int32_t value,
    std::int32_t modulus,
    std::uint32_t reciprocal) {
    const std::uint32_t magnitude = value < 0
        ? static_cast<std::uint32_t>(-static_cast<std::int64_t>(value))
        : static_cast<std::uint32_t>(value);
    std::uint32_t residue = fast_mod_u32(
        magnitude,
        static_cast<std::uint32_t>(modulus),
        reciprocal);
    if (value < 0 && residue != 0U) {
        residue = static_cast<std::uint32_t>(modulus) - residue;
    }
    return static_cast<std::int32_t>(residue);
}

__device__ __forceinline__ std::int32_t canonical_mod_i32_lut(
    std::int32_t value,
    std::int32_t modulus,
    const std::int16_t* __restrict__ table,
    int table_channel) {
    const std::uint32_t magnitude = value < 0
        ? static_cast<std::uint32_t>(-static_cast<std::int64_t>(value))
        : static_cast<std::uint32_t>(value);
    const std::int64_t base = static_cast<std::int64_t>(table_channel) * 4 * 256;
    std::int32_t residue = 0;
#pragma unroll
    for (int byte_position = 0; byte_position < 4; ++byte_position) {
        const std::uint32_t byte_value =
            (magnitude >> (8 * byte_position)) & 0xFFU;
        residue += static_cast<std::int32_t>(
            table[base + byte_position * 256 + byte_value]);
    }
#pragma unroll
    for (int correction = 0; correction < 4; ++correction) {
        if (residue >= modulus) {
            residue -= modulus;
        }
    }
    if (value < 0 && residue != 0) {
        residue = modulus - residue;
    }
    return residue;
}

struct U128 {
    std::uint64_t lo;
    std::uint64_t hi;
};

__device__ __forceinline__ U128 u128_from_u64(std::uint64_t value) {
    return U128{value, 0ULL};
}

__device__ __forceinline__ U128 u128_add(U128 a, U128 b) {
    U128 out;
    out.lo = a.lo + b.lo;
    const std::uint64_t carry = out.lo < a.lo ? 1ULL : 0ULL;
    out.hi = a.hi + b.hi + carry;
    return out;
}

__device__ __forceinline__ U128 u128_sub(U128 a, U128 b) {
    U128 out;
    const std::uint64_t borrow = a.lo < b.lo ? 1ULL : 0ULL;
    out.lo = a.lo - b.lo;
    out.hi = a.hi - b.hi - borrow;
    return out;
}

__device__ __forceinline__ int u128_compare(U128 a, U128 b) {
    if (a.hi < b.hi) return -1;
    if (a.hi > b.hi) return 1;
    if (a.lo < b.lo) return -1;
    if (a.lo > b.lo) return 1;
    return 0;
}

__device__ __forceinline__ U128 u128_shift_right_one(U128 value) {
    return U128{
        (value.lo >> 1) | (value.hi << 63),
        value.hi >> 1,
    };
}

__device__ __forceinline__ U128 u128_mul_small(U128 value, std::uint32_t factor) {
    const std::uint64_t f = static_cast<std::uint64_t>(factor);
    const std::uint64_t lo = value.lo * f;
    const std::uint64_t carry = __umul64hi(value.lo, f);
    const std::uint64_t hi = value.hi * f + carry;
    return U128{lo, hi};
}

__device__ __forceinline__ std::uint32_t u128_mod_small(U128 value, std::uint32_t modulus) {
    const std::uint32_t limbs[4] = {
        static_cast<std::uint32_t>(value.hi >> 32),
        static_cast<std::uint32_t>(value.hi),
        static_cast<std::uint32_t>(value.lo >> 32),
        static_cast<std::uint32_t>(value.lo),
    };
    std::uint64_t remainder = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        remainder = ((remainder << 32) + limbs[i]) % modulus;
    }
    return static_cast<std::uint32_t>(remainder);
}

__device__ __forceinline__ double u128_to_double(U128 value) {
    constexpr double two64 = 18446744073709551616.0;
    return static_cast<double>(value.hi) * two64 + static_cast<double>(value.lo);
}

__device__ __forceinline__ double garner_reconstruct_signed_double(
    const std::int32_t* canonical,
    const std::int32_t* moduli,
    const std::int32_t* prefix_inverses,
    int channels) {
    U128 x = u128_from_u64(static_cast<std::uint64_t>(canonical[0]));
    U128 prefix = u128_from_u64(static_cast<std::uint64_t>(moduli[0]));
    for (int channel = 1; channel < channels; ++channel) {
        const std::uint32_t modulus = static_cast<std::uint32_t>(moduli[channel]);
        const std::uint32_t target = static_cast<std::uint32_t>(canonical[channel]);
        const std::uint32_t x_mod = u128_mod_small(x, modulus);
        std::int32_t delta = static_cast<std::int32_t>(target) - static_cast<std::int32_t>(x_mod);
        delta %= static_cast<std::int32_t>(modulus);
        if (delta < 0) {
            delta += static_cast<std::int32_t>(modulus);
        }
        const std::uint32_t digit = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(delta)) *
             static_cast<std::uint64_t>(static_cast<std::uint32_t>(prefix_inverses[channel]))) %
            static_cast<std::uint64_t>(modulus));
        x = u128_add(x, u128_mul_small(prefix, digit));
        prefix = u128_mul_small(prefix, modulus);
    }
    const U128 half = u128_shift_right_one(prefix);
    if (u128_compare(x, half) > 0) {
        const U128 magnitude = u128_sub(prefix, x);
        return -u128_to_double(magnitude);
    }
    return u128_to_double(x);
}

__global__ void rns_reduce_garner_dequant_kernel(
    const std::int32_t* __restrict__ accumulators,
    const std::int32_t* __restrict__ moduli,
    const std::int64_t* __restrict__ reciprocals,
    const std::int32_t* __restrict__ prefix_inverses,
    const std::int16_t* __restrict__ compact_lut,
    const float* __restrict__ activation_scales,
    const float* __restrict__ weight_scales,
    float* __restrict__ output,
    std::int64_t elements,
    int n,
    int channels,
    int lut_channels) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    std::int32_t canonical[kMaxChannels];
    for (int channel = 0; channel < channels; ++channel) {
        const std::int32_t value = accumulators[
            static_cast<std::int64_t>(channel) * elements + index];
        if (channel < lut_channels) {
            canonical[channel] = canonical_mod_i32_lut(
                value, moduli[channel], compact_lut, channel);
        } else {
            canonical[channel] = canonical_mod_i32_barrett(
                value,
                moduli[channel],
                static_cast<std::uint32_t>(reciprocals[channel]));
        }
    }
    const double reconstructed = garner_reconstruct_signed_double(
        canonical, moduli, prefix_inverses, channels);
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<std::int64_t>(row) * n);
    const double scaled = reconstructed *
        static_cast<double>(activation_scales[row]) *
        static_cast<double>(weight_scales[col]);
    output[index] = static_cast<float>(scaled);
}

__global__ void dequant_int32_fp32_kernel(
    const std::int32_t* __restrict__ accumulators,
    const float* __restrict__ activation_scales,
    const float* __restrict__ weight_scales,
    float* __restrict__ output,
    std::int64_t elements,
    int n) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<std::int64_t>(row) * n);
    output[index] = static_cast<float>(accumulators[index]) *
        activation_scales[row] * weight_scales[col];
}

void set_cublas_stream_and_host_pointer_mode(
    cublasHandle_t handle,
    cudaStream_t stream,
    cublasPointerMode_t* previous_pointer_mode) {
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
    TORCH_CUDABLAS_CHECK(cublasGetPointerMode(handle, previous_pointer_mode));
    TORCH_CUDABLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
}

void run_cublas_int8_batched(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& accumulators,
    int channels,
    int m,
    int k,
    int n,
    cudaStream_t stream) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasPointerMode_t previous_pointer_mode;
    set_cublas_stream_and_host_pointer_mode(handle, stream, &previous_pointer_mode);
    const std::int32_t alpha = 1;
    const std::int32_t beta = 0;
    const cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b.data_ptr<std::int8_t>(),
        CUDA_R_8I,
        n,
        static_cast<long long>(k) * n,
        a.data_ptr<std::int8_t>(),
        CUDA_R_8I,
        k,
        static_cast<long long>(m) * k,
        &beta,
        accumulators.data_ptr<std::int32_t>(),
        CUDA_R_32I,
        n,
        static_cast<long long>(m) * n,
        channels,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CUDABLAS_CHECK(cublasSetPointerMode(handle, previous_pointer_mode));
    TORCH_CUDABLAS_CHECK(status);
}

void run_cublas_int8(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& accumulators,
    int m,
    int k,
    int n,
    cudaStream_t stream) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasPointerMode_t previous_pointer_mode;
    set_cublas_stream_and_host_pointer_mode(handle, stream, &previous_pointer_mode);
    const std::int32_t alpha = 1;
    const std::int32_t beta = 0;
    const cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b.data_ptr<std::int8_t>(),
        CUDA_R_8I,
        n,
        a.data_ptr<std::int8_t>(),
        CUDA_R_8I,
        k,
        &beta,
        accumulators.data_ptr<std::int32_t>(),
        CUDA_R_32I,
        n,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CUDABLAS_CHECK(cublasSetPointerMode(handle, previous_pointer_mode));
    TORCH_CUDABLAS_CHECK(status);
}

}  // namespace

std::vector<torch::Tensor> quantize_rows_int8_out(
    torch::Tensor input,
    torch::Tensor quantized,
    torch::Tensor scales) {
    check_cuda_contiguous(input, "input", 2);
    check_cuda_contiguous(quantized, "quantized", 2);
    check_cuda_contiguous(scales, "scales", 1);
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(quantized.scalar_type() == torch::kInt8, "quantized must be int8");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(input.sizes() == quantized.sizes(), "quantized shape mismatch");
    TORCH_CHECK(scales.numel() == input.size(0), "scales length mismatch");
    TORCH_CHECK(input.device() == quantized.device() && input.device() == scales.device(),
                "all tensors must share a device");
    c10::cuda::CUDAGuard guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_rows_int8_kernel<<<rows, kThreads, 32 * sizeof(float), stream>>>(
        input.data_ptr<float>(), quantized.data_ptr<std::int8_t>(), scales.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {quantized, scales};
}

std::vector<torch::Tensor> quantize_encode_rows_out(
    torch::Tensor input,
    torch::Tensor moduli,
    std::int64_t qmax,
    torch::Tensor residues,
    torch::Tensor scales) {
    check_cuda_contiguous(input, "input", 2);
    check_cuda_contiguous(moduli, "moduli", 1);
    check_cuda_contiguous(residues, "residues", 3);
    check_cuda_contiguous(scales, "scales", 1);
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32, "moduli must be int32");
    TORCH_CHECK(residues.scalar_type() == torch::kInt8, "residues must be int8");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(qmax > 0 && qmax <= 2147483647LL, "qmax must be in 1..2^31-1");
    const int channels = static_cast<int>(moduli.numel());
    TORCH_CHECK(channels >= 2 && channels <= kMaxChannels, "channel count must be 2..20");
    TORCH_CHECK(residues.size(0) == channels && residues.size(1) == input.size(0) &&
                residues.size(2) == input.size(1), "residue shape mismatch");
    TORCH_CHECK(scales.numel() == input.size(0), "scales length mismatch");
    TORCH_CHECK(input.device() == moduli.device() && input.device() == residues.device() &&
                input.device() == scales.device(), "all tensors must share a device");
    c10::cuda::CUDAGuard guard(input.device());
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_encode_rows_kernel<<<rows, kThreads, 32 * sizeof(float), stream>>>(
        input.data_ptr<float>(),
        moduli.data_ptr<std::int32_t>(),
        residues.data_ptr<std::int8_t>(),
        scales.data_ptr<float>(),
        rows,
        cols,
        channels,
        qmax);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {residues, scales};
}

torch::Tensor native_int8_mm_dequant_out(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    torch::Tensor accumulators,
    torch::Tensor output) {
    check_cuda_contiguous(a, "a", 2);
    check_cuda_contiguous(b, "b", 2);
    check_cuda_contiguous(activation_scales, "activation_scales", 1);
    check_cuda_contiguous(weight_scales, "weight_scales", 1);
    check_cuda_contiguous(accumulators, "accumulators", 2);
    check_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(a.scalar_type() == torch::kInt8 && b.scalar_type() == torch::kInt8,
                "a and b must be int8");
    TORCH_CHECK(activation_scales.scalar_type() == torch::kFloat32 &&
                weight_scales.scalar_type() == torch::kFloat32,
                "scales must be float32");
    TORCH_CHECK(accumulators.scalar_type() == torch::kInt32, "accumulators must be int32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(a.size(1) == b.size(0), "K dimension mismatch");
    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));
    TORCH_CHECK(k % 4 == 0 && n % 4 == 0, "INT8 GEMM requires K and N multiples of 4");
    TORCH_CHECK(activation_scales.numel() == m && weight_scales.numel() == n,
                "scale shape mismatch");
    TORCH_CHECK(accumulators.size(0) == m && accumulators.size(1) == n,
                "accumulator shape mismatch");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
    TORCH_CHECK(a.device() == b.device() && a.device() == activation_scales.device() &&
                a.device() == weight_scales.device() && a.device() == accumulators.device() &&
                a.device() == output.device(), "all tensors must share a device");
    c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_cublas_int8(a, b, accumulators, m, k, n, stream);
    const std::int64_t elements = static_cast<std::int64_t>(m) * n;
    const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
    dequant_int32_fp32_kernel<<<blocks, kThreads, 0, stream>>>(
        accumulators.data_ptr<std::int32_t>(),
        activation_scales.data_ptr<float>(),
        weight_scales.data_ptr<float>(),
        output.data_ptr<float>(),
        elements,
        n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor rns_mm_dequant_out(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor moduli,
    torch::Tensor reciprocals,
    torch::Tensor prefix_inverses,
    torch::Tensor compact_lut,
    std::int64_t lut_channels,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    torch::Tensor accumulators,
    torch::Tensor output) {
    check_cuda_contiguous(a, "a", 3);
    check_cuda_contiguous(b, "b", 3);
    check_cuda_contiguous(moduli, "moduli", 1);
    check_cuda_contiguous(reciprocals, "reciprocals", 1);
    check_cuda_contiguous(prefix_inverses, "prefix_inverses", 1);
    check_cuda_contiguous(compact_lut, "compact_lut", 3);
    check_cuda_contiguous(activation_scales, "activation_scales", 1);
    check_cuda_contiguous(weight_scales, "weight_scales", 1);
    check_cuda_contiguous(accumulators, "accumulators", 3);
    check_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(a.scalar_type() == torch::kInt8 && b.scalar_type() == torch::kInt8,
                "a and b must be int8 residue planes");
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32, "moduli must be int32");
    TORCH_CHECK(reciprocals.scalar_type() == torch::kInt64, "reciprocals must be int64");
    TORCH_CHECK(prefix_inverses.scalar_type() == torch::kInt32, "prefix_inverses must be int32");
    TORCH_CHECK(compact_lut.scalar_type() == torch::kInt16, "compact_lut must be int16");
    TORCH_CHECK(activation_scales.scalar_type() == torch::kFloat32 &&
                weight_scales.scalar_type() == torch::kFloat32,
                "scales must be float32");
    TORCH_CHECK(accumulators.scalar_type() == torch::kInt32, "accumulators must be int32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
    const int channels = static_cast<int>(a.size(0));
    TORCH_CHECK(channels >= 2 && channels <= kMaxChannels, "channel count must be 2..20");
    TORCH_CHECK(b.size(0) == channels && moduli.numel() == channels &&
                reciprocals.numel() == channels && prefix_inverses.numel() == channels,
                "channel metadata mismatch");
    TORCH_CHECK(a.size(2) == b.size(1), "K dimension mismatch");
    const int m = static_cast<int>(a.size(1));
    const int k = static_cast<int>(a.size(2));
    const int n = static_cast<int>(b.size(2));
    TORCH_CHECK(k % 4 == 0 && n % 4 == 0, "RNS INT8 GEMM requires K and N multiples of 4");
    TORCH_CHECK(lut_channels >= 0 && lut_channels <= channels, "invalid lut_channels");
    TORCH_CHECK(compact_lut.size(0) == lut_channels && compact_lut.size(1) == 4 &&
                compact_lut.size(2) == 256, "compact LUT shape mismatch");
    TORCH_CHECK(activation_scales.numel() == m && weight_scales.numel() == n,
                "scale shape mismatch");
    TORCH_CHECK(accumulators.size(0) == channels && accumulators.size(1) == m &&
                accumulators.size(2) == n, "accumulator shape mismatch");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
    TORCH_CHECK(a.device() == b.device() && a.device() == moduli.device() &&
                a.device() == reciprocals.device() && a.device() == prefix_inverses.device() &&
                a.device() == compact_lut.device() && a.device() == activation_scales.device() &&
                a.device() == weight_scales.device() && a.device() == accumulators.device() &&
                a.device() == output.device(), "all tensors must share a device");

    c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_cublas_int8_batched(a, b, accumulators, channels, m, k, n, stream);
    const std::int64_t elements = static_cast<std::int64_t>(m) * n;
    const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
    rns_reduce_garner_dequant_kernel<<<blocks, kThreads, 0, stream>>>(
        accumulators.data_ptr<std::int32_t>(),
        moduli.data_ptr<std::int32_t>(),
        reciprocals.data_ptr<std::int64_t>(),
        prefix_inverses.data_ptr<std::int32_t>(),
        compact_lut.numel() == 0 ? nullptr : compact_lut.data_ptr<std::int16_t>(),
        activation_scales.data_ptr<float>(),
        weight_scales.data_ptr<float>(),
        output.data_ptr<float>(),
        elements,
        n,
        channels,
        static_cast<int>(lut_channels));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_rows_int8_out", &quantize_rows_int8_out,
          "Fused per-row FP32 to INT8 quantization (CUDA)");
    m.def("quantize_encode_rows_out", &quantize_encode_rows_out,
          "Fused per-row FP32 quantization and RNS encoding for q8/q16/q32 (CUDA)");
    m.def("native_int8_mm_dequant_out", &native_int8_mm_dequant_out,
          "Native INT8 GEMM and FP32 dequantization (CUDA)");
    m.def("rns_mm_dequant_out", &rns_mm_dequant_out,
          "Full-RNS batched INT8 GEMM with LUT/Barrett reduction and 128-bit Garner (CUDA)");
}
