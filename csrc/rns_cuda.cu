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

namespace {

constexpr int kTile = 16;
constexpr int kPackedStride = 20;  // 16 bytes + padding; always 4-byte aligned.

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

__device__ __forceinline__ std::int32_t centered_mod_i32(
    std::int32_t value,
    std::int32_t modulus) {
    std::int32_t remainder = value % modulus;
    const std::int32_t half = modulus / 2;
    if (remainder > half) {
        remainder -= modulus;
    } else if (remainder < -half) {
        remainder += modulus;
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
    int channel) {
    const std::uint32_t magnitude = value < 0
        ? static_cast<std::uint32_t>(-static_cast<std::int64_t>(value))
        : static_cast<std::uint32_t>(value);
    const std::int64_t base = static_cast<std::int64_t>(channel) * 4 * 256;
    std::int32_t residue = 0;
#pragma unroll
    for (int byte_position = 0; byte_position < 4; ++byte_position) {
        const std::uint32_t byte_value = (magnitude >> (8 * byte_position)) & 0xFFU;
        residue += static_cast<std::int32_t>(
            table[base + byte_position * 256 + byte_value]);
    }
    // At most four values below modulus were added.
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

template <typename scalar_t>
__global__ void encode_centered_kernel(
    const scalar_t* __restrict__ values,
    std::int8_t* __restrict__ output,
    const std::int32_t* __restrict__ moduli,
    std::int64_t elements,
    int channels) {
    const std::int64_t global =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = elements * channels;
    if (global >= total) {
        return;
    }

    const int channel = static_cast<int>(global / elements);
    const std::int64_t index = global - static_cast<std::int64_t>(channel) * elements;
    const int modulus = moduli[channel];
    const std::int64_t value = static_cast<std::int64_t>(values[index]);
    std::int64_t residue = value % modulus;
    const int half = modulus / 2;
    if (residue > half) {
        residue -= modulus;
    } else if (residue < -half) {
        residue += modulus;
    }
    output[global] = static_cast<std::int8_t>(residue);
}

__global__ void encode_int8_kernel(
    const std::int8_t* __restrict__ values,
    std::uint8_t* __restrict__ output,
    const std::int32_t* __restrict__ moduli,
    std::int64_t elements,
    int channels) {
    const std::int64_t global =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = elements * channels;
    if (global >= total) {
        return;
    }

    const int channel = static_cast<int>(global / elements);
    const std::int64_t index = global - static_cast<std::int64_t>(channel) * elements;
    const int modulus = moduli[channel];
    int value = static_cast<int>(values[index]);
    int residue = value % modulus;
    if (residue < 0) {
        residue += modulus;
    }
    output[global] = static_cast<std::uint8_t>(residue);
}

__global__ void rns_gemm_naive_kernel(
    const std::uint8_t* __restrict__ a,
    const std::uint8_t* __restrict__ b,
    std::uint8_t* __restrict__ c,
    const std::int32_t* __restrict__ moduli,
    int m,
    int k,
    int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z;

    if (row >= m || col >= n) {
        return;
    }

    const std::int64_t a_channel_offset =
        static_cast<std::int64_t>(channel) * m * k;
    const std::int64_t b_channel_offset =
        static_cast<std::int64_t>(channel) * k * n;
    const std::int64_t c_channel_offset =
        static_cast<std::int64_t>(channel) * m * n;

    std::uint64_t accumulator = 0;
    for (int inner = 0; inner < k; ++inner) {
        accumulator +=
            static_cast<std::uint64_t>(a[a_channel_offset + row * k + inner]) *
            static_cast<std::uint64_t>(b[b_channel_offset + inner * n + col]);
    }

    c[c_channel_offset + row * n + col] = static_cast<std::uint8_t>(
        accumulator % static_cast<std::uint32_t>(moduli[channel]));
}

template <bool PeriodicReduction>
__global__ void rns_gemm_tiled_kernel(
    const std::uint8_t* __restrict__ a,
    const std::uint8_t* __restrict__ b,
    std::uint8_t* __restrict__ c,
    const std::int32_t* __restrict__ moduli,
    const std::int64_t* __restrict__ reciprocals,
    int m,
    int k,
    int n) {
    __shared__ std::uint32_t tile_a[kTile][kTile + 1];
    __shared__ std::uint32_t tile_b[kTile][kTile + 1];
    __shared__ std::uint32_t shared_modulus;
    __shared__ std::uint32_t shared_reciprocal;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * kTile + tx;
    const int row = blockIdx.y * kTile + ty;
    const int channel = blockIdx.z;

    if (tx == 0 && ty == 0) {
        shared_modulus = static_cast<std::uint32_t>(moduli[channel]);
        shared_reciprocal = static_cast<std::uint32_t>(reciprocals[channel]);
    }

    const std::int64_t a_channel_offset =
        static_cast<std::int64_t>(channel) * m * k;
    const std::int64_t b_channel_offset =
        static_cast<std::int64_t>(channel) * k * n;
    const std::int64_t c_channel_offset =
        static_cast<std::int64_t>(channel) * m * n;

    std::uint32_t accumulator = 0;

    for (int tile_start = 0; tile_start < k; tile_start += kTile) {
        const int a_col = tile_start + tx;
        const int b_row = tile_start + ty;

        tile_a[ty][tx] =
            (row < m && a_col < k)
                ? static_cast<std::uint32_t>(a[a_channel_offset + row * k + a_col])
                : 0U;
        tile_b[ty][tx] =
            (b_row < k && col < n)
                ? static_cast<std::uint32_t>(b[b_channel_offset + b_row * n + col])
                : 0U;

        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kTile; ++inner) {
            accumulator += tile_a[ty][inner] * tile_b[inner][tx];
        }

        __syncthreads();

        if constexpr (PeriodicReduction) {
            accumulator = fast_mod_u32(
                accumulator,
                shared_modulus,
                shared_reciprocal);
        }
    }

    if (row < m && col < n) {
        const std::uint32_t residue = fast_mod_u32(
            accumulator,
            shared_modulus,
            shared_reciprocal);
        c[c_channel_offset + row * n + col] = static_cast<std::uint8_t>(residue);
    }
}

__global__ void rns_gemm_centered_scalar_kernel(
    const std::int8_t* __restrict__ a,
    const std::int8_t* __restrict__ b,
    std::int8_t* __restrict__ c,
    const std::int32_t* __restrict__ moduli,
    int m,
    int k,
    int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z;
    if (row >= m || col >= n) {
        return;
    }

    const std::int64_t a_offset = static_cast<std::int64_t>(channel) * m * k;
    const std::int64_t b_offset = static_cast<std::int64_t>(channel) * k * n;
    const std::int64_t c_offset = static_cast<std::int64_t>(channel) * m * n;

    std::int64_t accumulator = 0;
    for (int inner = 0; inner < k; ++inner) {
        accumulator +=
            static_cast<std::int32_t>(a[a_offset + row * k + inner]) *
            static_cast<std::int32_t>(b[b_offset + inner * n + col]);
    }

    std::int64_t residue = accumulator % moduli[channel];
    const int half = moduli[channel] / 2;
    if (residue > half) {
        residue -= moduli[channel];
    } else if (residue < -half) {
        residue += moduli[channel];
    }
    c[c_offset + row * n + col] = static_cast<std::int8_t>(residue);
}

template <bool PeriodicReduction>
__global__ void rns_gemm_dp4a_kernel(
    const std::int8_t* __restrict__ a,
    const std::int8_t* __restrict__ b,
    std::int8_t* __restrict__ c,
    const std::int32_t* __restrict__ moduli,
    int m,
    int k,
    int n) {
    // B is transposed while entering shared memory. Then both A's row and B's
    // logical column are four contiguous signed bytes, ready for __dp4a.
    __shared__ __align__(16) std::int8_t tile_a[kTile][kPackedStride];
    __shared__ __align__(16) std::int8_t tile_b_transposed[kTile][kPackedStride];
    __shared__ std::int32_t shared_modulus;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * kTile + tx;
    const int row = blockIdx.y * kTile + ty;
    const int channel = blockIdx.z;

    if (tx == 0 && ty == 0) {
        shared_modulus = moduli[channel];
    }

    const std::int64_t a_offset = static_cast<std::int64_t>(channel) * m * k;
    const std::int64_t b_offset = static_cast<std::int64_t>(channel) * k * n;
    const std::int64_t c_offset = static_cast<std::int64_t>(channel) * m * n;

    std::int32_t accumulator = 0;

    for (int tile_start = 0; tile_start < k; tile_start += kTile) {
        const int a_col = tile_start + tx;
        const int b_row = tile_start + ty;

        tile_a[ty][tx] =
            (row < m && a_col < k)
                ? a[a_offset + row * k + a_col]
                : static_cast<std::int8_t>(0);
        tile_b_transposed[tx][ty] =
            (b_row < k && col < n)
                ? b[b_offset + b_row * n + col]
                : static_cast<std::int8_t>(0);

        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kTile; inner += 4) {
            const int packed_a =
                *reinterpret_cast<const int*>(&tile_a[ty][inner]);
            const int packed_b =
                *reinterpret_cast<const int*>(&tile_b_transposed[tx][inner]);
            accumulator = __dp4a(packed_a, packed_b, accumulator);
        }

        __syncthreads();

        if constexpr (PeriodicReduction) {
            accumulator = centered_mod_i32(accumulator, shared_modulus);
        }
    }

    if (row < m && col < n) {
        c[c_offset + row * n + col] = static_cast<std::int8_t>(
            centered_mod_i32(accumulator, shared_modulus));
    }
}

__global__ void reduce_i32_to_centered_kernel(
    const std::int32_t* __restrict__ input,
    std::int8_t* __restrict__ output,
    const std::int32_t* __restrict__ moduli,
    std::int64_t elements_per_channel,
    int channels) {
    const std::int64_t global =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = elements_per_channel * channels;
    if (global >= total) {
        return;
    }
    const int channel = static_cast<int>(global / elements_per_channel);
    output[global] = static_cast<std::int8_t>(
        centered_mod_i32(input[global], moduli[channel]));
}


template <int Channels>
__device__ __forceinline__ std::int64_t garner_reconstruct(
    const std::int32_t (&canonical_residues)[Channels],
    const std::int32_t* __restrict__ moduli,
    const std::int32_t* __restrict__ pairwise_inverses,
    const std::int64_t* __restrict__ reciprocals,
    const std::int64_t modulus_product) {
    std::int32_t digits[Channels];

#pragma unroll
    for (int i = 0; i < Channels; ++i) {
        std::int32_t value = canonical_residues[i];
#pragma unroll
        for (int j = 0; j < i; ++j) {
            const std::int32_t inverse = pairwise_inverses[j * Channels + i];
            value = canonical_mod_i32_barrett(
                (value - digits[j]) * inverse,
                moduli[i],
                static_cast<std::uint32_t>(reciprocals[i]));
        }
        digits[i] = value;
    }

    std::int64_t result = 0;
    std::int64_t prefix = 1;
#pragma unroll
    for (int i = 0; i < Channels; ++i) {
        result += prefix * static_cast<std::int64_t>(digits[i]);
        prefix *= static_cast<std::int64_t>(moduli[i]);
    }
    if (result > modulus_product / 2) {
        result -= modulus_product;
    }
    return result;
}

template <int Channels>
__global__ void decode_garner_centered_kernel(
    const std::int8_t* __restrict__ residues,
    std::int64_t* __restrict__ output,
    const std::int32_t* __restrict__ moduli,
    const std::int32_t* __restrict__ pairwise_inverses,
    const std::int64_t* __restrict__ reciprocals,
    const std::int64_t modulus_product,
    const std::int64_t elements) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }

    std::int32_t canonical[Channels];
#pragma unroll
    for (int channel = 0; channel < Channels; ++channel) {
        const std::int32_t centered = static_cast<std::int32_t>(
            residues[static_cast<std::int64_t>(channel) * elements + index]);
        canonical[channel] = centered < 0 ? centered + moduli[channel] : centered;
    }
    output[index] = garner_reconstruct<Channels>(
        canonical, moduli, pairwise_inverses, reciprocals, modulus_product);
}

template <int Channels>
__global__ void fused_reduce_garner_kernel(
    const std::int32_t* __restrict__ accumulators,
    std::int64_t* __restrict__ output,
    const std::int32_t* __restrict__ moduli,
    const std::int64_t* __restrict__ reciprocals,
    const std::int32_t* __restrict__ pairwise_inverses,
    const std::int16_t* __restrict__ compact_lut,
    const std::int64_t modulus_product,
    const std::int64_t elements,
    const int lut_channels) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }

    std::int32_t canonical[Channels];
#pragma unroll
    for (int channel = 0; channel < Channels; ++channel) {
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
    output[index] = garner_reconstruct<Channels>(
        canonical, moduli, pairwise_inverses, reciprocals, modulus_product);
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
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));

    cublasPointerMode_t previous_pointer_mode;
    TORCH_CUDABLAS_CHECK(cublasGetPointerMode(handle, &previous_pointer_mode));
    TORCH_CUDABLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

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

void validate_moduli_tensor(const torch::Tensor& moduli) {
    TORCH_CHECK(moduli.is_cuda(), "moduli must be CUDA");
    TORCH_CHECK(moduli.is_contiguous(), "moduli must be contiguous");
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32, "moduli must be int32");
    TORCH_CHECK(moduli.dim() == 1, "moduli must be rank 1");
}

void validate_centered_inputs(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& moduli) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a and b must be contiguous");
    TORCH_CHECK(a.scalar_type() == torch::kInt8, "a must be int8 centered residues");
    TORCH_CHECK(b.scalar_type() == torch::kInt8, "b must be int8 centered residues");
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3, "expected [R,M,K] and [R,K,N]");
    TORCH_CHECK(a.device() == b.device(), "a and b must share device");
    validate_moduli_tensor(moduli);
    TORCH_CHECK(a.device() == moduli.device(), "all tensors must share device");
    TORCH_CHECK(a.size(0) == b.size(0), "channel count mismatch");
    TORCH_CHECK(a.size(2) == b.size(1), "K dimension mismatch");
    TORCH_CHECK(moduli.numel() == a.size(0), "moduli count mismatch");
}

}  // namespace

torch::Tensor rns_encode_int8_cuda(
    torch::Tensor values,
    torch::Tensor moduli) {
    TORCH_CHECK(values.is_cuda(), "values must be CUDA");
    TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
    TORCH_CHECK(values.scalar_type() == torch::kInt8, "values must be int8");
    validate_moduli_tensor(moduli);
    TORCH_CHECK(values.device() == moduli.device(), "values and moduli must share device");

    c10::cuda::CUDAGuard guard(values.device());
    const auto channels = static_cast<int>(moduli.numel());
    const auto elements = values.numel();
    TORCH_CHECK(channels > 0, "at least one modulus is required");

    auto output_sizes = values.sizes().vec();
    output_sizes.insert(output_sizes.begin(), channels);
    auto output = torch::empty(output_sizes, values.options().dtype(torch::kUInt8));

    constexpr int threads = 256;
    const std::int64_t total = elements * channels;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    encode_int8_kernel<<<blocks, threads, 0, stream>>>(
        values.data_ptr<std::int8_t>(),
        output.data_ptr<std::uint8_t>(),
        moduli.data_ptr<std::int32_t>(),
        elements,
        channels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor rns_encode_centered_cuda(
    torch::Tensor values,
    torch::Tensor moduli) {
    TORCH_CHECK(values.is_cuda(), "values must be CUDA");
    TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
    TORCH_CHECK(
        values.scalar_type() == torch::kInt8 ||
        values.scalar_type() == torch::kInt16 ||
        values.scalar_type() == torch::kInt32,
        "centered encoder supports int8, int16 and int32");
    validate_moduli_tensor(moduli);
    TORCH_CHECK(values.device() == moduli.device(), "values and moduli must share device");

    c10::cuda::CUDAGuard guard(values.device());
    const auto channels = static_cast<int>(moduli.numel());
    const auto elements = values.numel();
    TORCH_CHECK(channels > 0, "at least one modulus is required");

    auto output_sizes = values.sizes().vec();
    output_sizes.insert(output_sizes.begin(), channels);
    auto output = torch::empty(output_sizes, values.options().dtype(torch::kInt8));

    constexpr int threads = 256;
    const std::int64_t total = elements * channels;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (values.scalar_type() == torch::kInt8) {
        encode_centered_kernel<<<blocks, threads, 0, stream>>>(
            values.data_ptr<std::int8_t>(), output.data_ptr<std::int8_t>(),
            moduli.data_ptr<std::int32_t>(), elements, channels);
    } else if (values.scalar_type() == torch::kInt16) {
        encode_centered_kernel<<<blocks, threads, 0, stream>>>(
            values.data_ptr<std::int16_t>(), output.data_ptr<std::int8_t>(),
            moduli.data_ptr<std::int32_t>(), elements, channels);
    } else {
        encode_centered_kernel<<<blocks, threads, 0, stream>>>(
            values.data_ptr<std::int32_t>(), output.data_ptr<std::int8_t>(),
            moduli.data_ptr<std::int32_t>(), elements, channels);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor rns_matmul_residues_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor moduli,
    torch::Tensor reciprocals,
    std::int64_t kernel_id) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a and b must be contiguous");
    TORCH_CHECK(a.scalar_type() == torch::kUInt8, "a must be uint8");
    TORCH_CHECK(b.scalar_type() == torch::kUInt8, "b must be uint8");
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3, "expected [R,M,K] and [R,K,N]");
    TORCH_CHECK(a.device() == b.device(), "a and b must share device");
    validate_moduli_tensor(moduli);
    TORCH_CHECK(reciprocals.is_cuda(), "reciprocals must be CUDA");
    TORCH_CHECK(reciprocals.is_contiguous(), "reciprocals must be contiguous");
    TORCH_CHECK(reciprocals.scalar_type() == torch::kInt64, "reciprocals must be int64");
    TORCH_CHECK(reciprocals.dim() == 1, "reciprocals must be rank 1");
    TORCH_CHECK(a.device() == moduli.device() && a.device() == reciprocals.device(),
                "all tensors must share device");

    const auto channels64 = a.size(0);
    const auto m64 = a.size(1);
    const auto k64 = a.size(2);
    TORCH_CHECK(b.size(0) == channels64, "channel count mismatch");
    TORCH_CHECK(b.size(1) == k64, "K dimension mismatch");
    const auto n64 = b.size(2);
    TORCH_CHECK(moduli.numel() == channels64, "moduli count mismatch");
    TORCH_CHECK(reciprocals.numel() == channels64, "reciprocal count mismatch");
    TORCH_CHECK(channels64 <= 65535, "too many residue channels for grid.z");
    TORCH_CHECK(m64 <= std::numeric_limits<int>::max() &&
                k64 <= std::numeric_limits<int>::max() &&
                n64 <= std::numeric_limits<int>::max(),
                "matrix dimensions exceed int32 launch limits");
    TORCH_CHECK(m64 > 0 && k64 > 0 && n64 > 0, "matrix dimensions must be positive");
    TORCH_CHECK(kernel_id >= 0 && kernel_id <= 2, "kernel_id must be 0, 1 or 2");

    const int channels = static_cast<int>(channels64);
    const int m = static_cast<int>(m64);
    const int k = static_cast<int>(k64);
    const int n = static_cast<int>(n64);

    c10::cuda::CUDAGuard guard(a.device());
    auto output = torch::empty({channels64, m64, n64}, a.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const dim3 block(kTile, kTile);
    const dim3 grid(
        static_cast<unsigned int>((n + kTile - 1) / kTile),
        static_cast<unsigned int>((m + kTile - 1) / kTile),
        static_cast<unsigned int>(channels));

    if (kernel_id == 0) {
        rns_gemm_naive_kernel<<<grid, block, 0, stream>>>(
            a.data_ptr<std::uint8_t>(), b.data_ptr<std::uint8_t>(),
            output.data_ptr<std::uint8_t>(), moduli.data_ptr<std::int32_t>(),
            m, k, n);
    } else if (kernel_id == 1) {
        rns_gemm_tiled_kernel<false><<<grid, block, 0, stream>>>(
            a.data_ptr<std::uint8_t>(), b.data_ptr<std::uint8_t>(),
            output.data_ptr<std::uint8_t>(), moduli.data_ptr<std::int32_t>(),
            reciprocals.data_ptr<std::int64_t>(), m, k, n);
    } else {
        rns_gemm_tiled_kernel<true><<<grid, block, 0, stream>>>(
            a.data_ptr<std::uint8_t>(), b.data_ptr<std::uint8_t>(),
            output.data_ptr<std::uint8_t>(), moduli.data_ptr<std::int32_t>(),
            reciprocals.data_ptr<std::int64_t>(), m, k, n);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor rns_matmul_centered_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor moduli,
    std::int64_t kernel_id) {
    validate_centered_inputs(a, b, moduli);
    TORCH_CHECK(kernel_id >= 0 && kernel_id <= 3,
                "centered kernel_id must be scalar=0, dp4a=1, dp4a_safe=2 or cublas=3");

    const auto channels64 = a.size(0);
    const auto m64 = a.size(1);
    const auto k64 = a.size(2);
    const auto n64 = b.size(2);
    TORCH_CHECK(channels64 <= 65535, "too many residue channels for grid.z");
    TORCH_CHECK(m64 > 0 && k64 > 0 && n64 > 0, "matrix dimensions must be positive");
    TORCH_CHECK(m64 <= std::numeric_limits<int>::max() &&
                k64 <= std::numeric_limits<int>::max() &&
                n64 <= std::numeric_limits<int>::max(),
                "matrix dimensions exceed int32 limits");

    const int channels = static_cast<int>(channels64);
    const int m = static_cast<int>(m64);
    const int k = static_cast<int>(k64);
    const int n = static_cast<int>(n64);

    c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto output = torch::empty({channels64, m64, n64}, a.options());

    if (kernel_id == 3) {
        TORCH_CHECK(k % 4 == 0 && n % 4 == 0,
                    "cuBLAS int8 path requires K and N to be multiples of 4");

        auto accumulators = torch::empty(
            {channels64, m64, n64},
            a.options().dtype(torch::kInt32));

        cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
        TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));

        cublasPointerMode_t previous_pointer_mode;
        TORCH_CUDABLAS_CHECK(cublasGetPointerMode(handle, &previous_pointer_mode));
        TORCH_CUDABLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

        const std::int32_t alpha = 1;
        const std::int32_t beta = 0;

        // Row-major C=A@B is column-major C^T=B^T@A^T. Contiguous row-major B
        // is already a column-major view of B^T, and likewise for A^T.
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

        constexpr int threads = 256;
        const std::int64_t elements_per_channel = static_cast<std::int64_t>(m) * n;
        const std::int64_t total = elements_per_channel * channels;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        reduce_i32_to_centered_kernel<<<blocks, threads, 0, stream>>>(
            accumulators.data_ptr<std::int32_t>(),
            output.data_ptr<std::int8_t>(),
            moduli.data_ptr<std::int32_t>(),
            elements_per_channel,
            channels);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }

    const dim3 block(kTile, kTile);
    const dim3 grid(
        static_cast<unsigned int>((n + kTile - 1) / kTile),
        static_cast<unsigned int>((m + kTile - 1) / kTile),
        static_cast<unsigned int>(channels));

    if (kernel_id == 0) {
        rns_gemm_centered_scalar_kernel<<<grid, block, 0, stream>>>(
            a.data_ptr<std::int8_t>(), b.data_ptr<std::int8_t>(),
            output.data_ptr<std::int8_t>(), moduli.data_ptr<std::int32_t>(),
            m, k, n);
    } else if (kernel_id == 1) {
        rns_gemm_dp4a_kernel<false><<<grid, block, 0, stream>>>(
            a.data_ptr<std::int8_t>(), b.data_ptr<std::int8_t>(),
            output.data_ptr<std::int8_t>(), moduli.data_ptr<std::int32_t>(),
            m, k, n);
    } else {
        rns_gemm_dp4a_kernel<true><<<grid, block, 0, stream>>>(
            a.data_ptr<std::int8_t>(), b.data_ptr<std::int8_t>(),
            output.data_ptr<std::int8_t>(), moduli.data_ptr<std::int32_t>(),
            m, k, n);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor rns_decode_garner_cuda(
    torch::Tensor residues,
    torch::Tensor moduli,
    torch::Tensor reciprocals,
    torch::Tensor pairwise_inverses,
    std::int64_t modulus_product) {
    TORCH_CHECK(residues.is_cuda(), "residues must be CUDA");
    TORCH_CHECK(residues.is_contiguous(), "residues must be contiguous");
    TORCH_CHECK(residues.scalar_type() == torch::kInt8,
                "Garner decoder expects centered int8 residues");
    TORCH_CHECK(residues.dim() >= 2, "residues must have shape [R,...]");
    validate_moduli_tensor(moduli);
    TORCH_CHECK(reciprocals.is_cuda() && reciprocals.is_contiguous(),
                "reciprocals must be contiguous CUDA tensor");
    TORCH_CHECK(reciprocals.scalar_type() == torch::kInt64 && reciprocals.dim() == 1,
                "reciprocals must be int64 [R]");
    TORCH_CHECK(pairwise_inverses.is_cuda(), "pairwise inverses must be CUDA");
    TORCH_CHECK(pairwise_inverses.is_contiguous(), "pairwise inverses must be contiguous");
    TORCH_CHECK(pairwise_inverses.scalar_type() == torch::kInt32,
                "pairwise inverses must be int32");
    TORCH_CHECK(pairwise_inverses.dim() == 2,
                "pairwise inverses must have shape [R,R]");
    TORCH_CHECK(residues.device() == moduli.device() &&
                residues.device() == reciprocals.device() &&
                residues.device() == pairwise_inverses.device(),
                "all tensors must share device");

    const int channels = static_cast<int>(residues.size(0));
    TORCH_CHECK(moduli.numel() == channels, "moduli count mismatch");
    TORCH_CHECK(reciprocals.numel() == channels, "reciprocal count mismatch");
    TORCH_CHECK(pairwise_inverses.size(0) == channels &&
                pairwise_inverses.size(1) == channels,
                "pairwise inverse shape mismatch");

    auto output_sizes = residues.sizes().vec();
    output_sizes.erase(output_sizes.begin());
    auto output = torch::empty(output_sizes, residues.options().dtype(torch::kInt64));
    const std::int64_t elements = output.numel();

    c10::cuda::CUDAGuard guard(residues.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);

#define LAUNCH_DECODE(CHANNELS) \
    decode_garner_centered_kernel<CHANNELS><<<blocks, threads, 0, stream>>>( \
        residues.data_ptr<std::int8_t>(), output.data_ptr<std::int64_t>(), \
        moduli.data_ptr<std::int32_t>(), pairwise_inverses.data_ptr<std::int32_t>(), \
        reciprocals.data_ptr<std::int64_t>(), modulus_product, elements)

    switch (channels) {
        case 2: LAUNCH_DECODE(2); break;
        case 3: LAUNCH_DECODE(3); break;
        case 4: LAUNCH_DECODE(4); break;
        case 5: LAUNCH_DECODE(5); break;
        case 6: LAUNCH_DECODE(6); break;
        case 7: LAUNCH_DECODE(7); break;
        case 8: LAUNCH_DECODE(8); break;
        case 9: LAUNCH_DECODE(9); break;
        case 10: LAUNCH_DECODE(10); break;
        default:
            TORCH_CHECK(false, "Garner decoder supports 2..10 channels");
    }
#undef LAUNCH_DECODE

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

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
    torch::Tensor output) {
    validate_centered_inputs(a, b, moduli);
    TORCH_CHECK(reciprocals.is_cuda() && reciprocals.is_contiguous(),
                "reciprocals must be contiguous CUDA tensor");
    TORCH_CHECK(reciprocals.scalar_type() == torch::kInt64 && reciprocals.dim() == 1,
                "reciprocals must be int64 [R]");
    TORCH_CHECK(pairwise_inverses.is_cuda() && pairwise_inverses.is_contiguous(),
                "pairwise inverses must be contiguous CUDA tensor");
    TORCH_CHECK(pairwise_inverses.scalar_type() == torch::kInt32 &&
                pairwise_inverses.dim() == 2,
                "pairwise inverses must be int32 [R,R]");
    TORCH_CHECK(compact_lut.is_cuda() && compact_lut.is_contiguous(),
                "compact LUT must be contiguous CUDA tensor");
    TORCH_CHECK(compact_lut.scalar_type() == torch::kInt16 && compact_lut.dim() == 3,
                "compact LUT must be int16 [R,4,256]");
    TORCH_CHECK(accumulators.is_cuda() && accumulators.is_contiguous(),
                "accumulator workspace must be contiguous CUDA tensor");
    TORCH_CHECK(accumulators.scalar_type() == torch::kInt32 && accumulators.dim() == 3,
                "accumulator workspace must be int32 [R,M,N]");
    TORCH_CHECK(output.is_cuda() && output.is_contiguous(),
                "output workspace must be contiguous CUDA tensor");
    TORCH_CHECK(output.scalar_type() == torch::kInt64 && output.dim() == 2,
                "output workspace must be int64 [M,N]");
    TORCH_CHECK(a.device() == reciprocals.device() &&
                a.device() == pairwise_inverses.device() &&
                a.device() == compact_lut.device() &&
                a.device() == accumulators.device() &&
                a.device() == output.device(),
                "all tensors must share device");

    const int channels = static_cast<int>(a.size(0));
    const int m = static_cast<int>(a.size(1));
    const int k = static_cast<int>(a.size(2));
    const int n = static_cast<int>(b.size(2));
    TORCH_CHECK(k % 4 == 0 && n % 4 == 0,
                "fused cuBLAS path requires K and N multiples of 4");
    TORCH_CHECK(reciprocals.numel() == channels, "reciprocal count mismatch");
    TORCH_CHECK(pairwise_inverses.size(0) == channels &&
                pairwise_inverses.size(1) == channels,
                "pairwise inverse shape mismatch");
    TORCH_CHECK(compact_lut.size(0) == channels &&
                compact_lut.size(1) == 4 && compact_lut.size(2) == 256,
                "compact LUT shape mismatch");
    TORCH_CHECK(accumulators.size(0) == channels &&
                accumulators.size(1) == m && accumulators.size(2) == n,
                "accumulator workspace shape mismatch");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n,
                "output workspace shape mismatch");
    TORCH_CHECK(lut_channels >= 0 && lut_channels <= 2 && lut_channels <= channels,
                "lut_channels must be 0, 1 or 2 and <= channel count");

    c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_cublas_int8_batched(a, b, accumulators, channels, m, k, n, stream);

    const std::int64_t elements = static_cast<std::int64_t>(m) * n;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);

#define LAUNCH_FUSED(CHANNELS) \
    fused_reduce_garner_kernel<CHANNELS><<<blocks, threads, 0, stream>>>( \
        accumulators.data_ptr<std::int32_t>(), output.data_ptr<std::int64_t>(), \
        moduli.data_ptr<std::int32_t>(), reciprocals.data_ptr<std::int64_t>(), \
        pairwise_inverses.data_ptr<std::int32_t>(), compact_lut.data_ptr<std::int16_t>(), \
        modulus_product, elements, static_cast<int>(lut_channels))

    switch (channels) {
        case 2: LAUNCH_FUSED(2); break;
        case 3: LAUNCH_FUSED(3); break;
        case 4: LAUNCH_FUSED(4); break;
        case 5: LAUNCH_FUSED(5); break;
        case 6: LAUNCH_FUSED(6); break;
        case 7: LAUNCH_FUSED(7); break;
        case 8: LAUNCH_FUSED(8); break;
        case 9: LAUNCH_FUSED(9); break;
        case 10: LAUNCH_FUSED(10); break;
        default:
            TORCH_CHECK(false, "fused Garner path supports 2..10 channels");
    }
#undef LAUNCH_FUSED

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
