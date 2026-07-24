#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

struct U128 {
    std::uint64_t lo;
    std::uint64_t hi;
};

__device__ __forceinline__ U128 u128_from_u64(std::uint64_t value) {
    return U128{value, 0};
}

__device__ __forceinline__ U128 u128_add(U128 a, U128 b) {
    const std::uint64_t lo = a.lo + b.lo;
    const std::uint64_t carry = lo < a.lo ? 1ULL : 0ULL;
    return U128{lo, a.hi + b.hi + carry};
}

__device__ __forceinline__ U128 u128_sub(U128 a, U128 b) {
    const std::uint64_t borrow = a.lo < b.lo ? 1ULL : 0ULL;
    return U128{a.lo - b.lo, a.hi - b.hi - borrow};
}

__device__ __forceinline__ bool u128_gt(U128 a, U128 b) {
    return (a.hi > b.hi) || (a.hi == b.hi && a.lo > b.lo);
}

__device__ __forceinline__ U128 u128_shr1(U128 a) {
    return U128{(a.lo >> 1) | (a.hi << 63), a.hi >> 1};
}

__device__ __forceinline__ U128 u128_mul_small(U128 a, std::uint32_t b) {
    const std::uint64_t multiplier = static_cast<std::uint64_t>(b);
    const std::uint64_t lo = a.lo * multiplier;
    const std::uint64_t carry = __umul64hi(a.lo, multiplier);
    const std::uint64_t hi = a.hi * multiplier + carry;
    return U128{lo, hi};
}

__device__ __forceinline__ std::uint32_t u128_mod_small(
    U128 value,
    std::uint32_t modulus,
    std::uint32_t two64_mod) {
    const std::uint64_t hi_mod = value.hi % modulus;
    const std::uint64_t lo_mod = value.lo % modulus;
    return static_cast<std::uint32_t>(
        (hi_mod * static_cast<std::uint64_t>(two64_mod) + lo_mod) % modulus);
}

__device__ __forceinline__ double u128_to_double(U128 value) {
    constexpr double TWO64 = 18446744073709551616.0;
    return static_cast<double>(value.hi) * TWO64 + static_cast<double>(value.lo);
}

__device__ __forceinline__ std::int64_t quantize_double(
    float value,
    double scale,
    std::int64_t quant_max) {
    if (!(scale > 0.0)) {
        return 0;
    }
    std::int64_t quantized = __double2ll_rn(static_cast<double>(value) / scale);
    if (quantized > quant_max) {
        quantized = quant_max;
    } else if (quantized < -quant_max) {
        quantized = -quant_max;
    }
    return quantized;
}

__global__ void encode_activation_fp32_kernel(
    const float* __restrict__ input,
    const double* __restrict__ scales,
    const std::int32_t* __restrict__ moduli,
    std::int8_t* __restrict__ output,
    std::int64_t elements,
    int k,
    int channels,
    std::int64_t quant_max) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = static_cast<int>(index / k);
    const std::int64_t quantized = quantize_double(
        input[index], scales[row], quant_max);
    for (int channel = 0; channel < channels; ++channel) {
        const int modulus = moduli[channel];
        std::int64_t residue = quantized % modulus;
        const int half = modulus / 2;
        if (residue > half) {
            residue -= modulus;
        } else if (residue < -half) {
            residue += modulus;
        }
        output[static_cast<std::int64_t>(channel) * elements + index] =
            static_cast<std::int8_t>(residue);
    }
}

__global__ void encode_weight_fp32_kernel(
    const float* __restrict__ weight,
    const double* __restrict__ scales,
    const std::int32_t* __restrict__ moduli,
    std::int8_t* __restrict__ output,
    std::int64_t elements,
    int n,
    int k,
    int channels,
    std::int64_t quant_max) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row_n = static_cast<int>(index / k);
    const int col_k = static_cast<int>(index - static_cast<std::int64_t>(row_n) * k);
    const std::int64_t quantized = quantize_double(
        weight[index], scales[row_n], quant_max);
    const std::int64_t transposed =
        static_cast<std::int64_t>(col_k) * n + row_n;
    for (int channel = 0; channel < channels; ++channel) {
        const int modulus = moduli[channel];
        std::int64_t residue = quantized % modulus;
        const int half = modulus / 2;
        if (residue > half) {
            residue -= modulus;
        } else if (residue < -half) {
            residue += modulus;
        }
        output[static_cast<std::int64_t>(channel) * elements + transposed] =
            static_cast<std::int8_t>(residue);
    }
}

__global__ void quantize_activation_int8_kernel(
    const float* __restrict__ input,
    const double* __restrict__ scales,
    std::int8_t* __restrict__ output,
    std::int64_t elements,
    int k) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = static_cast<int>(index / k);
    const std::int64_t quantized = quantize_double(input[index], scales[row], 127);
    output[index] = static_cast<std::int8_t>(quantized);
}

__global__ void quantize_weight_int8_kernel(
    const float* __restrict__ weight,
    const double* __restrict__ scales,
    std::int8_t* __restrict__ output,
    std::int64_t elements,
    int n,
    int k) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row_n = static_cast<int>(index / k);
    const int col_k = static_cast<int>(index - static_cast<std::int64_t>(row_n) * k);
    const std::int64_t quantized = quantize_double(weight[index], scales[row_n], 127);
    output[static_cast<std::int64_t>(col_k) * n + row_n] =
        static_cast<std::int8_t>(quantized);
}

__global__ void native_dequant_fp32_kernel(
    const std::int32_t* __restrict__ accumulators,
    float* __restrict__ output,
    const double* __restrict__ activation_scales,
    const double* __restrict__ weight_scales,
    const float* __restrict__ bias,
    std::int64_t elements,
    int n,
    bool has_bias) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<std::int64_t>(row) * n);
    double value = static_cast<double>(accumulators[index]);
    value *= activation_scales[row] * weight_scales[col];
    if (has_bias) {
        value += static_cast<double>(bias[col]);
    }
    output[index] = static_cast<float>(value);
}

__global__ void rns_garner_dequant_fp32_kernel(
    const std::int32_t* __restrict__ accumulators,
    float* __restrict__ output,
    const std::int32_t* __restrict__ moduli,
    const std::int32_t* __restrict__ prefix_inverses,
    const std::int32_t* __restrict__ two64_mod,
    const double* __restrict__ activation_scales,
    const double* __restrict__ weight_scales,
    const float* __restrict__ bias,
    std::int64_t elements,
    int n,
    int channels,
    bool has_bias) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }

    const int first_modulus = moduli[0];
    std::int64_t first = accumulators[index] % first_modulus;
    if (first < 0) {
        first += first_modulus;
    }
    U128 value = u128_from_u64(static_cast<std::uint64_t>(first));
    U128 prefix = u128_from_u64(static_cast<std::uint64_t>(first_modulus));

    for (int channel = 1; channel < channels; ++channel) {
        const int modulus = moduli[channel];
        std::int64_t residue = accumulators[
            static_cast<std::int64_t>(channel) * elements + index] % modulus;
        if (residue < 0) {
            residue += modulus;
        }
        const std::uint32_t value_mod = u128_mod_small(
            value,
            static_cast<std::uint32_t>(modulus),
            static_cast<std::uint32_t>(two64_mod[channel]));
        int delta = static_cast<int>(residue) - static_cast<int>(value_mod);
        delta %= modulus;
        if (delta < 0) {
            delta += modulus;
        }
        const std::uint32_t digit = static_cast<std::uint32_t>(
            (static_cast<std::int64_t>(delta) * prefix_inverses[channel]) % modulus);
        value = u128_add(value, u128_mul_small(prefix, digit));
        prefix = u128_mul_small(prefix, static_cast<std::uint32_t>(modulus));
    }

    const U128 half = u128_shr1(prefix);
    double reconstructed;
    if (u128_gt(value, half)) {
        reconstructed = -u128_to_double(u128_sub(prefix, value));
    } else {
        reconstructed = u128_to_double(value);
    }

    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<std::int64_t>(row) * n);
    double dequantized = reconstructed * activation_scales[row] * weight_scales[col];
    if (has_bias) {
        dequantized += static_cast<double>(bias[col]);
    }
    output[index] = static_cast<float>(dequantized);
}

void validate_cuda_contiguous(const torch::Tensor& tensor, const char* name, int rank) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == rank, name, " rank mismatch");
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


torch::Tensor encode_activation_fp32_out(
    torch::Tensor input,
    torch::Tensor scales,
    std::int64_t quant_max,
    torch::Tensor moduli,
    torch::Tensor output) {
    validate_cuda_contiguous(input, "input", 2);
    validate_cuda_contiguous(scales, "scales", 1);
    validate_cuda_contiguous(moduli, "moduli", 1);
    validate_cuda_contiguous(output, "output", 3);
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat64, "scales must be float64");
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32, "moduli must be int32");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8");
    TORCH_CHECK(quant_max > 0 && quant_max <= 2147483647LL, "invalid quant_max");
    const int m = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    const int channels = static_cast<int>(moduli.numel());
    TORCH_CHECK(channels >= 2 && channels <= 12, "channels must be 2..12");
    TORCH_CHECK(scales.numel() == m, "one activation scale per row is required");
    TORCH_CHECK(output.size(0) == channels && output.size(1) == m && output.size(2) == k,
                "activation residue output shape mismatch");
    TORCH_CHECK(input.device() == scales.device() && input.device() == moduli.device() &&
                input.device() == output.device(), "all tensors must share a device");
    c10::cuda::CUDAGuard guard(input.device());
    const std::int64_t elements = input.numel();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    encode_activation_fp32_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(), scales.data_ptr<double>(), moduli.data_ptr<std::int32_t>(),
        output.data_ptr<std::int8_t>(), elements, k, channels, quant_max);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor encode_weight_fp32_out(
    torch::Tensor weight,
    torch::Tensor scales,
    std::int64_t quant_max,
    torch::Tensor moduli,
    torch::Tensor output) {
    validate_cuda_contiguous(weight, "weight", 2);
    validate_cuda_contiguous(scales, "scales", 1);
    validate_cuda_contiguous(moduli, "moduli", 1);
    validate_cuda_contiguous(output, "output", 3);
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat64, "scales must be float64");
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32, "moduli must be int32");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8");
    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    const int channels = static_cast<int>(moduli.numel());
    TORCH_CHECK(channels >= 2 && channels <= 12, "channels must be 2..12");
    TORCH_CHECK(scales.numel() == n, "one weight scale per output row is required");
    TORCH_CHECK(output.size(0) == channels && output.size(1) == k && output.size(2) == n,
                "weight residue output shape mismatch");
    TORCH_CHECK(weight.device() == scales.device() && weight.device() == moduli.device() &&
                weight.device() == output.device(), "all tensors must share a device");
    c10::cuda::CUDAGuard guard(weight.device());
    const std::int64_t elements = weight.numel();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    encode_weight_fp32_kernel<<<blocks, threads, 0, stream>>>(
        weight.data_ptr<float>(), scales.data_ptr<double>(), moduli.data_ptr<std::int32_t>(),
        output.data_ptr<std::int8_t>(), elements, n, k, channels, quant_max);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor quantize_activation_int8_out(
    torch::Tensor input,
    torch::Tensor scales,
    torch::Tensor output) {
    validate_cuda_contiguous(input, "input", 2);
    validate_cuda_contiguous(scales, "scales", 1);
    validate_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat64, "scales must be float64");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8");
    const int m = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    TORCH_CHECK(scales.numel() == m, "one scale per row is required");
    TORCH_CHECK(output.sizes() == input.sizes(), "output shape mismatch");
    c10::cuda::CUDAGuard guard(input.device());
    const std::int64_t elements = input.numel();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_activation_int8_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(), scales.data_ptr<double>(), output.data_ptr<std::int8_t>(),
        elements, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor quantize_weight_int8_out(
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor output) {
    validate_cuda_contiguous(weight, "weight", 2);
    validate_cuda_contiguous(scales, "scales", 1);
    validate_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat64, "scales must be float64");
    TORCH_CHECK(output.scalar_type() == torch::kInt8, "output must be int8");
    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(scales.numel() == n, "one scale per output row is required");
    TORCH_CHECK(output.size(0) == k && output.size(1) == n, "output must have shape [K,N]");
    c10::cuda::CUDAGuard guard(weight.device());
    const std::int64_t elements = weight.numel();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_weight_int8_kernel<<<blocks, threads, 0, stream>>>(
        weight.data_ptr<float>(), scales.data_ptr<double>(), output.data_ptr<std::int8_t>(),
        elements, n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor native_mm_dequant_fp32_out(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    torch::Tensor accumulators,
    torch::Tensor output) {
    validate_cuda_contiguous(a, "a", 2);
    validate_cuda_contiguous(b, "b", 2);
    validate_cuda_contiguous(activation_scales, "activation_scales", 1);
    validate_cuda_contiguous(weight_scales, "weight_scales", 1);
    validate_cuda_contiguous(bias, "bias", 1);
    validate_cuda_contiguous(accumulators, "accumulators", 2);
    validate_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(a.scalar_type() == torch::kInt8 && b.scalar_type() == torch::kInt8,
                "a and b must be int8");
    TORCH_CHECK(activation_scales.scalar_type() == torch::kFloat64 &&
                weight_scales.scalar_type() == torch::kFloat64,
                "scales must be float64");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(accumulators.scalar_type() == torch::kInt32, "accumulators must be int32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));
    TORCH_CHECK(b.size(0) == k, "K mismatch");
    TORCH_CHECK(k % 4 == 0 && n % 4 == 0, "K and N must be multiples of 4");
    TORCH_CHECK(activation_scales.numel() == m && weight_scales.numel() == n,
                "scale shape mismatch");
    TORCH_CHECK(bias.numel() == 0 || bias.numel() == n, "bias length mismatch");
    TORCH_CHECK(accumulators.size(0) == m && accumulators.size(1) == n,
                "accumulator shape mismatch");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
    c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_cublas_int8(a, b, accumulators, m, k, n, stream);
    const std::int64_t elements = static_cast<std::int64_t>(m) * n;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    native_dequant_fp32_kernel<<<blocks, threads, 0, stream>>>(
        accumulators.data_ptr<std::int32_t>(), output.data_ptr<float>(),
        activation_scales.data_ptr<double>(), weight_scales.data_ptr<double>(),
        bias.numel() == 0 ? nullptr : bias.data_ptr<float>(), elements, n,
        bias.numel() == n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor rns_mm_dequant_fp32_out(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor moduli,
    torch::Tensor prefix_inverses,
    torch::Tensor two64_mod,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    torch::Tensor accumulators,
    torch::Tensor output) {
    validate_cuda_contiguous(a, "a", 3);
    validate_cuda_contiguous(b, "b", 3);
    validate_cuda_contiguous(moduli, "moduli", 1);
    validate_cuda_contiguous(prefix_inverses, "prefix_inverses", 1);
    validate_cuda_contiguous(two64_mod, "two64_mod", 1);
    validate_cuda_contiguous(activation_scales, "activation_scales", 1);
    validate_cuda_contiguous(weight_scales, "weight_scales", 1);
    validate_cuda_contiguous(bias, "bias", 1);
    validate_cuda_contiguous(accumulators, "accumulators", 3);
    validate_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(a.scalar_type() == torch::kInt8 && b.scalar_type() == torch::kInt8,
                "a and b must be int8 residue planes");
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32 &&
                prefix_inverses.scalar_type() == torch::kInt32 &&
                two64_mod.scalar_type() == torch::kInt32,
                "RNS constants must be int32");
    TORCH_CHECK(activation_scales.scalar_type() == torch::kFloat64 &&
                weight_scales.scalar_type() == torch::kFloat64,
                "scales must be float64");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(accumulators.scalar_type() == torch::kInt32, "accumulators must be int32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
    const int channels = static_cast<int>(a.size(0));
    const int m = static_cast<int>(a.size(1));
    const int k = static_cast<int>(a.size(2));
    const int n = static_cast<int>(b.size(2));
    TORCH_CHECK(channels >= 2 && channels <= 12, "channels must be 2..12");
    TORCH_CHECK(b.size(0) == channels && b.size(1) == k, "RNS operand shape mismatch");
    TORCH_CHECK(k % 4 == 0 && n % 4 == 0, "K and N must be multiples of 4");
    TORCH_CHECK(moduli.numel() == channels && prefix_inverses.numel() == channels &&
                two64_mod.numel() == channels, "RNS constant length mismatch");
    TORCH_CHECK(activation_scales.numel() == m && weight_scales.numel() == n,
                "scale shape mismatch");
    TORCH_CHECK(bias.numel() == 0 || bias.numel() == n, "bias length mismatch");
    TORCH_CHECK(accumulators.size(0) == channels && accumulators.size(1) == m &&
                accumulators.size(2) == n, "accumulator shape mismatch");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
    c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_cublas_int8_batched(a, b, accumulators, channels, m, k, n, stream);
    const std::int64_t elements = static_cast<std::int64_t>(m) * n;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    rns_garner_dequant_fp32_kernel<<<blocks, threads, 0, stream>>>(
        accumulators.data_ptr<std::int32_t>(), output.data_ptr<float>(),
        moduli.data_ptr<std::int32_t>(), prefix_inverses.data_ptr<std::int32_t>(),
        two64_mod.data_ptr<std::int32_t>(), activation_scales.data_ptr<double>(),
        weight_scales.data_ptr<double>(), bias.numel() == 0 ? nullptr : bias.data_ptr<float>(),
        elements, n, channels, bias.numel() == n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_activation_fp32_out", &encode_activation_fp32_out);
    m.def("encode_weight_fp32_out", &encode_weight_fp32_out);
    m.def("quantize_activation_int8_out", &quantize_activation_int8_out);
    m.def("quantize_weight_int8_out", &quantize_weight_int8_out);
    m.def("native_mm_dequant_fp32_out", &native_mm_dequant_fp32_out);
    m.def("rns_mm_dequant_fp32_out", &rns_mm_dequant_fp32_out);
}
