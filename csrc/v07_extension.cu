#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

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
    int channel) {
    const std::uint32_t magnitude = value < 0
        ? static_cast<std::uint32_t>(-static_cast<std::int64_t>(value))
        : static_cast<std::uint32_t>(value);
    const std::int64_t base = static_cast<std::int64_t>(channel) * 4 * 256;
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
__global__ void fused_reduce_garner_dequant_fp16_kernel(
    const std::int32_t* __restrict__ accumulators,
    __half* __restrict__ output,
    const std::int32_t* __restrict__ moduli,
    const std::int64_t* __restrict__ reciprocals,
    const std::int32_t* __restrict__ pairwise_inverses,
    const std::int16_t* __restrict__ compact_lut,
    const std::int64_t modulus_product,
    const float* __restrict__ activation_scales,
    const float* __restrict__ weight_scales,
    const float* __restrict__ bias,
    const std::int64_t elements,
    const int n,
    const int lut_channels,
    const bool per_row_scale,
    const bool has_bias) {
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

    const std::int64_t reconstructed = garner_reconstruct<Channels>(
        canonical, moduli, pairwise_inverses, reciprocals, modulus_product);
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<std::int64_t>(row) * n);
    const float a_scale = activation_scales[per_row_scale ? row : 0];
    float value = static_cast<float>(reconstructed) * a_scale * weight_scales[col];
    if (has_bias) {
        value += bias[col];
    }
    output[index] = __float2half_rn(value);
}

__global__ void quantize_fp16_kernel(
    const __half* __restrict__ input,
    std::int8_t* __restrict__ output,
    const float* __restrict__ scales,
    const std::int64_t elements,
    const int k,
    const int quant_max,
    const bool per_row_scale) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = static_cast<int>(index / k);
    const float scale = scales[per_row_scale ? row : 0];
    const float value = __half2float(input[index]) / scale;
    int quantized = __float2int_rn(value);
    if (quantized > quant_max) {
        quantized = quant_max;
    } else if (quantized < -quant_max) {
        quantized = -quant_max;
    }
    output[index] = static_cast<std::int8_t>(quantized);
}

__global__ void quantize_encode_fp16_kernel(
    const __half* __restrict__ input,
    std::int8_t* __restrict__ output,
    const float* __restrict__ scales,
    const std::int32_t* __restrict__ moduli,
    const std::int64_t elements,
    const int k,
    const int channels,
    const int quant_max,
    const bool per_row_scale) {
    const std::int64_t global =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = elements * channels;
    if (global >= total) {
        return;
    }
    const int channel = static_cast<int>(global / elements);
    const std::int64_t index = global - static_cast<std::int64_t>(channel) * elements;
    const int row = static_cast<int>(index / k);
    const float scale = scales[per_row_scale ? row : 0];
    const float value = __half2float(input[index]) / scale;
    int quantized = __float2int_rn(value);
    if (quantized > quant_max) {
        quantized = quant_max;
    } else if (quantized < -quant_max) {
        quantized = -quant_max;
    }

    const int modulus = moduli[channel];
    int residue = quantized % modulus;
    const int half = modulus / 2;
    if (residue > half) {
        residue -= modulus;
    } else if (residue < -half) {
        residue += modulus;
    }
    output[global] = static_cast<std::int8_t>(residue);
}

__global__ void dequant_int32_fp16_kernel(
    const std::int32_t* __restrict__ accumulators,
    __half* __restrict__ output,
    const float* __restrict__ activation_scales,
    const float* __restrict__ weight_scales,
    const float* __restrict__ bias,
    const std::int64_t elements,
    const int n,
    const bool per_row_scale,
    const bool has_bias) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<std::int64_t>(row) * n);
    const float a_scale = activation_scales[per_row_scale ? row : 0];
    float value = static_cast<float>(accumulators[index])
        * a_scale * weight_scales[col];
    if (has_bias) {
        value += bias[col];
    }
    output[index] = __float2half_rn(value);
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
    set_cublas_stream_and_host_pointer_mode(
        handle, stream, &previous_pointer_mode);

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
    set_cublas_stream_and_host_pointer_mode(
        handle, stream, &previous_pointer_mode);

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

void validate_matrix_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name,
    int rank) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == rank, name, " has unexpected rank");
}

void validate_scales_and_output(
    const torch::Tensor& activation_scales,
    const torch::Tensor& weight_scales,
    const torch::Tensor& bias,
    const torch::Tensor& output,
    int m,
    int n,
    const c10::Device& device) {
    validate_matrix_cuda_contiguous(activation_scales, "activation_scales", 1);
    validate_matrix_cuda_contiguous(weight_scales, "weight_scales", 1);
    TORCH_CHECK(activation_scales.scalar_type() == torch::kFloat32,
                "activation_scales must be float32");
    TORCH_CHECK(weight_scales.scalar_type() == torch::kFloat32,
                "weight_scales must be float32");
    TORCH_CHECK(activation_scales.numel() == 1 || activation_scales.numel() == m,
                "activation_scales must have one value or one value per row");
    TORCH_CHECK(weight_scales.numel() == n,
                "weight_scales must have N values");
    TORCH_CHECK(activation_scales.device() == device && weight_scales.device() == device,
                "scales must share the input device");

    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(),
                "bias must be contiguous CUDA tensor (empty is allowed)");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32,
                "bias must be float32");
    TORCH_CHECK(bias.numel() == 0 || bias.numel() == n,
                "bias must be empty or have N values");
    TORCH_CHECK(bias.device() == device, "bias must share the input device");

    validate_matrix_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(output.scalar_type() == torch::kFloat16,
                "output must be float16");
    TORCH_CHECK(output.size(0) == m && output.size(1) == n,
                "output shape mismatch");
    TORCH_CHECK(output.device() == device, "output must share the input device");
}

}  // namespace

torch::Tensor quantize_fp16_out_cuda(
    torch::Tensor input,
    torch::Tensor scales,
    std::int64_t quant_max,
    torch::Tensor output) {
    validate_matrix_cuda_contiguous(input, "input", 2);
    validate_matrix_cuda_contiguous(scales, "scales", 1);
    validate_matrix_cuda_contiguous(output, "output", 2);
    TORCH_CHECK(input.scalar_type() == torch::kFloat16,
                "input must be float16");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
                "scales must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kInt8,
                "output must be int8");
    TORCH_CHECK(input.device() == scales.device() && input.device() == output.device(),
                "input, scales and output must share a device");
    TORCH_CHECK(output.size(0) == input.size(0) &&
                output.size(1) == input.size(1),
                "output shape mismatch");
    TORCH_CHECK(quant_max > 0 && quant_max <= 127,
                "quant_max must be in 1..127 for the fused INT8 path");

    const int m = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    TORCH_CHECK(scales.numel() == 1 || scales.numel() == m,
                "scales must have one value or one value per row");
    const std::int64_t elements = input.numel();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    c10::cuda::CUDAGuard guard(input.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_fp16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        output.data_ptr<std::int8_t>(),
        scales.data_ptr<float>(),
        elements,
        k,
        static_cast<int>(quant_max),
        scales.numel() == m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor quantize_encode_fp16_out_cuda(
    torch::Tensor input,
    torch::Tensor scales,
    torch::Tensor moduli,
    std::int64_t quant_max,
    torch::Tensor output) {
    validate_matrix_cuda_contiguous(input, "input", 2);
    validate_matrix_cuda_contiguous(scales, "scales", 1);
    validate_matrix_cuda_contiguous(moduli, "moduli", 1);
    validate_matrix_cuda_contiguous(output, "output", 3);
    TORCH_CHECK(input.scalar_type() == torch::kFloat16,
                "input must be float16");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
                "scales must be float32");
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32,
                "moduli must be int32");
    TORCH_CHECK(output.scalar_type() == torch::kInt8,
                "output must be int8 centered residues");
    TORCH_CHECK(input.device() == scales.device() &&
                input.device() == moduli.device() && input.device() == output.device(),
                "all tensors must share a device");
    TORCH_CHECK(quant_max > 0 && quant_max <= 127,
                "quant_max must be in 1..127 for the fused INT8 path");

    const int channels = static_cast<int>(moduli.numel());
    const int m = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    TORCH_CHECK(output.size(0) == channels &&
                output.size(1) == m && output.size(2) == k,
                "encoded output shape mismatch");
    TORCH_CHECK(scales.numel() == 1 || scales.numel() == m,
                "scales must have one value or one value per row");
    const std::int64_t elements = input.numel();
    const std::int64_t total = elements * channels;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    c10::cuda::CUDAGuard guard(input.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_encode_fp16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        output.data_ptr<std::int8_t>(),
        scales.data_ptr<float>(),
        moduli.data_ptr<std::int32_t>(),
        elements,
        k,
        channels,
        static_cast<int>(quant_max),
        scales.numel() == m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor native_int8_mm_out_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor accumulators) {
    validate_matrix_cuda_contiguous(a, "a", 2);
    validate_matrix_cuda_contiguous(b, "b", 2);
    validate_matrix_cuda_contiguous(accumulators, "accumulators", 2);
    TORCH_CHECK(a.scalar_type() == torch::kInt8 && b.scalar_type() == torch::kInt8,
                "a and b must be int8");
    TORCH_CHECK(accumulators.scalar_type() == torch::kInt32,
                "accumulators must be int32");
    TORCH_CHECK(a.device() == b.device() && a.device() == accumulators.device(),
                "all tensors must share a device");
    TORCH_CHECK(a.size(1) == b.size(0), "K dimension mismatch");

    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));
    TORCH_CHECK(m > 0 && k > 0 && n > 0, "dimensions must be positive");
    TORCH_CHECK(k % 4 == 0 && n % 4 == 0,
                "native INT8 Tensor Core path requires K and N multiples of 4");
    TORCH_CHECK(accumulators.size(0) == m && accumulators.size(1) == n,
                "accumulator shape mismatch");

    c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_cublas_int8(a, b, accumulators, m, k, n, stream);
    return accumulators;
}

torch::Tensor native_int8_dequant_fp16_out_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    torch::Tensor accumulators,
    torch::Tensor output) {
    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(1));
    validate_scales_and_output(
        activation_scales, weight_scales, bias, output, m, n, a.device());
    native_int8_mm_out_cuda(a, b, accumulators);
    c10::cuda::CUDAGuard guard(a.device());

    const std::int64_t elements = static_cast<std::int64_t>(m) * n;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dequant_int32_fp16_kernel<<<blocks, threads, 0, stream>>>(
        accumulators.data_ptr<std::int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        activation_scales.data_ptr<float>(),
        weight_scales.data_ptr<float>(),
        bias.numel() == 0 ? nullptr : bias.data_ptr<float>(),
        elements,
        n,
        activation_scales.numel() == m,
        bias.numel() == n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor rns_matmul_dequant_fp16_out_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor moduli,
    torch::Tensor reciprocals,
    torch::Tensor pairwise_inverses,
    torch::Tensor compact_lut,
    std::int64_t modulus_product,
    std::int64_t lut_channels,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    torch::Tensor bias,
    torch::Tensor accumulators,
    torch::Tensor output) {
    validate_matrix_cuda_contiguous(a, "a", 3);
    validate_matrix_cuda_contiguous(b, "b", 3);
    TORCH_CHECK(a.scalar_type() == torch::kInt8 && b.scalar_type() == torch::kInt8,
                "a and b must be centered int8 residue planes");
    TORCH_CHECK(a.device() == b.device(), "a and b must share a device");
    TORCH_CHECK(a.size(0) == b.size(0), "channel count mismatch");
    TORCH_CHECK(a.size(2) == b.size(1), "K dimension mismatch");

    const int channels = static_cast<int>(a.size(0));
    const int m = static_cast<int>(a.size(1));
    const int k = static_cast<int>(a.size(2));
    const int n = static_cast<int>(b.size(2));
    TORCH_CHECK(channels >= 2 && channels <= 10,
                "v0.7 fused epilogue supports 2..10 channels");
    TORCH_CHECK(k % 4 == 0 && n % 4 == 0,
                "RNS Tensor Core path requires K and N multiples of 4");
    TORCH_CHECK(lut_channels >= 0 && lut_channels <= 2 && lut_channels <= channels,
                "lut_channels must be 0..2 and <= channel count");

    validate_matrix_cuda_contiguous(moduli, "moduli", 1);
    validate_matrix_cuda_contiguous(reciprocals, "reciprocals", 1);
    validate_matrix_cuda_contiguous(pairwise_inverses, "pairwise_inverses", 2);
    validate_matrix_cuda_contiguous(compact_lut, "compact_lut", 3);
    TORCH_CHECK(moduli.scalar_type() == torch::kInt32,
                "moduli must be int32");
    TORCH_CHECK(reciprocals.scalar_type() == torch::kInt64,
                "reciprocals must be int64");
    TORCH_CHECK(pairwise_inverses.scalar_type() == torch::kInt32,
                "pairwise_inverses must be int32");
    TORCH_CHECK(compact_lut.scalar_type() == torch::kInt16,
                "compact_lut must be int16");
    TORCH_CHECK(moduli.numel() == channels && reciprocals.numel() == channels,
                "constant vector length mismatch");
    TORCH_CHECK(pairwise_inverses.size(0) == channels &&
                pairwise_inverses.size(1) == channels,
                "pairwise inverse shape mismatch");
    TORCH_CHECK(compact_lut.size(0) == channels &&
                compact_lut.size(1) == 4 && compact_lut.size(2) == 256,
                "compact LUT shape mismatch");

    validate_matrix_cuda_contiguous(accumulators, "accumulators", 3);
    TORCH_CHECK(accumulators.scalar_type() == torch::kInt32,
                "accumulators must be int32");
    TORCH_CHECK(accumulators.size(0) == channels &&
                accumulators.size(1) == m && accumulators.size(2) == n,
                "accumulator shape mismatch");
    validate_scales_and_output(
        activation_scales, weight_scales, bias, output, m, n, a.device());

    TORCH_CHECK(a.device() == moduli.device() &&
                a.device() == reciprocals.device() &&
                a.device() == pairwise_inverses.device() &&
                a.device() == compact_lut.device() &&
                a.device() == accumulators.device(),
                "all RNS tensors must share a device");

    c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_cublas_int8_batched(
        a, b, accumulators, channels, m, k, n, stream);

    const std::int64_t elements = static_cast<std::int64_t>(m) * n;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);

#define LAUNCH_RNS_EPILOGUE(CHANNELS) \
    fused_reduce_garner_dequant_fp16_kernel<CHANNELS> \
        <<<blocks, threads, 0, stream>>>( \
            accumulators.data_ptr<std::int32_t>(), \
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
            moduli.data_ptr<std::int32_t>(), \
            reciprocals.data_ptr<std::int64_t>(), \
            pairwise_inverses.data_ptr<std::int32_t>(), \
            compact_lut.data_ptr<std::int16_t>(), \
            modulus_product, \
            activation_scales.data_ptr<float>(), \
            weight_scales.data_ptr<float>(), \
            bias.numel() == 0 ? nullptr : bias.data_ptr<float>(), \
            elements, n, static_cast<int>(lut_channels), \
            activation_scales.numel() == m, bias.numel() == n)

    switch (channels) {
        case 2: LAUNCH_RNS_EPILOGUE(2); break;
        case 3: LAUNCH_RNS_EPILOGUE(3); break;
        case 4: LAUNCH_RNS_EPILOGUE(4); break;
        case 5: LAUNCH_RNS_EPILOGUE(5); break;
        case 6: LAUNCH_RNS_EPILOGUE(6); break;
        case 7: LAUNCH_RNS_EPILOGUE(7); break;
        case 8: LAUNCH_RNS_EPILOGUE(8); break;
        case 9: LAUNCH_RNS_EPILOGUE(9); break;
        case 10: LAUNCH_RNS_EPILOGUE(10); break;
        default: TORCH_CHECK(false, "unsupported channel count");
    }
#undef LAUNCH_RNS_EPILOGUE

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "quantize_fp16_out",
        &quantize_fp16_out_cuda,
        "Fused FP16 to INT8 quantization into preallocated output (CUDA)");
    m.def(
        "quantize_encode_fp16_out",
        &quantize_encode_fp16_out_cuda,
        "Fused FP16 quantization plus centered RNS encoding (CUDA)");
    m.def(
        "native_int8_mm_out",
        &native_int8_mm_out_cuda,
        "Native cuBLAS INT8 GEMM into preallocated INT32 output (CUDA)");
    m.def(
        "native_int8_dequant_fp16_out",
        &native_int8_dequant_fp16_out_cuda,
        "Native cuBLAS INT8 GEMM plus fused dequant/bias to FP16 (CUDA)");
    m.def(
        "rns_matmul_dequant_fp16_out",
        &rns_matmul_dequant_fp16_out_cuda,
        "RNS cuBLAS channels plus fused Garner/dequant/bias to FP16 (CUDA)");
}
