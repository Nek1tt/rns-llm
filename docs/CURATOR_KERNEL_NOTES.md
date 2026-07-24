# Notes on the curator's tiled matrix multiplication kernel

The provided kernel is useful because it expresses the correct GPU dataflow:
load a tile of A and B into shared memory, synchronize, accumulate a dot-product
fragment, then move to the next K tile.

For production use it needs these corrections:

1. declare the function as `__global__`;
2. support rectangular `M x K` and `K x N` matrices;
3. add bounds checks for dimensions not divisible by 16;
4. declare shared tiles once per kernel, not inside the loop;
5. use INT8 packed loads and `__dp4a` or cuBLAS/Tensor Core paths;
6. map the RNS channel to `blockIdx.z` or a strided-batched GEMM;
7. delay modulo and CRT/Garner reconstruction until the dot product is complete;
8. reuse output/accumulator workspaces and avoid per-call allocations.

The current repository's `rns_gemm_dp4a_kernel` and cuBLAS fused path implement
these principles. The Colab benchmark compares them directly.
