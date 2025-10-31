#include <cuda_runtime.h>
#include <iostream>

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float val = 0;
        for (int e = 0; e < K; e++)
            val += A[row * K + e] * B[e * N + col];
        C[row * N + col] = val;
    }
}

extern "C" void run_matmul_layer(float* A, float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "[CUDA Error] " << cudaGetErrorString(err) << std::endl;
}
