#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

__global__ void init_matrix_kernel(float* data, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

void run_matmul_layer(void* dA, void* dB, void* dC, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta = 0.0f;

    // Perform C = A * B
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        static_cast<const float*>(dA), m,
        static_cast<const float*>(dB), k,
        &beta,
        static_cast<float*>(dC), m
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cuBLAS] Matrix multiply failed!" << std::endl;
    }

    cudaDeviceSynchronize();
    cublasDestroy(handle);
}
