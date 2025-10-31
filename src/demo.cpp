#include "vgmm_controller.h"
#include "prefetcher.h"
#include "simple_kernel.cuh"

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <cuda_runtime.h>


void cpu_matmul(const std::vector<float>& A, const std::vector<float>& B,
                std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

int main() {
    std::cout << "\n=====================================\n";
    std::cout << "  Virtual GPU Memory Manager (Final)\n";
    std::cout << "=====================================\n\n";

    Logger logger("vgmm_log.csv");
    VGMMController vgmm(128ULL * 1024 * 1024, logger); // 128 MB VRAM limit (forces eviction)
    Prefetcher prefetcher(vgmm);
    prefetcher.start();

    const int NUM_LAYERS = 6;
    const int M = 256, N = 256, K = 256; // matrix dimensions
    const size_t layer_bytes = M * K * sizeof(float);

    std::cout << "[Setup] Registering layer tensors...\n";
    for (int i = 0; i < NUM_LAYERS; ++i) {
        vgmm.register_tensor("A_" + std::to_string(i), layer_bytes);
        vgmm.register_tensor("B_" + std::to_string(i), layer_bytes);
        vgmm.register_tensor("C_" + std::to_string(i), layer_bytes);
    }

    std::cout << "\n[Compute] Beginning streaming computation...\n";
    for (int i = 0; i < NUM_LAYERS; ++i) {
        std::string idA = "A_" + std::to_string(i);
        std::string idB = "B_" + std::to_string(i);
        std::string idC = "C_" + std::to_string(i);

        // Hint prefetch for next layer
        if (i + 1 < NUM_LAYERS)
            prefetcher.hint_prefetch({
                "A_" + std::to_string(i + 1),
                "B_" + std::to_string(i + 1),
                "C_" + std::to_string(i + 1)
            });

        // Load to VRAM
        float* dA = static_cast<float*>(vgmm.load_to_vram(idA));
        float* dB = static_cast<float*>(vgmm.load_to_vram(idB));
        float* dC = static_cast<float*>(vgmm.load_to_vram(idC));

        // Fill host tensors with test data
        std::vector<float> hostA(M * K, 1.0f);
        std::vector<float> hostB(K * N, 2.0f);
        std::vector<float> hostC(M * N, 0.0f);
        cudaMemcpy(dA, hostA.data(), hostA.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hostB.data(), hostB.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dC, hostC.data(), hostC.size() * sizeof(float), cudaMemcpyHostToDevice);

        std::cout << "[GPU] Running matmul on layer " << i << "...\n";
        run_matmul_layer(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();

        // Copy result back for verification
        cudaMemcpy(hostC.data(), dC, hostC.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // CPU reference
        std::vector<float> hostC_ref(M * N, 0.0f);
        cpu_matmul(hostA, hostB, hostC_ref, M, N, K);

        // Compute error
        float max_err = 0;
        for (size_t j = 0; j < hostC.size(); ++j)
            max_err = std::max(max_err, std::fabs(hostC[j] - hostC_ref[j]));

        if (max_err < 1e-4)
            std::cout << "[CHECK] ✅ GPU result matches CPU (max error=" << max_err << ")\n";
        else
            std::cout << "[CHECK] ❌ Mismatch detected (max error=" << max_err << ")\n";

        // Mark used for LRU tracking
        vgmm.mark_used(idA);
        vgmm.mark_used(idB);
        vgmm.mark_used(idC);
        vgmm.print_status_and_log();

        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    prefetcher.stop();
    std::cout << "\n[VGMM] Demo complete.\n";
    return 0;
}
