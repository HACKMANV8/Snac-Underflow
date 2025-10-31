#include "vgmm_controller.h"
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_RET(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
        << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return nullptr; \
    } \
} while(0)

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
        << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return; \
    } \
} while(0)

VGMMController::VGMMController(size_t limit, Logger& logger_)
    : vram_used(0), vram_limit(limit), logger(logger_) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        vram_total = prop.totalGlobalMem;
    else
        vram_total = limit;
}

void VGMMController::register_tensor(const std::string& id, size_t size_bytes) {
    TensorInfo t{};
    t.size_bytes = size_bytes;
    CHECK_CUDA(cudaHostAlloc(&t.host_ptr, size_bytes, cudaHostAllocDefault));
    t.dev_ptr = nullptr;
    t.in_vram = false;
    tensors[id] = t;
    std::cout << "[VGMM] registered " << id << " size " << size_bytes << " bytes (pinned)\n";
}

bool VGMMController::has_enough_vram(size_t bytes) const {
    return (vram_used + bytes) <= vram_limit;
}

void VGMMController::try_evict_until_free(size_t required) {
    while (!has_enough_vram(required)) {
        std::string victim = scheduler.pick_victim();
        if (victim.empty()) break;

        // skip tensors that are locked (in active compute)
        if (tensors[victim].locked) {
            std::cout << "[VGMM] Skipping locked tensor " << victim << " for eviction\n";
            continue;
        }

        std::cout << "[VGMM] Eviction required, freeing space...\n";
        evict(victim);
    }
}


void* VGMMController::load_to_vram(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx);
    auto& t = tensors[id];
    if (t.in_vram) return t.dev_ptr;
    try_evict_until_free(t.size_bytes);

    CHECK_CUDA_RET(cudaMalloc(&t.dev_ptr, t.size_bytes));
    CHECK_CUDA_RET(cudaMemcpy(t.dev_ptr, t.host_ptr, t.size_bytes, cudaMemcpyHostToDevice));

    t.in_vram = true;
    vram_used += t.size_bytes;
    scheduler.touch(id);
    std::cout << "[VGMM] Loaded " << id << " into VRAM (" << t.size_bytes << " bytes). VRAM used: " << vram_used << std::endl;
    return t.dev_ptr;
}

void VGMMController::evict(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx);
    auto& t = tensors[id];
    if (!t.in_vram) return;
    std::cout << "[VGMM] Evicting " << id << " to host\n";
    CHECK_CUDA(cudaMemcpy(t.host_ptr, t.dev_ptr, t.size_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(t.dev_ptr));
    t.dev_ptr = nullptr;
    t.in_vram = false;
    vram_used -= t.size_bytes;
    std::cout << "[VGMM] Evicted " << id << " from VRAM\n";
}

void VGMMController::mark_used(const std::string& id) {
    scheduler.touch(id);
    std::cout << "[VGMM] Marked " << id << " as recently used.\n";
}

void VGMMController::print_status_and_log() {
    size_t free_vram, total_vram;
    CHECK_CUDA(cudaMemGetInfo(&free_vram, &total_vram));
    logger.log_line(free_vram, total_vram, vram_used, vram_limit, vram_total);
    std::cout << "[VGMM] status: VRAM free=" << free_vram
              << " total=" << total_vram
              << " vram_used=" << vram_used << std::endl;
}

size_t VGMMController::get_size_bytes(const std::string& id) { return tensors[id].size_bytes; }
size_t VGMMController::get_vram_used() const { return vram_used; }
void VGMMController::print_status() {
    size_t free_vram, total_vram;
    cudaMemGetInfo(&free_vram, &total_vram);
    std::cout << "[VGMM] status: VRAM free=" << free_vram
              << " total=" << total_vram
              << " vram_used=" << vram_used << std::endl;
}

void VGMMController::sync_device() {
    cudaDeviceSynchronize();
}

void VGMMController::lock_tensor(const std::string& id) {
    if (tensors.find(id) != tensors.end())
        tensors[id].locked = true;
}

void VGMMController::unlock_tensor(const std::string& id) {
    if (tensors.find(id) != tensors.end())
        tensors[id].locked = false;
}
