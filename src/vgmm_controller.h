#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include "scheduler.h"
#include "logger.h"

struct TensorInfo {
    size_t size_bytes;
    void* dev_ptr;
    void* host_ptr;
    bool in_vram;
};

class VGMMController {
public:
    VGMMController(size_t vram_limit, Logger& logger);

    void register_tensor(const std::string& id, size_t size_bytes);
    void* load_to_vram(const std::string& id);
    void evict(const std::string& id);
    void mark_used(const std::string& id);
    void print_status_and_log();

    size_t get_size_bytes(const std::string& id);
    size_t get_vram_used() const;

private:
    std::unordered_map<std::string, TensorInfo> tensors;
    size_t vram_used;
    size_t vram_limit;
    size_t vram_total;
    Logger& logger;
    Scheduler scheduler;
    std::mutex mtx;

    bool has_enough_vram(size_t bytes) const;
    void try_evict_until_free(size_t required);
};
