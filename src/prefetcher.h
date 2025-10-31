#pragma once
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <mutex>
#include "vgmm_controller.h"

class Prefetcher {
public:
    Prefetcher(VGMMController &ctrl);
    ~Prefetcher();

    // hint next ids to prefetch (order matters)
    void hint_prefetch(const std::vector<std::string> &next_ids);

    void start();
    void stop();

private:
    VGMMController &ctrl;
    std::thread worker;
    std::atomic<bool> running;
    std::vector<std::string> queue;
    std::mutex qlock;

    void run_loop();
};
