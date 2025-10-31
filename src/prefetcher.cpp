#include "prefetcher.h"
#include <chrono>
#include <iostream>

Prefetcher::Prefetcher(VGMMController &c) : ctrl(c), running(false) {}

Prefetcher::~Prefetcher() { stop(); }

void Prefetcher::start() {
    running = true;
    worker = std::thread(&Prefetcher::run_loop, this);
}

void Prefetcher::stop() {
    running = false;
    if (worker.joinable()) worker.join();
}

void Prefetcher::hint_prefetch(const std::vector<std::string> &next_ids) {
    std::lock_guard<std::mutex> g(qlock);
    queue = next_ids; // overwrite with latest hints
}

void Prefetcher::run_loop() {
    while (running) {
        std::vector<std::string> work;
        {
            std::lock_guard<std::mutex> g(qlock);
            work = queue;
            queue.clear();
        }
        for (auto &id : work) {
            std::cout << "[Prefetch] prefetching " << id << "\n";
            // blocking prefetch for simplicity (still real movement)
            ctrl.load_to_vram(id);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
