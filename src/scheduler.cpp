#include "scheduler.h"

void LRUScheduler::touch(const std::string &id) {
    std::lock_guard<std::mutex> g(m);
    for (auto it = lru.begin(); it != lru.end(); ++it) {
        if (*it == id) {
            lru.erase(it);
            lru.push_back(id);
            return;
        }
    }
    lru.push_back(id);
}

void LRUScheduler::remove(const std::string &id) {
    std::lock_guard<std::mutex> g(m);
    for (auto it = lru.begin(); it != lru.end(); ++it) {
        if (*it == id) { lru.erase(it); return; }
    }
}

std::string LRUScheduler::pick_victim() {
    std::lock_guard<std::mutex> g(m);
    if (lru.empty()) return "";
    std::string v = lru.front();
    lru.pop_front();
    return v;
}
    