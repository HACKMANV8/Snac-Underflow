#pragma once
#include <list>
#include <mutex>
#include <string>

// LRU scheduler: stores ids, oldest at front
class LRUScheduler {
public:
    LRUScheduler() = default;
    void touch(const std::string &id); // mark MRU
    void remove(const std::string &id);
    // returns victim id (empty if none)
    std::string pick_victim();

private:
    std::list<std::string> lru;
    std::mutex m;
};
