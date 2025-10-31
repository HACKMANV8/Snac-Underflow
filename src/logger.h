#pragma once
#include <fstream>
#include <string>

class Logger {
public:
    explicit Logger(const std::string& filename);
    ~Logger();
    void log_line(size_t free_vram, size_t total_vram, size_t vram_used, size_t vram_limit, size_t vram_total);

private:
    std::ofstream file;
};
