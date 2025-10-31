#include "logger.h"

Logger::Logger(const std::string& path) {
    file.open(path, std::ios::out | std::ios::trunc);
    if (file.is_open())
        file << "FreeVRAM,TotalVRAM,UsedVRAM,Pinned,Evicted\n";
}

Logger::~Logger() {
    if (file.is_open()) file.close();
}

void Logger::log_line(size_t free_vram, size_t total_vram, size_t used_vram, size_t pinned, size_t evicted) {
    if (file.is_open())
        file << free_vram << "," << total_vram << "," << used_vram << "," << pinned << "," << evicted << "\n";
}
