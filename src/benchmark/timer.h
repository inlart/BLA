#pragma once

#include <chrono>

class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    
public:
Timer() : start(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - start).count() / 1000.0;
    }
};
