#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstddef>

namespace {
    constexpr int BLOCK = 256;
    inline unsigned grid(std::size_t n) {
        return static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    }
    inline void check(const char* what) {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("launch ") + what + " failed: "
                + cudaGetErrorString(e));
    }
}
