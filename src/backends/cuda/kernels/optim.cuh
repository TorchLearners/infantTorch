#pragma once

#include <cuda_runtime.h>

// w[i] -= lr * g[i]
template<typename U>
__global__ void sgd_update_kernel(U* __restrict__ w, const U* __restrict__ g, U lr, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * g[i];
}
