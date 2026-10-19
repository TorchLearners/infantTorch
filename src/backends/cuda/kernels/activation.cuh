#pragma once

#include <cuda_runtime.h>

template<typename U>
__global__ void relu_kernel(const U* __restrict__ x, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = x[idx] > 0 ? x[idx] : 0;
}

// dx += (x > 0) ? dz same reaosning above : 0
template<typename U>
__global__ void relu_backward_kernel(const U* __restrict__ gout, const U* __restrict__ x, U* __restrict__ gx, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        gx[idx] += x[idx] > U(0) ? gout[idx] : U(0);
    }
}
