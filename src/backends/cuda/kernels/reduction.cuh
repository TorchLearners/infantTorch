#pragma once

#include <cuda_runtime.h>

template<typename U>
__global__ void mean_reduce_kernel(const U* __restrict__ in, U* __restrict__ out, int M) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        U s = U(0);
        for (int i = 0; i < M; ++i) s += in[i];
        out[0] = s / U(M);
    }
}
