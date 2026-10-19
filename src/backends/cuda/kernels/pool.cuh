#pragma once

#include <cuda_runtime.h>

// mat[N,C,H,W] becomesout [N,C,oH,oW]
template<typename U>
__global__ void maxpool_kernel(const U* __restrict__ mat, U* __restrict__ out, int* __restrict__ argmax_cache, int N, int C, int H, int W, int pool, int oH, int oW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C * oH * oW) {
        int ow = idx % oW;
        int oh = (idx / oW) % oH;
        int c = (idx / (oW * oH)) % C;
        int n = idx / (oW * oH * C);
        size_t sliceBase = ((size_t)n * C + c) * H * W;
        int base = (oh * pool) * W + (ow * pool);
        U best = mat[sliceBase + base];
        int best_idx = (int)(sliceBase + base);
        for (int dx = 0; dx < pool; dx++) {
            for (int dy = 0; dy < pool; dy++) {
                int cur = base + dx * W + dy;
                if (mat[sliceBase + cur] > best) {
                    best = mat[sliceBase + cur];
                    best_idx = (int)(sliceBase + cur);
                }
            }
        }
        out[idx] = best;
        argmax_cache[idx] = best_idx;
    }
}

template<typename U>
__global__ void maxpool_backward_kernel(const U* __restrict__ dOut, U* __restrict__ dIn, const int* __restrict__ argmax_cache, int N, int C, int oH, int oW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C * oH * oW) {
        dIn[argmax_cache[idx]] += dOut[idx];
    }
}
