#pragma once

#include <cuda_runtime.h>

template<typename U>
__global__ void softmax_ce_forward_kernel(const U* __restrict__ Z, const int* __restrict__ labels, U* __restrict__ probs, U* __restrict__ lossp, int M, int C) {
    int m = blockIdx.x * blockDim.x + threadIdx.x; //what example we are currently looking at
    if (m >= M) return;
    const U* z = Z + m * C; //pointer to where in Z we are, m is our currnet example and C is containers
    U* p = probs + m * C;
    U mx = z[0];
    for (int c = 1; c < C; ++c) mx = z[c] > mx ? z[c] : mx;
    U sum = U(0);
    for (int c = 0; c < C; ++c) { U e = exp(z[c] - mx); p[c] = e; sum += e; }
    for (int c = 0; c < C; ++c) p[c] /= sum;
    lossp[m] = -log(p[labels[m]] + U(1e-9));
}

template<typename U>
__global__ void softmax_ce_backward_kernel(const U* __restrict__ gscalar, const U* __restrict__ probs, const int* __restrict__ labels, U* __restrict__ dZ, int M, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * C) {
        int m = idx / C, c = idx % C;
        U t = (c == labels[m]) ? U(1) : U(0);
        //gscalar[0] = 1 but we can add scaled  (perfect would be probs[idx] = 1 when c==labels[m]
        dZ[idx] += gscalar[0] * (probs[idx] - t) / U(M);
    }
}
