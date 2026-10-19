#pragma once

#include <cuda_runtime.h>

template<typename U>
__global__ void accumulate_loss_correct(U* in, const U* loss, const U* logits, int* labels, int M, int C) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < M) {
        atomicAdd(&in[0], loss[0]);
        U mx = logits[idx * C + 0];
        int pred = 0;
        for (int i = 1; i < C; i++) {
            if (mx < logits[idx * C + i]) {
                mx = logits[idx * C + i];
                pred = i;
            }
        }
        if (pred == labels[idx]) atomicAdd(&in[1], (U)1);
    }
}
