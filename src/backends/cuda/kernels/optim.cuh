#pragma once

#include <type_traits>
#include <cuda_runtime.h>

// w[i] -= lr * g[i]
template<typename U>
__global__ void sgd_update_kernel(U* __restrict__ w, const U* __restrict__ g, U lr, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * g[i];
}


template<typename U>
__global__ void adam_update_kernel(U* __restrict__ w, U* __restrict__ m, U* __restrict__ v, const U* __restrict__ g, U lr, U momentum, U decay, U eps, U step, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        m[i] = (momentum * m[i] + (1-momentum) * g[i]);
        v[i] = (decay * v[i] + (1-decay) * g[i] * g[i]);
        
        U m_hat = m[i] / (1 - pow(momentum, step));
        U v_hat = v[i] / (1 - pow(decay, step));

        w[i] = w[i] - lr * m_hat / (sqrt(v_hat) + eps);
    }
}