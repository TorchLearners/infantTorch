#pragma once

#include <cuda_runtime.h>

//c = a x b (a is (M, K) b is (K, N) so c is (M,N))
template<typename U, int T>
__global__ void matmul_kernel(const U* __restrict__ A, const U* __restrict__ B, U* __restrict__ C, int M, int N, int K) {
    __shared__ U A_cache[T * T];
    __shared__ U B_cache[T * T];
    //tiling helps becuz cache hits are more importnat than operations
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    int num_sections = (K + T - 1) / T;

    U acc = U(0);

    for (int section = 0; section < num_sections; section++) {
        int tCol_A = section * T + threadIdx.x;
        int tRow_B = section * T + threadIdx.y;
        A_cache[(row % T) * T + (col % T)] = (row < M && tCol_A < K ? A[row * K + tCol_A] : 0);
        B_cache[(row % T) * T + (col % T)] = (tRow_B < K && col < N ? B[tRow_B * N + col] : 0);
        __syncthreads();

        for (int i = 0; i < T; i++) {
            acc += A_cache[(row % T) * T + i] * B_cache[i * T + (col % T)];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// dA[m,k] += sum_n dC[m,n] * B[k,n]
template<typename U>
__global__ void matmul_backward_A_kernel(const U* __restrict__ dC, const U* __restrict__ B,
    U* __restrict__ dA, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && k < K) {
        U acc = U(0);
        for (int n = 0; n < N; ++n) acc += dC[m * N + n] * B[k * N + n];
        dA[m * K + k] += acc;
    }
}

// dB[k,n] += sum_m A[m,k] * dC[m,n]
template<typename U>
//__restrict__ 
__global__ void matmul_backward_B_kernel(const U* __restrict__ A, const U* __restrict__ dC,
    U* __restrict__ dB, int M, int N, int K) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K && n < N) {
        U acc = U(0);
        //TODO: call from memory is unoptimal
        for (int m = 0; m < M; ++m) acc += A[m * K + k] * dC[m * N + n];
        dB[k * N + n] += acc;
    }
}

//add column wise
template<typename U>
__global__ void bias_add_kernel(const U* __restrict__ in, const U* __restrict__ b,
    U* __restrict__ out, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) out[idx] = in[idx] + b[idx % N];
}

//bias acts on everything in a col so we just add grad of col
template<typename U>
__global__ void bias_grad_kernel(const U* __restrict__ gout, U* __restrict__ db, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        U acc = U(0);
        for (int m = 0; m < M; ++m) acc += gout[m * N + n];
        db[n] += acc;
    }
}
