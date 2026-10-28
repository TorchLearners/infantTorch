#pragma once

#include <cuda_runtime.h>
#include <cuda/pipeline>

//c = a x b (a is (M, K) b is (K, N) so c is (M,N))
template<typename U, int T>
__global__ void matmul_kernel(const U* __restrict__ A, const U* __restrict__ B, U* __restrict__ C, int M, int N, int K) {
    __shared__ U A_cache[2][T * T]; //async buffers
    __shared__ U B_cache[2][T * T];


    int row = blockIdx.y * T + threadIdx.y; //will be row of a (need col A [based on thread idx])
    int col = blockIdx.x * T + threadIdx.x; //will be col of b (need row of b [based on thread idx])
    int num_sections = (K + T - 1) / T;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    //load memory into cache[a]
    auto load = [&](int section, int buf) {
        pipe.producer_acquire(); //acquries the buffer
        int tCol_A = section * T + tx;
        int tRow_B = section * T + ty;
        U* a_dst = &A_cache[buf][ty * T + tx];
        U* b_dst = &B_cache[buf][ty * T + tx];
        if (row < M && tCol_A < K) cuda::memcpy_async(a_dst, &A[row * K + tCol_A], cuda::aligned_size_t<alignof(U)>(sizeof(U)), pipe);
        else *a_dst = U(0);
        if (tRow_B < K && col < N) cuda::memcpy_async(b_dst, &B[tRow_B * N + col], cuda::aligned_size_t<alignof(U)>(sizeof(U)), pipe);
        else *b_dst = U(0);
        pipe.producer_commit(); //commits the buffer
    };

    U acc = U(0);
    load(0,0);

    for (int section = 0; section < num_sections; section++) {
        const int buf = section & 1;
        if(section + 1 < num_sections){ //load next one
            load(section+1, buf ^ 1);
        }
        //wait for first one in queue aka buf 0 for section 0 buf 1 for section 1 etc
        pipe.consumer_wait(); //pauses thread until the most recent buffer (aka s0 for section 0 [queues up s1] s1 for section 1 [queues up s0] s0 again for section 2)
        __syncthreads();
        for (int i = 0; i < T; i++) {
            acc += A_cache[buf][threadIdx.y * T + i] * B_cache[buf][(i*T)+threadIdx.x];
        }
        __syncthreads();
        pipe.consumer_release(); //releases current buffer
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// dA[m,k] += sum_n dC[m,n] * B[k,n]
template<typename U, bool RELU>
__global__ void matmul_backward_A_kernel(const U* __restrict__ B, const U* __restrict__ C, const U* __restrict__ dC, 
    U* __restrict__ dA, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && k < K) {
        U acc = U(0);
        for (int n = 0; n < N; ++n){
            if constexpr (RELU){
                acc += (C[m*N +n] > U(0) ? dC[m * N + n] * B[k * N + n] : U(0));
            }else{
                acc += dC[m * N + n] * B[k * N + n];
            }
        }
        dA[m * K + k] += acc;
    }
}

// dB[k,n] += sum_m A[m,k] * dC[m,n]
template<typename U, bool RELU>
//__restrict__ 
__global__ void matmul_backward_B_kernel(const U* __restrict__ A, const U* __restrict__ C, const U* __restrict__ dC,
    U* __restrict__ dB, int M, int N, int K) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K && n < N) {
        U acc = U(0);
        //TODO: call from memory is unoptimal
        for (int m = 0; m < M; ++m){
            if constexpr (RELU){
                acc += (C[m*N+n] > U(0) ? A[m * K + k] * dC[m * N + n] : U(0));
            }else{
                acc += A[m * K + k] * dC[m * N + n];
            }
        }
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
template<typename U, bool RELU>
__global__ void bias_grad_kernel(const U* __restrict__ out, const U* __restrict__ gout, U* __restrict__ db, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        U acc = U(0);
        for (int m = 0; m < M; ++m){
            if constexpr (RELU){
                acc += (out[m * N + n] > U(0) ? gout[m * N + n] : U(0));
            }else{
                acc += gout[m * N + n];
            }
        }
        db[n] += acc;
    }
}


//c = a x b (a is (M, K) b is (K, N) so c is (M,N))
//then bias add is size M 

template<typename U, int T, bool RELU>
__global__ void linear_forward_kernel(const U* __restrict__ A, const U* __restrict__ B, const U* __restrict__ bias, U* __restrict__ out,
    int M, int K, int N
){
    __shared__ U A_cache[T * T];
    __shared__ U B_cache[T * T];
    //tiling helps becuz cache hits are more importnat than operations
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    int num_sections = (K + T - 1) / T;

    U acc = col < N ? bias[col] : U(0);

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

    if (row < M && col < N) {
        if constexpr (RELU) out[row * N + col] = acc > U(0) ? acc : U(0);
        else out[row * N + col] = acc;
    }
    
}

