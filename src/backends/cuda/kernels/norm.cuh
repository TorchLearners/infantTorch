#pragma once
#include <cmath>
#include <cuda_runtime.h>


//takes in logins T in of size N, outputs softmax of size O
template<typename U, int K>
__global__ void softmax_good(const U* __restrict__  in, U* __restrict__ out, size_t ROWS, size_t N){
    size_t row = (blockDim.x * blockIdx.x + threadIdx.x) / 32; //32 threads per single row
    if(row >= ROWS) return;

    size_t lane = threadIdx.x % 32;

    U max_arr[K];
    U sum_arr[K];
    U nums[K];
    #pragma unroll
    for(int i = 0; i<K; i++){
        if((lane + 32 *i) >= N){
            max_arr[i] = U(-INFINITY); 
            sum_arr[i] = U(0);
        }else{
            max_arr[i] = in[row * N + lane + 32 * i];
            sum_arr[i] = in[row * N + lane + 32 * i];
        }
    }

    #pragma unroll
    for(int n = K/2; n > 0; n>>=1){
        #pragma unroll
        for(int i = 0; i<n; i++){
            max_arr[i] = max(max_arr[i], max_arr[i+n]);
        }
    }
    #pragma unroll
    for(int n = 16; n>0; n>>=1){
        max_arr[0] = max(max_arr[0], __shfl_down_sync(0xffffffff, max_arr[0], n));
    }
 
    U mx = __shfl_xor_sync(0xffffffff, max_arr[0], (threadIdx.x % 32));

    #pragma unroll
    for(int i = 0; i<K; i++){
        if((lane + 32 * i) >= N){
            nums[i] = 0;
        }else{
            if constexpr (std::is_same_v<U, float>){
                nums[i] = expf(sum_arr[i] - mx);
            }else{
                nums[i] = exp(sum_arr[i] - mx);
            }
        }
    }
    sum_arr[0] = 0;
    #pragma unroll
    for(int i = 0; i<K; i++){
        sum_arr[0] = (sum_arr[0] + nums[i]); 
    }
    #pragma unroll
    for(int n = 16; n>0; n>>=1){
        sum_arr[0] = (sum_arr[0] + __shfl_down_sync(0xffffffff, sum_arr[0], n));
    }

    U sum = __shfl_xor_sync(0xffffffff, sum_arr[0], (threadIdx.x % 32));

    #pragma unroll
    for(int i = 0; i<K; i++){
        size_t out_idx =  lane + 32 * i;
        if(out_idx >= N) continue;
        else{
            out[row * N + out_idx] = nums[i] / sum;
        }
    }

}

//remmeber N is super big
template<typename U>
__global__ void softmax_bad(const U* __restrict__ in, U* __restrict__ out, size_t M, size_t C, size_t N){
    __shared__ U warp_vals[32];
    size_t row = blockIdx.x;
    if(row >= (M*C)) return;

    size_t offset = row * N;

    size_t lane = threadIdx.x % 32;
    size_t warp = threadIdx.x / 32;
    size_t num_warps = blockDim.x / 32;

    U mx = U(-INFINITY);

    for(size_t i = threadIdx.x; i < N; i += blockDim.x){
        mx = max(mx, in[offset + i]);
    }

    for(int n = 16; n > 0; n >>= 1){
        mx = max(mx, __shfl_down_sync(0xffffffff, mx, n));
    }

    if(lane == 0) warp_vals[warp] = mx;
    __syncthreads();

    if(warp == 0){
        mx = lane < num_warps ? warp_vals[lane] : U(-INFINITY);
        for(int n = 16; n > 0; n >>= 1){
            mx = max(mx, __shfl_down_sync(0xffffffff, mx, n));
        }
        if(lane == 0) warp_vals[0] = mx;
    }
    __syncthreads();
    mx = warp_vals[0];
    __syncthreads();

    U sum_term = U(0);
    for(size_t i = threadIdx.x; i < N; i += blockDim.x){
        if constexpr (std::is_same_v<U, float>){
            sum_term += expf(in[offset + i] - mx);
        }else{
            sum_term += exp(in[offset + i] - mx);
        }
    }
    for(int n = 16; n > 0; n >>= 1){
        sum_term += __shfl_down_sync(0xffffffff, sum_term, n);
    }
    if(lane == 0) warp_vals[warp] = sum_term;
    __syncthreads();
    if(warp == 0){
        sum_term = lane < num_warps ? warp_vals[lane] : U(0);
        for(int n = 16; n > 0; n >>= 1){
            sum_term += __shfl_down_sync(0xffffffff, sum_term, n);
        }
        if(lane == 0) warp_vals[0] = sum_term;
    }
    __syncthreads();
    sum_term = warp_vals[0];

    for(size_t i = threadIdx.x; i < N; i += blockDim.x){
        if constexpr (std::is_same_v<U, float>){
            out[offset + i] = expf(in[offset + i] - mx) / sum_term;
        }else{
            out[offset + i] = exp(in[offset + i] - mx) / sum_term;
        }
    }
}