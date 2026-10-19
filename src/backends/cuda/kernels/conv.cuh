#pragma once

#include <cuda_runtime.h>

// out[n,co,oh,ow] = sum ci,ki,kj in[n,ci,oh+ki,ow+kj] * filt[co,ci,ki,kj]
template<typename U>
__global__ void conv2d_forward_kernel(const U* __restrict__ in, const U* __restrict__ filt,
    U* __restrict__ out,
    int N, int Cin, int H, int W,
    int Cout, int K1, int K2, int oH, int oW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Cout * oH * oW) return;
    int ow = idx % oW;
    int oh = (idx / oW) % oH;
    int co = (idx / (oW * oH)) % Cout;
    int n = idx / (oW * oH * Cout);

    U acc = U(0);
    for (int ci = 0; ci < Cin; ++ci) {
        const U* img = in + ((size_t)n * Cin + ci) * H * W;
        const U* f = filt + ((size_t)co * Cin + ci) * K1 * K2;
        for (int ki = 0; ki < K1; ++ki)
            for (int kj = 0; kj < K2; ++kj)
                acc += img[(oh + ki) * W + (ow + kj)] * f[ki * K2 + kj];
    }
    out[idx] = acc;
}

// dW[co,ci,ki,kj] += sum n,oh,ow in[n,ci,oh+ki,ow+kj] * dOut[n,co,oh,ow]

// grid = (P, S): blockIdx.x = weight p, blockIdx.y = chunk of the R = N*oH*oW terms
template<typename U>
__global__ void conv2d_dW_kernel(const U* __restrict__ in, const U* __restrict__ dOut, U* __restrict__ dW,
                                 int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW,
                                 int chunk) {
    __shared__ U partial[256];

    // TODO: decode p = blockIdx.x into (co, ci, ki, kj).
    //       Your current kernel already does this from 'idx'; reuse those four lines with p instead.

    // TODO: R, r0, r1 exactly as in the bias kernel (what is R here?)
    const int p = blockIdx.x; //blockIdx.y = "chunk "
    const int R = N * oH * oW;
    const int r0 = blockIdx.y * chunk;
    const int r1 = min(R, r0 + chunk);

    const int co = p / (Cin * K1 * K2);
    const int cin = (p % (Cin * K1 * K2)) / (K1* K2); //p = co * (Cin * K1 * K2) + ci * (K1 * K2) + ki * K2 + kj
    const int ki = (p % (K1*K2)) / K2;
    const int kj = p % K2;

    U acc = U(0);
    /*
    gridDim	number of blocks in each direction	(72, 22, 1)
    blockIdx	which block this is	.x = weight p (0–71), .y = chunk (0–21)
    blockDim	number of threads in each block	(256, 1, 1) .x = # of threads
    threadIdx	which thread within its block	x = 0–255
    */
    for (int r = r0 + threadIdx.x; r < r1; r+=blockDim.x) {
        const int n  = r / (oH * oW);
        const int oh = (r % (oH * oW)) / oW;
        const int ow = r % oW;

        const U x = in[n * (Cin * H * W) + cin * (H * W) + (oh + ki) * W + (ow + kj)];
        const U g = dOut[n * (Cout * oH * oW) + co * (oH * oW) + oh * oW + ow];

        acc += x * g;
    }

    partial[threadIdx.x] = acc;
    __syncthreads();
    //256 block dim?
    for(int stride = blockDim.x / 2; stride > 0; stride/=2){
        if(threadIdx.x < stride){
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // TODO: the same tree reduction as the bias kernel
    if(threadIdx.x == 0){
        atomicAdd(&dW[p], partial[0]); 
    }
}

// dIn[n,ci,h,w] += sum_{co,ki,kj valid} dOut[n,co,h-ki,w-kj] * filt[co,ci,ki,kj]
template<typename U>
__global__ void conv2d_dIn_kernel(const U* __restrict__ dOut, const U* __restrict__ filt,
    U* __restrict__ dIn,
    int N, int Cin, int H, int W,
    int Cout, int K1, int K2, int oH, int oW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Cin * H * W) return;
    int w = idx % W;
    int h = (idx / W) % H;
    int ci = (idx / (W * H)) % Cin;
    int n = idx / (W * H * Cin);

    U acc = U(0);
    for (int co = 0; co < Cout; ++co) {
        const U* dImg = dOut + ((size_t)n * Cout + co) * oH * oW;
        const U* f = filt + ((size_t)co * Cin + ci) * K1 * K2;
        for (int ki = 0; ki < K1; ++ki) {
            int oh = h - ki; if (oh < 0 || oh >= oH) continue;
            for (int kj = 0; kj < K2; ++kj) {
                int ow = w - kj; if (ow < 0 || ow >= oW) continue;
                acc += dImg[oh * oW + ow] * f[ki * K2 + kj];
            }
        }
    }
    dIn[idx] += acc;
}

// each chanel has a bias
template<typename U>
__global__ void conv_bias_forward_kernel(const U* __restrict__ in, const U* __restrict__ bias,
    U* __restrict__ out, int N, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C * HW) {
        int c = (idx / HW) % C;
        out[idx] = in[idx] + bias[c];
    }
}

// dBias[c] += sum over all examples and HW (of dOut)
// grid = (C, S): blockIdx.x = channel, blockIdx.y = which chunk of the R = N*HW terms
template<typename U>
__global__ void conv_bias_grad_kernel(const U* __restrict__ dOut, U* __restrict__ dBias,
                                      int N, int C, int HW, int chunk) {
    
    __shared__ U partial[256];                 // one slot per thread; must match the block size

    const int c  = blockIdx.x;                 // which channel bias this block works on
    const int R  = N * HW;                     // total in this bias's sum
    const int r0 = blockIdx.y * chunk;         // this block's chunk: terms [r0, r1)
    const int r1 = min(r0 + chunk, R);

    // ---- step 1: each thread sums a strided slice of the chunk
    U acc = U(0);
    for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) {
        // TODO: decode r into (n, hw). Term r is the r-th value of channel c across all images.
        //       Hint: there are HW values per image.

        int n  = r / HW; //there are HW values per example thus this is our example 
        int hw = r % HW;  //mod to get the pixel within the channel at the current example
        acc += dOut[n * C * HW  + c * HW + hw];
    }

    // ---- step 2: tree reduction of the 256 partial sums in shared memory
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            // TODO: one line. Thread t adds the value 'stride' slots away into its own slot.
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();                       // outside the if: every thread must reach it
    }

    // ---- step 3: one thread adds the block's total into the output
    if (threadIdx.x == 0) {
        // TODO: atomicAdd the block total (which slot holds it now?) into dBias[c]
        atomicAdd(&dBias[c], partial[0]);
    }
}
