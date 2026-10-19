/* TODO: Make fast

*/

#pragma once

#include <cuda_runtime.h>
#include <concepts>
#include <numeric>



template <typename U>
__global__ void add_kernel(const U* __restrict__ x, const U* __restrict__ y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}

template <typename U>
__global__ void scalar_add_kernel(const U* __restrict__ x, const U y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y;
    }
}

template <typename U>
__global__ void subtract_kernel(const U* __restrict__ x, const U* __restrict__ y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] - y[idx];
    }
}

template <typename U>
__global__ void scalar_subtract_kernel(const U* __restrict__ x, const U y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] - y;
    }
}

template <typename U>
__global__ void multiply_kernel(const U* __restrict__ x, const U* __restrict__ y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] * y[idx];
    }
}

template <typename U>
__global__ void scalar_multiply_kernel(const U* __restrict__ x, const U y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] * y;
    }
}

template <typename U>
__global__ void divide_kernel(const U* __restrict__ x, const U* __restrict__ y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] / y[idx];
    }
}

template <typename U>
__global__ void scalar_divide_kernel(const U* __restrict__ x, const U y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] / y;
    }
}

template <typename U>
__global__ void pow_kernel(const U* __restrict__ x, const U power, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<U, float>) out[idx] = powf(x[idx], power);
        else out[idx] = static_cast<U>(pow(x[idx], power));
    }
}

template <typename U>
__global__ void exp_kernel(const U* __restrict__ x, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<U, float>) out[idx] = expf(x[idx]);
        else out[idx] = exp(x[idx]);
    }
}

template <typename U>
__global__ void neg_kernel(const U* __restrict__ x, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -x[idx];
    }
}

template<typename U>
__global__ void fill_kernel(U* __restrict__ out, const U val, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = val;
}

//output = x + y
//L = output
//dL/doutput = gout
//dL/dx = dL/doutput * dOutput/dx
//dOuput/dx = 1
//dL/dx = dL/doutput * 1 (same for y)
//so dx += dldoutput
template<typename U>
__global__ void add_backward_kernel(const U* __restrict__ gout, U* __restrict__ gx, U* __restrict__ gy, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        gx[idx] += gout[idx];
        gy[idx] += gout[idx];
    }
}

// z = x * y, dL/dx = dL/dz * dz/dx = y
template<typename U>
__global__ void mul_backward_kernel(const U* __restrict__ gout, const U* __restrict__ x, const U* __restrict__ y, U* __restrict__ gx, U* __restrict__ gy, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        gx[idx] += gout[idx] * y[idx];
        gy[idx] += gout[idx] * x[idx];
    }
}

//pass dL/doutput into dL/dX (dL/dx = dL/dOutput * dOutput/dX (1))
template<typename U>
__global__ void accumulate_kernel(U* __restrict__ dst, const U* __restrict__ src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}
