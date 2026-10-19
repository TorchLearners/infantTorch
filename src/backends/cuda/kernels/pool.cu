#include <infanttorch/detail/launch.hpp>
#include "pool.cuh"
#include "launch_utils.cuh"

template <typename T>
void launch_maxpool(const T* mat, T* out, int* argmax_cache, int N, int C, int H, int W, int pool, int oH, int oW) {
    maxpool_kernel<T> << <grid((std::size_t)N * C * oH * oW), BLOCK >> > (mat, out, argmax_cache, N, C, H, W, pool, oH, oW);
    check("maxpool");
}

template <typename T>
void launch_backward_maxpool(const T* dOut, T* dIn, const int* argmax_cache, int N, int C, int oH, int oW) {
    maxpool_backward_kernel<T> << <grid((std::size_t)N * C * oH * oW), BLOCK >> > (dOut, dIn, argmax_cache, N, C, oH, oW);
    check("maxpool_backward");
}

#define INSTANTIATE(T)                                                              \
    template void launch_maxpool<T>(const T*, T*, int*, int,int,int,int,int,int,int); \
    template void launch_backward_maxpool<T>(const T*, T*, const int*, int,int,int,int); \

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
