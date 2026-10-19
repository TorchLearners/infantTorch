#include <infanttorch/detail/launch.hpp>
#include "reduction.cuh"
#include "launch_utils.cuh"

template<typename T>
void launch_mean_reduce_kernel(const T* in, T* out, int M) {
    mean_reduce_kernel<T> << <1, 1 >> > (in, out, M);
    check("mean_reduce_kernel");
}

#define INSTANTIATE(T)                                                              \
    template void launch_mean_reduce_kernel<T>(const T*, T*, int); \

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
