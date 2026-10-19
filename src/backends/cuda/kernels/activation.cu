#include <infanttorch/detail/launch.hpp>
#include "activation.cuh"
#include "launch_utils.cuh"

template<typename T>
void launch_relu(const T* x, T* out, std::size_t n) {
    relu_kernel<T> << <grid(n), BLOCK >> > (x, out, n);
    check("relu");
}

template<typename T>
void launch_relu_backward(const T* gout, const T* x, T* gx, std::size_t n) {
    relu_backward_kernel<T> << <grid(n), BLOCK >> > (gout, x, gx, n);
    check("relu_backward");
}

#define INSTANTIATE(T)                                                              \
    template void launch_relu<T>(const T*, T*, std::size_t);                        \
    template void launch_relu_backward<T>(const T*, const T*, T*, std::size_t);     \

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
