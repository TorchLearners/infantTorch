#include <infanttorch/detail/launch.hpp>
#include "loss.cuh"
#include "launch_utils.cuh"

template<typename T>
void launch_softmax_ce_forward(const T* Z, const int* labels, T* probs, T* lossp, int M, int C) {
    softmax_ce_forward_kernel<T> << <grid(static_cast<std::size_t>(M)), BLOCK >> > (Z, labels, probs, lossp, M, C);
    check("softmax_ce_forward");
}
template<typename T>
void launch_softmax_ce_backward(const T* gscalar, const T* probs, const int* labels, T* dZ, int M, int C) {
    softmax_ce_backward_kernel<T> << <grid(static_cast<std::size_t>(M) * C), BLOCK >> > (gscalar, probs, labels, dZ, M, C);
    check("softmax_ce_backward");
}

#define INSTANTIATE(T)                                                              \
    template void launch_softmax_ce_forward<T>(const T*, const int*, T*, T*, int, int); \
    template void launch_softmax_ce_backward<T>(const T*, const T*, const int*, T*, int, int); \

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
