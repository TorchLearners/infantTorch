#include <infanttorch/detail/launch.hpp>
#include "elementwise.cuh"
#include "launch_utils.cuh"

template<typename T>
void launch_add(const T* a, const T* b, T* out, std::size_t n) {
    add_kernel<T> << <grid(n), BLOCK >> > (a, b, out, n);
    check("add");
}
template<typename T>
void launch_mul(const T* a, const T* b, T* out, std::size_t n) {
    multiply_kernel<T> << <grid(n), BLOCK >> > (a, b, out, n);
    check("mul");
}

template<typename T>
void launch_fill(T* out, T val, std::size_t n) {
    fill_kernel<T> << <grid(n), BLOCK >> > (out, val, n);
    check("fill");
}

template<typename T>
void launch_add_backward(const T* gout, T* ga, T* gb, std::size_t n) {
    add_backward_kernel<T> << <grid(n), BLOCK >> > (gout, ga, gb, n);
    check("add_backward");
}
template<typename T>
void launch_mul_backward(const T* gout, const T* a, const T* b, T* ga, T* gb, std::size_t n) {
    mul_backward_kernel<T> << <grid(n), BLOCK >> > (gout, a, b, ga, gb, n);
    check("mul_backward");
}

template<typename T>
void launch_accumulate(T* dst, const T* src, std::size_t n) {
    accumulate_kernel<T> << <grid(n), BLOCK >> > (dst, src, n);
    check("accumulate");
}

#define INSTANTIATE(T)                                                              \
    template void launch_add<T>(const T*, const T*, T*, std::size_t);               \
    template void launch_mul<T>(const T*, const T*, T*, std::size_t);               \
    template void launch_fill<T>(T*, T, std::size_t);                               \
    template void launch_add_backward<T>(const T*, T*, T*, std::size_t);            \
    template void launch_mul_backward<T>(const T*, const T*, const T*, T*, T*, std::size_t); \
    template void launch_accumulate<T>(T*, const T*, std::size_t);                  \

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
