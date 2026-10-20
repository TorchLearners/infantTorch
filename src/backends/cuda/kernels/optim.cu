#include <infanttorch/detail/launch.hpp>
#include "optim.cuh"
#include "launch_utils.cuh"

template<typename T>
void launch_sgd_update(T* w, const T* g, T lr, std::size_t n) {
    sgd_update_kernel<T> << <grid(n), BLOCK >> > (w, g, lr, n);
    check("sgd_update");
}



#define INSTANTIATE(T)                                                              \
    template void launch_sgd_update<T>(T*, const T*, T, std::size_t); \

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
