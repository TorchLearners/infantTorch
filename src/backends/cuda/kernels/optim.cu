#include <infanttorch/detail/launch.hpp>
#include "optim.cuh"
#include "launch_utils.cuh"

template<typename T>
void launch_sgd_update(T* w, const T* g, T lr, std::size_t n) {
    sgd_update_kernel<T> << <grid(n), BLOCK >> > (w, g, lr, n);
    check("sgd_update");
}

template<typename T>
//U lr, U momentum, U decay, U eps, size_t step, size_t n
void launch_adam_update(T* w, T* m, T* v, const T* g, T lr, T momentum, T decay, T eps, T step, std::size_t n){
    adam_update_kernel<T> <<<grid(n), BLOCK>>>(w, m, v, g, lr, momentum, decay, eps, step, n);
    check("adam_update_kernel");
}


#define INSTANTIATE(T)                                                              \
    template void launch_sgd_update<T>(T*, const T*, T, std::size_t); \
    template void launch_adam_update<T>(T*, T*, T*, const T*, T, T, T, T, T, std::size_t); \

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
