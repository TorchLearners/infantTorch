#include <infanttorch/detail/launch.hpp>
#include "norm.cuh"
#include "launch_utils.cuh"

template<typename T>
void launch_softmax(const T* in, T* out, std::size_t M, std::size_t C, std::size_t N) {
    const std::size_t ROWS = M * C;
    const std::size_t K = (N + 31) / 32;
    const unsigned g = grid(ROWS * 32);
    // K = how much each thread must look at
    if      (K <= 1)  softmax_good<T, 1> <<<g, BLOCK>>>(in, out, ROWS, N);
    else if (K <= 2)  softmax_good<T, 2> <<<g, BLOCK>>>(in, out, ROWS, N);
    else if (K <= 4)  softmax_good<T, 4> <<<g, BLOCK>>>(in, out, ROWS, N);
    else if (K <= 8)  softmax_good<T, 8> <<<g, BLOCK>>>(in, out, ROWS, N);
    else if (K <= 16) softmax_good<T, 16><<<g, BLOCK>>>(in, out, ROWS, N);
    else if (K <= 32) softmax_good<T, 32><<<g, BLOCK>>>(in, out, ROWS, N);
    //one thread per channel
    else softmax_bad<T> <<<(M*C), BLOCK>>>(in, out, M, C, N);
    check("softmax");
}

#define INSTANTIATE(T)                                                              \
    template void launch_softmax<T>(const T*, T*, std::size_t, std::size_t, std::size_t);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE