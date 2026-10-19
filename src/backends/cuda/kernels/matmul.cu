#include <infanttorch/detail/launch.hpp>
#include "matmul.cuh"
#include "launch_utils.cuh"

namespace {
    //2d
    dim3 grid2d(int cols, int rows) {
        return dim3((cols + 15) / 16, (rows + 15) / 16);
    }
    const dim3 BLOCK2D(16, 16);
}

template<typename T>
void launch_matmul(const T* A, const T* B, T* C, int M, int N, int K) {
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<T, TILE> << <grid, block >> > (A, B, C, M, N, K);
    check("matmul");
}
template<typename T>
void launch_matmul_backward_A(const T* dC, const T* B, T* dA, int M, int N, int K) {
    matmul_backward_A_kernel<T> << <grid2d(K, M), BLOCK2D >> > (dC, B, dA, M, N, K);
    check("matmul_backward_A");
}
template<typename T>
void launch_matmul_backward_B(const T* A, const T* dC, T* dB, int M, int N, int K) {
    matmul_backward_B_kernel<T> << <grid2d(N, K), BLOCK2D >> > (A, dC, dB, M, N, K);
    check("matmul_backward_B");
}
template<typename T>
void launch_bias_add(const T* in, const T* b, T* out, int M, int N) {
    bias_add_kernel<T> << <grid(static_cast<std::size_t>(M) * N), BLOCK >> > (in, b, out, M, N);
    check("bias_add");
}
template<typename T>
void launch_bias_grad(const T* gout, T* db, int M, int N) {
    bias_grad_kernel<T> << <grid(static_cast<std::size_t>(N)), BLOCK >> > (gout, db, M, N);
    check("bias_grad");
}

#define INSTANTIATE(T)                                                              \
    template void launch_matmul<T>(const T*, const T*, T*, int, int, int);          \
    template void launch_matmul_backward_A<T>(const T*, const T*, T*, int, int, int); \
    template void launch_matmul_backward_B<T>(const T*, const T*, T*, int, int, int); \
    template void launch_bias_add<T>(const T*, const T*, T*, int, int);             \
    template void launch_bias_grad<T>(const T*, T*, int, int);                      \

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
