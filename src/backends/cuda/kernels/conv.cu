#include <infanttorch/detail/launch.hpp>
#include "conv.cuh"
#include "launch_utils.cuh"

template<typename T>
void launch_conv2d_forward(const T* in, const T* filt, T* out,
    int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW) {
    conv2d_forward_kernel<T> << <grid((std::size_t)N * Cout * oH * oW), BLOCK >> > (
        in, filt, out, N, Cin, H, W, Cout, K1, K2, oH, oW);
    check("conv2d_forward");
}

template<typename T>
void launch_conv_dW(const T* in, const T* dOut, T* dW,
    int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW) {
    const int R = N * oH * oW;
    const int chunk = BLOCK * 16;
    const int P = Cout * Cin * K1 * K2;
    dim3 grid2(P, (R+chunk-1)/chunk);

    conv2d_dW_kernel<T, false> <<<grid2, BLOCK>>>(in, dOut, nullptr, dW, N, Cin, H, W, Cout, K1, K2, oH, oW, chunk);
    check("conv_dW");
}

template<typename T>
void launch_conv_dW_RELU(const T* in, const T* dOut, const T* out, T* dW,
    int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW){
    const int R = N * oH * oW;
    const int chunk = BLOCK * 16;
    const int P = Cout * Cin * K1 * K2;
    dim3 grid2(P, (R+chunk-1)/chunk);

    conv2d_dW_kernel<T, true><<<grid2, BLOCK>>>(in, dOut, out, dW, N, Cin, H, W, Cout, K1, K2, oH, oW, chunk);
    check("conv_dW_relu");
}

template<typename T>
void launch_conv_dIn(const T* dOut, const T* filt, T* dIn,
    int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW) {
    conv2d_dIn_kernel<T, false> << <grid((std::size_t)N * Cin * H * W), BLOCK >> > (
        dOut, nullptr, filt, dIn, N, Cin, H, W, Cout, K1, K2, oH, oW);
    check("conv_dIn");
}

template<typename T>
void launch_conv_dIn_RELU(const T* dOut, const T* out, const T* filt, T* dIn,
    int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW){
conv2d_dIn_kernel<T, true> << <grid((std::size_t)N * Cin * H * W), BLOCK >> > (
        dOut, out, filt, dIn, N, Cin, H, W, Cout, K1, K2, oH, oW);
    check("conv_dIn_relu");
}


template <typename T>
void launch_conv_bias(const T* in, const T* bias, T* out, int N, int C, int HW) {
    conv_bias_forward_kernel<T> << <grid((std::size_t)N * C * HW), BLOCK >> > (in, bias, out, N, C, HW);
    check("conv_bias");
}




template <typename T>
void launch_conv_bias_grad(const T* dOut, T* dBias, int N, int C, int HW) {
    const int R = N * HW;
    const int chunk = BLOCK * 16;                        // 256 threads x 16 terms each
    //1 row per channel
    //we have
    dim3 grid2(C, (R + chunk - 1)/chunk);
    conv_bias_grad_kernel<T, false><<<grid2, BLOCK>>>(dOut, nullptr, dBias, N, C, HW, chunk);
    check("conv_bias_grad");
}

template<typename T>
void launch_conv_bias_grad_RELU(const T* dOut, const T* out, T* dBias, int N, int C, int HW){
    const int R = N * HW; 
    const int chunk = BLOCK * 16;

    dim3 grid2(C, (R + chunk - 1)/chunk);
    conv_bias_grad_kernel<T, true><<<grid2, BLOCK>>>(dOut, out, dBias, N, C, HW, chunk);
    check("conv_bias_grad_relu");
}




template<typename T>
void launch_conv_forward(const T* in, const T* filt, 
    const T* bias, T* out, int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW, bool RELU){
            if (RELU){
                conv_forward<T, true> <<<grid((std::size_t)N * Cout * oH * oW), BLOCK>>>(in, filt, bias, out, N, Cin, H, W, Cout, K1, K2, oH, oW);
            }
            else{
                conv_forward<T, false><<<grid((std::size_t)N * Cout * oH * oW), BLOCK>>>(in, filt, bias, out, N, Cin, H, W, Cout, K1, K2, oH, oW);
            }
        check("conv forward");
}

#define INSTANTIATE(T)                                                              \
    template void launch_conv2d_forward<T>(const T*, const T*, T*, int,int,int,int,int,int,int,int,int); \
    template void launch_conv_dW<T>(const T*, const T*, T*, int,int,int,int,int,int,int,int,int); \
    template void launch_conv_dW_RELU<T>(const T*, const T*, const T*, T*, int,int,int,int,int,int,int,int,int); \
    template void launch_conv_dIn<T>(const T*, const T*, T*, int,int,int,int,int,int,int,int,int); \
    template void launch_conv_dIn_RELU<T>(const T*, const T*, const T*, T*, int,int,int,int,int,int,int,int,int); \
    template void launch_conv_bias<T>(const T*, const T*, T*, int,int,int);               \
    template void launch_conv_bias_grad<T>(const T*, T*, int,int,int); \
    template void launch_conv_bias_grad_RELU<T>(const T*, const T*, T*, int,int,int); \
    template void launch_conv_forward<T>(const T*, const T*, const T*, T*, int, int, int, int, int, int, int, int, int, bool); \

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE


/*
Nsight magic:

& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\target\windows-desktop-win7-x64\ncu.exe" --set full -k "regex:conv2d_dW|conv_bias_grad|conv2d_forward|conv2d_dIn|matmul" --launch-skip 100 --launch-count 18 -o bench\prof\cudagrad_kernels_splitk -f bench\build\Release\cudagrad_train.exe 1 C:/Users/Ynoj/Desktop/Portfolio-Dev-Projects-/WordHuntAI/data 12

*/
