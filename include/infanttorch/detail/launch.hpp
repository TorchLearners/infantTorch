#pragma once

#include <cstddef>

template<typename T> void launch_add(const T* a, const T* b, T* out, std::size_t n);
template<typename T> void launch_mul(const T* a, const T* b, T* out, std::size_t n);
template<typename T> void launch_relu(const T* x, T* out, std::size_t n);

template<typename T> void launch_fill(T* out, T val, std::size_t n);

template<typename T> void launch_add_backward(const T* gout, T* ga, T* gb, std::size_t n);
template<typename T> void launch_mul_backward(const T* gout, const T* a, const T* b, T* ga, T* gb, std::size_t n);
template<typename T> void launch_relu_backward(const T* gout, const T* x, T* gx, std::size_t n);

template<typename T> void launch_linear_forward(const T* A, const T* B, const T* bias, T* out,
    int M, int N, int K, bool RELU);

template<typename T> void launch_matmul(const T* A, const T* B, T* C, int M, int N, int K);
template<typename T> void launch_matmul_backward_A(const T* B, const T* C, const T* dC, T* dA, int M, int N, int K, bool RELU);
template<typename T> void launch_matmul_backward_B(const T* A, const T* C, const T* dC, T* dB, int M, int N, int K, bool RELU);
template<typename T> void launch_bias_add(const T* in, const T* b, T* out, int M, int N);
template<typename T> void launch_accumulate(T* dst, const T* src, std::size_t n);
template<typename T> void launch_bias_grad(const T* out, const T* gout, T* db, int M, int N, bool RELU);
template<typename T> void launch_softmax_ce_forward(const T* Z, const int* labels, T* probs, T* lossp, int M, int C);
template<typename T> void launch_softmax_ce_backward(const T* gscalar, const T* probs, const int* labels, T* dZ, int M, int C);
template<typename T> void launch_sgd_update(T* w, const T* g, T lr, std::size_t n);
template<typename T> void launch_adam_update(T* w, T* m, T* v, const T* g, T lr, T momentum, T decay, T eps, T step, std::size_t n);
template<typename T> void launch_softmax(const T* in, T* out, const std::size_t M, const std::size_t C, const std::size_t N);

template<typename T> void launch_conv2d_forward(const T* in, const T* filt, T* out,
    int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW);
template<typename T> void launch_conv_dW(const T* in, const T* dOut, T* dW,
    int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW);
template<typename T> void launch_conv_dIn(const T* dOut, const T* filt, T* dIn,
    int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW);
template<typename T> void launch_conv_forward(const T* in, const T* filt, 
    const T* bias, T* out, int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW, bool RELU);
template<typename T> void launch_conv_dW_RELU(const T* in, const T* dOut, const T* out, T* dW,
    int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW);
template<typename T> void launch_conv_dIn_RELU(const T* dOut, const T* out, const T* filt, T* dIn,
    int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW);
template<typename T> void launch_conv_bias_grad_RELU(const T* dOut, const T* out, T* dBias, int N, int C, int HW);

template<typename T> void launch_conv_bias(const T* in, const T* bias, T* out, int N, int C, int HW);
template<typename T> void launch_conv_bias_grad(const T* dOut, T* dBias, int N, int C, int HW);

template<typename T> void launch_maxpool(const T* mat, T* out, int* argmax_cache,
    int N, int C, int H, int W, int pool, int oH, int oW);
template<typename T> void launch_backward_maxpool(const T* dOut, T* dIn, const int* argmax_cache,
    int N, int C, int oH, int oW);

template<typename T> void launch_mean_reduce_kernel(const T* in, T* out, int M);

template<typename T> void launch_accumulate_loss_correct(T* in, const T* loss, const T* logits, int* labels, int M, int C);
