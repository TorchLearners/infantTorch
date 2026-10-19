#pragma once

#include <memory>
#include <infanttorch/core/cuda_tensor.cuh>
#include <infanttorch/autograd/node.hpp>
#include <infanttorch/detail/launch.hpp>

template <typename T>
NodePtr<T> maxpool(NodePtr<T> input, int pool) {
    NodePtr<T> out = std::make_shared<Node<T>>();
    const int N = static_cast<int>(input->value.shape[0]);
    const int C = static_cast<int>(input->value.shape[1]);
    const int H = static_cast<int>(input->value.shape[2]);
    const int W = static_cast<int>(input->value.shape[3]);
    const int oH = H / pool, oW = W / pool;

    out->value = CudaTensor<T>(std::vector<long long>{ N, C, oH, oW }, true);
    out->op = "maxpool";
    out->children = { input };

    //kept alive for backwards
    int* argmax_raw = nullptr;
    CUDA_CHECK(cudaMalloc(&argmax_raw, sizeof(int) * (size_t)N * C * oH * oW));
    std::shared_ptr<int> argmax_cache(argmax_raw, [](int* p) { cudaFree(p); });

    launch_maxpool<T>(input->value.data, out->value.data, argmax_cache.get(), N, C, H, W, pool, oH, oW);

    Node<T>* o = out.get(), * in = input.get();
    out->backward_fn = [o, in, N, C, argmax_cache, oH, oW]() {
        launch_backward_maxpool<T>(o->value.grad, in->value.grad, argmax_cache.get(), N, C, oH, oW);
        };
    return out;
}
