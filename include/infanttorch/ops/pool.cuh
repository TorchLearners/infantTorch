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

    out->value = CudaTensor<T>(std::vector<long long>{ N, C, oH, oW }, input->value.requires_grad);
    out->op = "maxpool";
    if(out->value.requires_grad) out->children = { input };
    std::shared_ptr<int> argmax_cache;

    if(out -> value.requires_grad){
        int* argmax_raw = nullptr;
        CUDA_CHECK(cudaMallocAsync(&argmax_raw, sizeof(int) * (size_t)N * C * oH * oW, 0));
        argmax_cache = std::shared_ptr<int>(argmax_raw, [](int* p) { cudaFreeAsync(p, 0); });
    }
    
    launch_maxpool<T>(input->value.data, out->value.data, argmax_cache.get(), N, C, H, W, pool, oH, oW);

    if(out -> value.requires_grad){
        Node<T>* o = out.get(), * in = input.get();
        out->backward_fn = [o, in, N, C, argmax_cache, oH, oW]() {
            launch_backward_maxpool<T>(o->value.grad, in->value.grad, argmax_cache.get(), N, C, oH, oW);
            };
    }
    return out;
}
