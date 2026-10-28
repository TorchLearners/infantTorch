#pragma once

#include <memory>
#include <vector>
#include <infanttorch/core/cuda_tensor.cuh>
#include <infanttorch/autograd/node.hpp>
#include <infanttorch/detail/launch.hpp>

template<typename T>
NodePtr<T> softmax(NodePtr<T> logits) {
    const int M = static_cast<int>(logits->value.shape[0]);
    const int C = static_cast<int>(logits->value.shape[1]);
    //softmax doesn't need a backwards function
    const int N = static_cast<int>(logits -> value.shape[2]);
    
    auto out = std::make_shared<Node<T>>();

    //requires grad here should always be false but i digress
    out->value = CudaTensor<T>(std::vector<long long>{M, C, N}, logits->value.requires_grad);
    out->op = "softmax";

    launch_softmax<T> (logits->value.data, out->value.data, M, C, N);

    //no requires grad or function
    return out;

}
