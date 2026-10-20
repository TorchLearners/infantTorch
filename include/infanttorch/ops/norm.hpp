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
}
