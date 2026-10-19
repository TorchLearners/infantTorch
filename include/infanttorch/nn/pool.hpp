#pragma once

#include <infanttorch/autograd/node.hpp>
#include <infanttorch/nn/module.hpp>
#include <infanttorch/ops/pool.cuh>

//dont need to take in an optimizer because there are no parameters we are optimizing
template<typename T>
struct MaxPool : public Module<T> {
    int pool;
    MaxPool(int pool = 2) : pool(pool) {}
    NodePtr<T> forward(NodePtr<T> input) override { return maxpool(input, pool); }
};
