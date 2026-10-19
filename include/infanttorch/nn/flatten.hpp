#pragma once

#include <infanttorch/autograd/node.hpp>
#include <infanttorch/nn/module.hpp>
#include <infanttorch/ops/shape.cuh>

template<typename T>
struct Flatten : public Module<T> {
    NodePtr<T> forward(NodePtr<T> x) override { return flatten(x); }
};
