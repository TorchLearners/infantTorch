#pragma once

#include <infanttorch/autograd/node.hpp>

template <typename T>
struct optim{
    virtual void add(NodePtr<T> p) = 0;

    virtual void step() = 0;

    virtual void zero_grad() = 0;

    virtual void stepEpoch() = 0;

    virtual ~optim() = default;
};