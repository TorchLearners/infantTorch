#pragma once

#include <infanttorch/autograd/node.hpp>

//Can do everything
template<typename T>
struct Module {
    virtual NodePtr<T> forward(NodePtr<T> x) = 0;
    virtual ~Module() = default;
};
