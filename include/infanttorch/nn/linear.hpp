#pragma once

#include <vector>
#include <infanttorch/core/tensor.hpp>
#include <infanttorch/autograd/node.hpp>
#include <infanttorch/nn/module.hpp>
#include <infanttorch/optim/optim.hpp>
#include <infanttorch/ops/matmul.hpp>

template<typename T>
struct Linear : public Module<T> {
    NodePtr<T> W, b;
    bool use_relu;

    Linear(optim<T>& opt, int in, int out, bool relu = true) : use_relu(relu) {
        Tensor<T> hw(std::vector<long long>{ (long long)in, (long long)out }, Init::He, in);
        Tensor<T> hb(std::vector<long long>{ (long long)out }, Init::Zero);
        W = leaf(hw);
        b = leaf(hb);
        opt.add(W);
        opt.add(b);
    }

    NodePtr<T> forward(NodePtr<T> x) override {
        //so this is bad i want to like not have so many kernel launches

        NodePtr<T> y = linear_forward(x, W, b, use_relu);
        return y;
    }
};
