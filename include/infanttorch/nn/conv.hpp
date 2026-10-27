#pragma once

#include <vector>
#include <infanttorch/core/tensor.hpp>
#include <infanttorch/autograd/node.hpp>
#include <infanttorch/nn/module.hpp>
#include <infanttorch/optim/sgd.hpp>
#include <infanttorch/ops/conv.hpp>
#include <infanttorch/ops/activation.hpp>

template <typename T>
struct Conv : public Module<T> {
    NodePtr<T> filters;
    NodePtr<T> bias;
    bool RELU;
    Conv(SGD<T>& opt, long long Cin, long long Cout, long long kernelSize, bool RELU = true) : RELU(RELU) {
        Tensor<T> filter(std::vector<long long> {Cout, Cin, kernelSize, kernelSize}, Init::He, Cin* kernelSize* kernelSize);
        filters = leaf(filter);
        Tensor<T> b(std::vector<long long> {Cout}, Init::Zero);
        bias = leaf(b);
        opt.add(filters);
        opt.add(bias);
    }
    NodePtr<T> forward(NodePtr<T> input) override {
        //think abt the shared_pointers
        //input and filters stored in this object
        /* conv(a,b) -> return temp ptr to object
            then conv_bias(conv(a,b), bias)
            this returns a temp pointer to a new object with the temp ptr++ and object
            so temp ptr has a ref count of 2 at end of semi colon it gets -- 
            so that ptr to the other object remains cuz its in conv_bias childrne
        */
        NodePtr<T> y = conv_forward(input, filters, bias, RELU);
        return y;
    }
};
