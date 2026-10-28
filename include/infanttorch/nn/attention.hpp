#pragma once


#include <infanttorch/core/tensor.hpp>
#include <infanttorch/autograd/node.hpp>
#include <infanttorch/nn/module.hpp>
#include <infanttorch/optim/optim.hpp>


//# Shapes: (batch_size, num_heads, sequence_length, head_dimension)

struct MultiheadedAttention : public Module<T>{
    Tensor<T>  
};