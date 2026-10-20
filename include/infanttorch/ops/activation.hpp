#pragma once

#include <infanttorch/autograd/node.hpp>
#include <infanttorch/detail/launch.hpp>

template<typename T>
NodePtr<T> relu(NodePtr<T> x) {
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(x->value.shape, x->value.requires_grad);
    out->op = "relu";
    if(out->value.requires_grad) out->children = { x };

    const std::size_t n = out->value.size;
    launch_relu<T>(x->value.data, out->value.data, n);
    
    //eval mode -> requires grad is false
    //train mode -> requires grad is true
    if(out->value.requires_grad){
        Node<T>* o = out.get(); Node<T>* px = x.get();
        out->backward_fn = [o, px, n]() {
            launch_relu_backward<T>(o->value.grad, px->value.data, px->value.grad, n);
            };
    }
    return out;
}
