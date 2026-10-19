#pragma once

#include <infanttorch/autograd/node.hpp>
#include <infanttorch/detail/launch.hpp>

template<typename T>
NodePtr<T> relu(NodePtr<T> x) {
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(x->value.shape, true);
    out->op = "relu";
    out->children = { x };

    const std::size_t n = out->value.size;
    launch_relu<T>(x->value.data, out->value.data, n);

    Node<T>* o = out.get(); Node<T>* px = x.get();
    out->backward_fn = [o, px, n]() {
        launch_relu_backward<T>(o->value.grad, px->value.data, px->value.grad, n);
        };
    return out;
}
