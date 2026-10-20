#pragma once

#include <infanttorch/autograd/node.hpp>
#include <infanttorch/detail/launch.hpp>

template<typename T>
NodePtr<T> add(NodePtr<T> a, NodePtr<T> b) {
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(a->value.shape, a->value.requires_grad);
    out->op = "add";
    if(out->value.requires_grad) out->children = { a, b };

    const std::size_t n = out->value.size;
    launch_add<T>(a->value.data, b->value.data, out->value.data, n);

    Node<T>* o = out.get(); Node<T>* pa = a.get(); Node<T>* pb = b.get();
    out->backward_fn = [o, pa, pb, n]() {
        launch_add_backward<T>(o->value.grad, pa->value.grad, pb->value.grad, n);
        };
    return out;
}

template<typename T>
NodePtr<T> mul(NodePtr<T> a, NodePtr<T> b) {
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(a->value.shape, a->value.requires_grad);
    out->op = "mul";
    if(out->value.requires_grad) out->children = { a, b };

    const std::size_t n = out->value.size;
    launch_mul<T>(a->value.data, b->value.data, out->value.data, n);

    if(out->value.requires_grad){
        Node<T>* o = out.get(); Node<T>* pa = a.get(); Node<T>* pb = b.get();
        out->backward_fn = [o, pa, pb, n]() {
            launch_mul_backward<T>(o->value.grad, pa->value.data, pb->value.data,
                pa->value.grad, pb->value.grad, n);
            };
    }
    return out;
}
