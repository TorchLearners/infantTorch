#pragma once

#include <infanttorch/autograd/node.hpp>
#include <infanttorch/detail/launch.hpp>

//A is (M, K) and B is (K,N)
template<typename T>
NodePtr<T> matmul(NodePtr<T> a, NodePtr<T> b) {
    const int M = static_cast<int>(a->value.shape[0]);
    const int K = static_cast<int>(a->value.shape[1]);
    const int N = static_cast<int>(b->value.shape[1]);
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ M, N }, a->value.requires_grad);
    out->op = "matmul";
    if(out->value.requires_grad) out->children = { a, b };

    launch_matmul<T>(a->value.data, b->value.data, out->value.data, M, N, K);

    if(out->value.requires_grad){
        Node<T>* o = out.get(); Node<T>* pa = a.get(); Node<T>* pb = b.get();
        out->backward_fn = [o, pa, pb, M, N, K]() {
            launch_matmul_backward_A<T>(o->value.grad, pb->value.data, pa->value.grad, M, N, K);
            launch_matmul_backward_B<T>(pa->value.data, o->value.grad, pb->value.grad, M, N, K);
            };
    }
    return out;
}

template<typename T>
NodePtr<T> bias_add(NodePtr<T> x, NodePtr<T> b) {
    const int M = static_cast<int>(x->value.shape[0]);
    const int N = static_cast<int>(x->value.shape[1]);

    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ M, N }, x->value.requires_grad);
    out->op = "bias_add";
    if(out -> value.requires_grad) out->children = { x, b };

    launch_bias_add<T>(x->value.data, b->value.data, out->value.data, M, N);

    if(out-> value.requires_grad){
        Node<T>* o = out.get(); Node<T>* px = x.get(); Node<T>* pb = b.get();
        out->backward_fn = [o, px, pb, M, N]() {
            launch_accumulate<T>(px->value.grad, o->value.grad, static_cast<std::size_t>(M) * N);
            launch_bias_grad<T>(o->value.grad, pb->value.grad, M, N);
        };
    }
    
    return out;
}
