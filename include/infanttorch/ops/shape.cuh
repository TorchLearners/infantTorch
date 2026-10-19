#pragma once

#include <infanttorch/core/cuda_tensor.cuh>
#include <infanttorch/autograd/node.hpp>
#include <infanttorch/detail/launch.hpp>

template<typename T>
NodePtr<T> flatten(NodePtr<T> input) {
    const long long N = input->value.shape[0];
    const std::size_t tot = input->value.size;
    const long long feat = (long long)(tot / (std::size_t)N);

    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ N, feat }, true);
    out->op = "flatten";
    out->children = { input };

    CUDA_CHECK(cudaMemcpy(out->value.data, input->value.data,
        tot * sizeof(T), cudaMemcpyDeviceToDevice));

    Node<T>* o = out.get(); Node<T>* px = input.get();
    out->backward_fn = [o, px, tot]() {
        launch_accumulate<T>(px->value.grad, o->value.grad, tot);
        };
    return out;
}
