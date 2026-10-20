#pragma once

#include <memory>
#include <vector>
#include <infanttorch/core/cuda_tensor.cuh>
#include <infanttorch/autograd/node.hpp>
#include <infanttorch/detail/launch.hpp>

template<typename T>
NodePtr<T> softmax_cross_entropy(NodePtr<T> logits, const std::vector<int>& labels) {
    const int M = static_cast<int>(logits->value.shape[0]);
    const int C = static_cast<int>(logits->value.shape[1]);


    int* dlabels_raw = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dlabels_raw, sizeof(int) * M, 0));
    CUDA_CHECK(cudaMemcpy(dlabels_raw, labels.data(), sizeof(int) * M, cudaMemcpyHostToDevice));
    std::shared_ptr<int> dlabels(dlabels_raw, [](int* p) { cudaFreeAsync(p, 0); });

    auto probs = std::make_shared<CudaTensor<T>>(std::vector<long long>{ M, C }, false);
    T* dlossp_raw = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dlossp_raw, sizeof(T) * M, 0));
    std::shared_ptr<T> dlossp(dlossp_raw, [](T* p) { cudaFreeAsync(p, 0); });

    //convert raw logits into probabiltiies
    launch_softmax_ce_forward<T>(logits->value.data, dlabels.get(), probs->data, dlossp.get(), M, C);
    //dloss IS NOT DERIVATIVE, IT IS DEVICE LOSS, ITS CONFUSING BUT IM LAZY AND I UNDERSTAND IT
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ 1 }, true);
    out->op = "softmax_ce";
    out->children = { logits };
    launch_mean_reduce_kernel<T>(dlossp.get(), out->value.data, M);

    Node<T>* o = out.get(); Node<T>* pl = logits.get();

    out->backward_fn = [o, pl, probs, dlabels, M, C]() {
        launch_softmax_ce_backward<T>(o->value.grad, probs->data, dlabels.get(), pl->value.grad, M, C);
        };
    return out;
}
