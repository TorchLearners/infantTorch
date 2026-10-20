#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>

#include <infanttorch/core/cuda_tensor.cuh>
#include <infanttorch/detail/launch.hpp>

template<typename T>
struct Node {
    CudaTensor<T> value;
    std::vector<std::shared_ptr<Node<T>>> children;
    std::function<void()> backward_fn;
    const char* op = "leaf";

    Node() = default;

    void backward() {
        std::vector<Node<T>*> topo;
        std::unordered_set<Node<T>*> seen;
        build_topo(this, topo, seen);
        launch_fill<T>(value.grad, T(1), value.size);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            if ((*it)->backward_fn) (*it)->backward_fn();

    }
    void zero_grad() {
        std::vector<Node<T>*> topo;
        std::unordered_set<Node<T>*> seen;
        build_topo(this, topo, seen);
        for (Node<T>* n : topo) n->value.zero_grad();
    }

private:
    static void build_topo(Node<T>* n, std::vector<Node<T>*>& topo,
        std::unordered_set<Node<T>*>& seen) {
        if (seen.count(n)) return;
        seen.insert(n);
        for (auto& c : n->children) build_topo(c.get(), topo, seen);
        topo.push_back(n);
    }
};

template<typename T>
using NodePtr = std::shared_ptr<Node<T>>;

template<typename T>
NodePtr<T> leaf(const Tensor<T>& h, bool requires_grad = true) {
    auto n = std::make_shared<Node<T>>();
    n->value = CudaTensor<T>::from_host(h, requires_grad);
    return n;
}
