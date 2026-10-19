#pragma once

#include <vector>
#include <functional>
#include <cstddef>
#include <infanttorch/autograd/node.hpp>
#include <infanttorch/detail/launch.hpp>
#include <infanttorch/optim/schedulers.hpp>

template<typename T>
struct SGD {
//params are all the weights/biases we must walk back on and apply SGD on
    std::vector<NodePtr<T>> params;
    T lr;
    T lrMax, lrMin;
    size_t epoch = 0, totalEpochs = 20;

    std::function<T(std::size_t, std::size_t, T, T)> lr_scheduler;

    explicit SGD(T lrMax = T(0.01), T lrMin = T(1e-6), size_t totalEpochs = 20,
        std::function<T(std::size_t, std::size_t, T, T)> sched = linear_LR<T>)
        : lr(lrMax), lrMax(lrMax), lrMin(lrMin), totalEpochs(totalEpochs),
        lr_scheduler(std::move(sched)) {
    }

    void add(NodePtr<T> p) { params.push_back(std::move(p)); }

    void zero_grad() { for (auto& p : params) p->value.zero_grad(); }

    void step() {
        for (auto& p : params)
            launch_sgd_update<T>(p->value.data, p->value.grad, lr, p->value.size);
    }

    void stepEpoch() {
        epoch++;
        lr = lr_scheduler(epoch, totalEpochs, lrMax, lrMin);
    }
};
