#pragma once

#include <vector>
#include <functional>
#include <cstddef>
#include <infanttorch/autograd/node.hpp>
#include <infanttorch/detail/launch.hpp>
#include <infanttorch/optim/schedulers.hpp>
#include <infanttorch/optim/optim.hpp>

template<typename T>
struct Adam : public optim<T>{
    std::vector<NodePtr<T>> params;
    std::vector<CudaTensor<T>> m, v;
    T lr;
    T lrMax, lrMin;
    size_t epoch = 0;
    size_t totalEpochs = 20;
    T momentum, decay, eps;
    T step_p = T(0);

    
    std::function<T(std::size_t, std::size_t, T, T)> lr_scheduler;

    explicit Adam(T lrMax = T(0.01), T lrMin = T(1e-6), size_t totalEpochs = 20,
    std::function<T(std::size_t, std::size_t, T, T)> sched = linear_LR<T>,
    T momentum = T(0.9), T decay = T(0.999), T eps = T(1e-8))
    : lr(lrMax), lrMax(lrMax), lrMin(lrMin), totalEpochs(totalEpochs),
      momentum(momentum), decay(decay), eps(eps), lr_scheduler(std::move(sched)) {

      }

    void add(NodePtr<T> p) override{ 
        m.emplace_back(p->value.shape);
        m.back().zero();
        v.emplace_back(p->value.shape);
        v.back().zero();
        params.push_back(std::move(p));
    }


    void step() override{
        step_p++;
        /*void launch_adam_update(T* w, T* m, T* v, const T* g, T lr, T momentum, T decay, T eps, T step, std::size_t n){
    adam_update_kernel<T> <<<grid(n), BLOCK>>>(w, m, g, lr, momentum, decay, eps, step, n);
    check("adam_update_kernel");
        } */
        for(size_t i{}; i < params.size(); i++){
            launch_adam_update<T>(params[i]->value.data, m[i].data, v[i].data, params[i] -> value.grad, lr, momentum, decay, eps, step_p, params[i]->value.size);
        }
    }

    void zero_grad() override { 
        for (auto& p : params) p->value.zero_grad(); 
    }

    void stepEpoch() override{
        epoch++;
        lr = lr_scheduler(epoch, totalEpochs, lrMax, lrMin);
    }


};