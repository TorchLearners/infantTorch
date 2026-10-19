#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <infanttorch/core/tensor.hpp>
#include <infanttorch/core/cuda_tensor.cuh>
#include <infanttorch/autograd/node.hpp>

//save and load like other one
template<typename T>
void save_params(const std::vector<NodePtr<T>>& params, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_params: cannot open " + path);
    for (const auto& p : params) {
        const auto dt = p->value.to_host();
        const std::vector<T>& d = dt.getData();
        f.write(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(T));
    }
}

template<typename T>
void load_params(std::vector<NodePtr<T>>& params, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_params: cannot open " + path);
    for (auto& p : params) {
        std::vector<T> buf(p->value.size);
        f.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(T));
        Tensor<T> h(p->value.shape, buf);
        p->value = CudaTensor<T>::from_host(h, true); //if writing we are always going to require a grad so always set to true
    }
}
