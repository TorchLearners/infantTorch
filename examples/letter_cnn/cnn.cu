#include "cnn.cuh"

int main() {
    using T = float;
    const int NUM_CLASSES = 26;
    const long long H = 28, W = 28;
    CNN<T>::Dataset data;
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> label(0, NUM_CLASSES - 1);
    for (int i = 0; i < 64; ++i) {
        std::vector<Tensor<T>> channels{ Tensor<T>(std::vector<long long>{ H, W }, Init::Random) };
        data.push_back({ std::move(channels), label(rng) });
    }

    CNN<T> net(T(0.01), T(1e-4), 2);
    net.add(std::make_unique<Conv<T>>(net.opt, 1, 8, 3))
       .add(std::make_unique<MaxPool<T>>(2))
       .add(std::make_unique<Flatten<T>>())
       .add(std::make_unique<Linear<T>>(net.opt, 8 * 13 * 13, NUM_CLASSES, false));

    net.train(data, 2, "smoke_", nullptr, 16);
    auto [loss, acc] = net.evaluate(data, 16);
    std::cout << "eval loss " << loss << " | acc " << (acc * 100) << "%\n";
    std::cout << "predicted " << (char)('A' + net.predict(data[0].first))
              << " for label " << (char)('A' + data[0].second) << "\n";
    return 0;
}
//compile with nvcc -std=c++20 -O2 -arch=native -I include examples/letter_cnn/cnn.cu src/backends/cuda/kernels/*.cu -o letter_cnn
