#pragma once

#include <infanttorch/autograd/node.hpp>
#include <infanttorch/detail/launch.hpp>

template<typename T>
NodePtr<T> conv(NodePtr<T> input, NodePtr<T> filters) {
    const int N = (int)input->value.shape[0];
    const int Cin = (int)input->value.shape[1];
    const int H = (int)input->value.shape[2];
    const int W = (int)input->value.shape[3];
    const int Cout = (int)filters->value.shape[0];
    const int K1 = (int)filters->value.shape[2];
    const int K2 = (int)filters->value.shape[3];
    const int oH = H - K1 + 1, oW = W - K2 + 1;

    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ N, Cout, oH, oW }, true);
    out->op = "conv";
    out->children = { input, filters };

    launch_conv2d_forward<T>(input->value.data, filters->value.data, out->value.data,
        N, Cin, H, W, Cout, K1, K2, oH, oW);

    Node<T>* o = out.get(); Node<T>* in = input.get(); Node<T>* fi = filters.get();
    out->backward_fn = [o, in, fi, N, Cin, H, W, Cout, K1, K2, oH, oW]() {
        launch_conv_dW<T>(in->value.data, o->value.grad, fi->value.grad,
            N, Cin, H, W, Cout, K1, K2, oH, oW);
        launch_conv_dIn<T>(o->value.grad, fi->value.data, in->value.grad,
            N, Cin, H, W, Cout, K1, K2, oH, oW);
        };
    return out;
}

template<typename T>
NodePtr<T> conv_bias(NodePtr<T> x, NodePtr<T> bias) {
    const int N = static_cast<int>(x->value.shape[0]);
    const int C = static_cast<int>(x->value.shape[1]);
    const int HW = static_cast<int>(x->value.shape[2] * x->value.shape[3]);

    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(x->value.shape, true);
    out->op = "conv_bias";
    out->children = { x, bias };

    launch_conv_bias<T>(x->value.data, bias->value.data, out->value.data, N, C, HW);

    Node<T>* o = out.get(); Node<T>* px = x.get(); Node<T>* pb = bias.get();
    out->backward_fn = [o, px, pb, N, C, HW]() {
        launch_accumulate<T>(px->value.grad, o->value.grad, static_cast<std::size_t>(N) * C * HW); // bias passthrough
        launch_conv_bias_grad<T>(o->value.grad, pb->value.grad, N, C, HW);
        };
    return out;
}
