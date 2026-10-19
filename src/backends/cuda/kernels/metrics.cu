#include <infanttorch/detail/launch.hpp>
#include "metrics.cuh"
#include "launch_utils.cuh"

template<typename T>
void launch_accumulate_loss_correct(T* in, const T* loss, const T* logits, int* labels, int M, int C) {
    accumulate_loss_correct<T> << <grid((std::size_t)M), BLOCK >> > (in, loss, logits, labels, M, C);
    check("accumulate_loss_correct");
}

#define INSTANTIATE(T)                                                              \
    template void launch_accumulate_loss_correct<T> (T*, const T*, const T*, int*, int, int); \

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
