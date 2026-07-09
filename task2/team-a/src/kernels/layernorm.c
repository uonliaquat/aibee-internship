#include "../../inc/kernels/layernorm.h"

void layernorm(
    const float* __restrict embed,
    const float* __restrict weight,
    const float* __restrict bias,
    float* __restrict y,
    size_t seq_len,
    size_t embed_dim,
    float eps
) {

}