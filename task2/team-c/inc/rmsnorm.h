#ifndef __RMS_NORM_H__
#define __RMS_NORM_H__
#include<stdio.h>
#include<stdint.h>

void kernel_rmsnorm_cpu_f32_forward(
    const float* __restrict embed,
    const float* __restrict weight,
    float* __restrict y,
    const size_t seq_len,
    const size_t embed_dim,
    const float eps
);

#endif