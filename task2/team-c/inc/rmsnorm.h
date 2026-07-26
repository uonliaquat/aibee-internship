#ifndef __RMS_NORM_H__
#define __RMS_NORM_H__
#include<stdio.h>
#include<stdint.h>

void kernel_rmsnorm_cpu_f32_forward(
    const double* __restrict embed,
    const double* __restrict weight,
    double* __restrict y,
    const size_t seq_len,
    const size_t embed_dim,
    const double eps
);

#endif