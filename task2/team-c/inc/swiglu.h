#ifndef SWIGLU_H
#define SWIGLU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * kernel_swiglu_cpu_f32_forward
 * -----------------------------
 * Computes the SwiGLU feed-forward block:
 *
 *   gate       = input * gate_weight^T
 *   up         = input * up_weight^T
 *   activated  = SILU(gate)
 *   multiplied = activated * up
 *   output     = multiplied * down_weight^T
 *
 * Shapes (row-major storage):
 *   input       : (batch, in_dim)
 *   gate_weight : (hidden_dim, in_dim)
 *   up_weight   : (hidden_dim, in_dim)
 *   down_weight : (out_dim, hidden_dim)
 *   output      : (batch, out_dim)
 *
 * Weight layout matches kernel_linear_nobias_cpu_f32_forward:
 * weights are stored as (out_features, in_features), and the linear
 * kernel computes input * weight^T.
 */
void kernel_swiglu_cpu_f32_forward(
    const float *input,
    const float *gate_weight,
    const float *up_weight,
    const float *down_weight,
    float *output,
    size_t batch,
    size_t in_dim,
    size_t hidden_dim,
    size_t out_dim
);

#ifdef __cplusplus
}
#endif

#endif /* SWIGLU_H */
