#ifndef ROPE_H
#define ROPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Apply Rotary Positional Embeddings (RoPE)
 *
 * Q and K are modified in-place.
 *
 * Shapes:
 *
 * q   : (seq_len, head_dim)
 * k   : (seq_len, head_dim)
 * cos : (seq_len, head_dim)
 * sin : (seq_len, head_dim)
 *
 */

void kernel_rope_cpu_f32_forward(
    float *q,
    float *k,
    const float *cos,
    const float *sin,
    size_t seq_len,
    size_t head_dim
);

#ifdef __cplusplus
}
#endif

#endif