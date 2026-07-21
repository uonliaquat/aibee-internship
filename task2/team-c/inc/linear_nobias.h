#ifndef LINEAR_NOBIAS_H
#define LINEAR_NOBIAS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * kernel_linear_nobias_cpu_f32_forward
 * -------------------------------------
 * Computes: output = input * weight^T   (no bias, matching LLaMA's linear layers)
 *
 * Shapes (row-major storage):
 *   input  : (seq_len, in_features)
 *   weight : (out_features, in_features)   <-- PyTorch/LLaMA weight layout
 *   output : (seq_len, out_features)
 *
 * Parameters:
 *   input        - pointer to input matrix data (seq_len x in_features)
 *   weight       - pointer to weight matrix data (out_features x in_features)
 *   output       - pointer to output matrix data (seq_len x out_features),
 *                  must be pre-allocated by the caller
 *   seq_len      - number of rows in input / output
 *   in_features  - number of columns in input, number of columns in weight
 *   out_features - number of rows in weight, number of columns in output
 */
void kernel_linear_nobias_cpu_f32_forward(
    const float *input,
    const float *weight,
    float *output,
    size_t seq_len,
    size_t in_features,
    size_t out_features
);

#ifdef __cplusplus
}
#endif

#endif /* LINEAR_NOBIAS_H */
