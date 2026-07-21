#include "linear_nobias.h"
#include "blis.h"
#include <stdlib.h>

/*
 * kernel_linear_nobias_cpu_f32_forward
 * -------------------------------------
 * Computes: output = input * weight^T   (no bias, matching LLaMA's linear layers)
 *
 * Inputs/outputs are float32 (matches the team's kernel interfaces), but
 * internally we cast up to double precision, multiply in double, then cast
 * back down to float32 only at the end. This greatly reduces accumulation
 * error for large in_features (K), since summing thousands of float32
 * products directly loses far more precision than summing in double.
 */
void kernel_linear_nobias_cpu_f32_forward(
    const float *input,
    const float *weight,
    float *output,
    size_t seq_len,
    size_t in_features,
    size_t out_features
) {
    size_t input_count  = seq_len * in_features;
    size_t weight_count = out_features * in_features;
    size_t output_count = seq_len * out_features;

    double *input_d  = (double *)malloc(input_count  * sizeof(double));
    double *weight_d = (double *)malloc(weight_count * sizeof(double));
    double *output_d = (double *)malloc(output_count * sizeof(double));

    if (input_d == NULL || weight_d == NULL || output_d == NULL) {
        free(input_d);
        free(weight_d);
        free(output_d);
        return;
    }

    for (size_t i = 0; i < input_count; i++) {
        input_d[i] = (double)input[i];
    }
    for (size_t i = 0; i < weight_count; i++) {
        weight_d[i] = (double)weight[i];
    }

    double alpha = 1.0;
    double beta  = 0.0;

    bli_dgemm(
        BLIS_NO_TRANSPOSE,
        BLIS_TRANSPOSE,
        (dim_t)seq_len,
        (dim_t)out_features,
        (dim_t)in_features,
        &alpha,
        input_d, (dim_t)in_features, 1,
        weight_d, (dim_t)in_features, 1,
        &beta,
        output_d, (dim_t)out_features, 1
    );

    for (size_t i = 0; i < output_count; i++) {
        output[i] = (float)output_d[i];
    }

    free(input_d);
    free(weight_d);
    free(output_d);
}
