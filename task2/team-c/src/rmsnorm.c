#include"../inc/rmsnorm.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void kernel_rmsnorm_cpu_f32_forward(
    const double* __restrict embed,         // embeddings
    const double* __restrict weight,        // gamma
    double* __restrict y,                   // output
    const size_t seq_len,                        // number of tokens
    const size_t embed_dim,                      // embedding dimension
    const double eps                              // div zero prevention 
) {

    const double inv_dim = 1.0 / (double)embed_dim;

    for (size_t i = 0; i < seq_len; i++) {
        const double* __restrict token = embed + i * embed_dim;
        double* __restrict output_token = y + i * embed_dim;

        double sum_sq = 0.0;
        for (size_t j = 0; j < embed_dim; j++) {
            double val = token[j];
            sum_sq = fma(val, val, sum_sq); // fused multiplication additive (a*b + c)
        }

        // Compute inverse RMS: 1 / sqrt(mean(x²) + eps)
        double inv_rms = 1.0 / sqrt(fma(sum_sq, inv_dim, eps));

        for (size_t j = 0; j < embed_dim; j++) {
            output_token[j] = token[j] * inv_rms * weight[j];
        }
    }
}