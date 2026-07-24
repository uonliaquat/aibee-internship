#include "../../include/kernel/layernorm.h"
#include <math.h>

// LayerNorm formula: ((x - mean) / sqrt(var + eps)) * gamma + beta
void layernorm(
    const float* __restrict embed,         // embeddings (x)
    const float* __restrict weight,        // gamma
    const float* __restrict bias,          // beta or shift
    float* __restrict y,                   // output
    size_t seq_len,                        // number of tokens
    size_t embed_dim,                      // embedding dimension (d)
    float eps                              // div zero prevention 
) {
    for (size_t i = 0; i < seq_len; i++) {
        const float* __restrict token = embed + i * embed_dim;
        float* __restrict output_token = y + i * embed_dim;

        double inv = 1.0 / (double)embed_dim;

        double sum = 0.0;
        for (size_t j = 0; j < embed_dim; j++) {
            sum += (double)token[j];
        }
        double mean = sum * inv;

        double var = 0.0;
        for (size_t j = 0; j < embed_dim; j++) {
            double diff = (double)token[j] - mean;
            var += diff * diff;
        }
        var = var * inv;

        double inv_std = 1.0 / sqrt(var + (double)eps);

        for (size_t j = 0; j < embed_dim; j++) {
            float norm = (float)(((double)token[j] - mean) * inv_std);
            output_token[j] = fmaf(norm, weight[j], bias[j]);
        }
    }
}