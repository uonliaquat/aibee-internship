#include "softmax_gelu.h"
#include <math.h>

void softmax(float *x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float *row = x + i * cols;

        float max = row[0];
        for (int j = 1; j < cols; j++) {
            if (row[j] > max) max = row[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }

        float inv_sum = 1.0f / sum;
        for (int j = 0; j < cols; j++) {
            row[j] *= inv_sum;
        }
    }
}

void gelu(float *x, size_t n) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        float cube = v * v * v;
        float inner = sqrt_2_over_pi * (v + coeff * cube);
        float tanh_val = tanhf(inner);
        x[i] = 0.5f * v * (1.0f + tanh_val);
    }
}