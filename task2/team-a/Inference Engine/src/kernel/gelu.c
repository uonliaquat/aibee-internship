#include "../../include/kernel/gelu.h"
#include <math.h>


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