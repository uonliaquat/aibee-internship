#include "../inc/swiglu.h"
#include "../inc/linear_nobias.h"
#include "../inc/silu.h"
#include "../inc/mul.h"

#include <stdint.h>
#include <stdlib.h>

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
) {
    size_t hidden_count = batch * hidden_dim;

    if (batch == 0 || hidden_dim == 0){
        return;
    }

    if (hidden_count > UINT32_MAX) {
        return;
    }

    float *gate = (float *)malloc(hidden_count * sizeof(float));
    float *up = (float *)malloc(hidden_count * sizeof(float));
    float *activated = (float *)malloc(hidden_count * sizeof(float));
    float *multiplied = (float *)malloc(hidden_count * sizeof(float));

    if (gate == NULL || up == NULL || activated == NULL || multiplied == NULL) {
        free(gate);
        free(up);
        free(activated);
        free(multiplied);
        return;
    }

    kernel_linear_nobias_cpu_f32_forward(
        input,
        gate_weight,
        gate,
        batch,
        in_dim,
        hidden_dim
    );

    kernel_linear_nobias_cpu_f32_forward(
        input,
        up_weight,
        up,
        batch,
        in_dim,
        hidden_dim
    );

    SILU(gate, activated, (uint32_t)hidden_count);
    MUL(activated, up, multiplied, (uint32_t)hidden_count);

    kernel_linear_nobias_cpu_f32_forward(
        multiplied,
        down_weight,
        output,
        batch,
        hidden_dim,
        out_dim
    );

    free(gate);
    free(up);
    free(activated);
    free(multiplied);
}
