#include "rope.h"

static void rotate_half(
    const float *x,
    float *out,
    size_t dim
)
{
    size_t half = dim / 2;

    for(size_t i = 0; i < half; i++)
    {
        out[i] = -x[half + i];
    }

    for(size_t i = 0; i < half; i++)
    {
        out[half + i] = x[i];
    }
}


void kernel_rope_cpu_f32_forward(
    float *q,
    float *k,
    const float *cos,
    const float *sin,
    size_t seq_len,
    size_t head_dim
)
{
    size_t row;

    float rotated_q[head_dim];
    float rotated_k[head_dim];

    for(row = 0; row < seq_len; row++)
    {
        float *q_row = q + row * head_dim;
        float *k_row = k + row * head_dim;

        const float *cos_row = cos + row * head_dim;
        const float *sin_row = sin + row * head_dim;

        rotate_half(q_row, rotated_q, head_dim);
        rotate_half(k_row, rotated_k, head_dim);

        for(size_t col = 0; col < head_dim; col++)
        {
            q_row[col] =
                q_row[col] * cos_row[col]
                +
                rotated_q[col] * sin_row[col];

            k_row[col] =
                k_row[col] * cos_row[col]
                +
                rotated_k[col] * sin_row[col];
        }
    }
}