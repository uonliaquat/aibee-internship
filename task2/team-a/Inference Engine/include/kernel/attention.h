#ifndef ATTENTION_H
#define ATTENTION_H

#include <stddef.h>
#include "blis.h"

void attention(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    float *temp,
    size_t context_win_size,
    size_t emb_dim,
    size_t num_heads,
    inc_t rs,
    inc_t cs
);

#endif