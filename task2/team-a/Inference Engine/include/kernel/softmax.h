#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <stddef.h>
#include "blis.h"

void softmax(
    float *out,
    size_t context_win_size,
    size_t emb_dim,
    inc_t rs,
    inc_t cs
);

#endif