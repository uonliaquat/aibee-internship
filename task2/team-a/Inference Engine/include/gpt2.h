#ifndef GPT2_BLOCK_H
#define GPT2_BLOCK_H

#include <stddef.h>
#include "kernel/weights.h"

void gpt2_block(const float *x, float *out, const GPT2Weights *w, float *workspace, size_t cws, size_t emb_dim, size_t num_heads, float eps);

#endif