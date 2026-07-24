#include "../include/gpt2.h"

#include "../include/kernel/layernorm.h"
#include "../include/kernel/linear.h"
#include "../include/kernel/attention.h"
#include "../include/kernel/add.h"
#include "../include/kernel/gelu.h"

#include "../include/kernel/weights.h"

void gpt2_block(const float *x, float *out, const GPT2Weights *w, float *workspace, size_t cws, size_t emb_dim, size_t num_heads, float eps)
{
    bli_thread_set_num_threads(4);
    const size_t ff_dim = emb_dim * 4;

    float *ln1_out = workspace;
    workspace += cws * emb_dim;

    float *Q = workspace;
    workspace += cws * emb_dim;

    float *K = workspace;
    workspace += cws * emb_dim;

    float *V = workspace;
    workspace += cws * emb_dim;

    float *attn_temp = workspace;
    workspace += cws * cws;

    float *attn_out = workspace;
    workspace += cws * emb_dim;

    float *proj_out = workspace;
    workspace += cws * emb_dim;

    float *residual1 = workspace;
    workspace += cws * emb_dim;

    float *ln2_out = workspace;
    workspace += cws * emb_dim;

    float *ff1 = workspace;
    workspace += cws * ff_dim;

    float *ff2 = workspace;

    layernorm(x, w->ln1_weight, w->ln1_bias, ln1_out, cws, emb_dim, eps);

    linear(ln1_out, w->Wq, w->bq, Q, cws, emb_dim, emb_dim, false);

    linear(ln1_out, w->Wk, w->bk, K, cws, emb_dim, emb_dim, false);

    linear(ln1_out, w->Wv, w->bv, V, cws, emb_dim, emb_dim, false);

    attention(Q, K, V, attn_out, attn_temp, cws, emb_dim, num_heads, emb_dim, 1);

    linear(attn_out, w->Wo, w->bo, proj_out, cws, emb_dim, emb_dim, false);

    add(x, proj_out, residual1, cws * emb_dim);

    layernorm(residual1, w->ln2_weight, w->ln2_bias, ln2_out, cws, emb_dim, eps);

    linear(ln2_out, w->W1, w->b1, ff1, cws, emb_dim, ff_dim, false);

    gelu(ff1, cws * ff_dim);

    linear(ff1, w->W2, w->b2, ff2, cws, ff_dim, emb_dim, false);

    add(residual1, ff2, out, cws * emb_dim);
}