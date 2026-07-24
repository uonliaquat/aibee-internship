#include <stdio.h>
#include <stdlib.h>

#include "include/gpt2.h"
#include "include/kernel/weights.h"

int main(void)
{
    const size_t cws = 4;
    const size_t embed_dim = 8;
    const size_t num_heads = 2;
    const float eps = 1e-5f;

    float input[cws * embed_dim];
    float output[cws * embed_dim];

    for (size_t i = 0; i < cws * embed_dim; i++)
        input[i] = (float)i / 10.0f;

    GPT2Weights w;
    size_t size = size;

    w.ln1_weight = malloc(embed_dim * size);
    w.ln1_bias   = malloc(embed_dim * size);

    w.Wq = malloc(embed_dim * embed_dim * size);
    w.bq = malloc(embed_dim * size);

    w.Wk = malloc(embed_dim * embed_dim * size);
    w.bk = malloc(embed_dim * size);

    w.Wv = malloc(embed_dim * embed_dim * size);
    w.bv = malloc(embed_dim * size);

    w.Wo = malloc(embed_dim * embed_dim * size);
    w.bo = malloc(embed_dim * size);

    w.ln2_weight = malloc(embed_dim * size);
    w.ln2_bias   = malloc(embed_dim * size);

    w.W1 = malloc(embed_dim * (4 * embed_dim) * size);
    w.b1 = malloc((4 * embed_dim) * size);

    w.W2 = malloc((4 * embed_dim) * embed_dim * size);
    w.b2 = malloc(embed_dim * size);


    for (size_t i = 0; i < embed_dim; i++)
    {
        w.ln1_weight[i] = 1.0f;
        w.ln2_weight[i] = 1.0f;

        w.ln1_bias[i] = 0.0f;
        w.ln2_bias[i] = 0.0f;

        w.bq[i] = 0.0f;
        w.bk[i] = 0.0f;
        w.bv[i] = 0.0f;
        w.bo[i] = 0.0f;
        w.b2[i] = 0.0f;
    }

    for (size_t i = 0; i < 4 * embed_dim; i++)
        w.b1[i] = 0.0f;

    for (size_t i = 0; i < embed_dim * embed_dim; i++)
    {
        w.Wq[i] = 0.01f;
        w.Wk[i] = 0.01f;
        w.Wv[i] = 0.01f;
        w.Wo[i] = 0.01f;
    }

    for (size_t i = 0; i < embed_dim * (4 * embed_dim); i++)
        w.W1[i] = 0.01f;

    for (size_t i = 0; i < (4 * embed_dim) * embed_dim; i++)
        w.W2[i] = 0.01f;


    size_t workspace_size =
        cws * embed_dim +          // ln1
        cws * embed_dim +          // Q
        cws * embed_dim +          // K
        cws * embed_dim +          // V
        cws * cws +            // attention temp
        cws * embed_dim +          // attention out
        cws * embed_dim +          // projection
        cws * embed_dim +          // residual1
        cws * embed_dim +          // ln2
        cws * (4 * embed_dim) +    // ff1
        cws * embed_dim;           // ff2

    float *workspace = malloc(workspace_size * size);

    gpt2_block(
        input,
        output,
        &w,
        workspace,
        cws,
        embed_dim,
        num_heads,
        eps);

    printf("Output:\n");

    for (size_t i = 0; i < cws; i++)
    {
        for (size_t j = 0; j < embed_dim; j++)
            printf("%8.4f ", output[i * embed_dim + j]);

        printf("\n");
    }

    free(workspace);

    free(w.ln1_weight);
    free(w.ln1_bias);

    free(w.Wq);
    free(w.bq);

    free(w.Wk);
    free(w.bk);

    free(w.Wv);
    free(w.bv);

    free(w.Wo);
    free(w.bo);

    free(w.ln2_weight);
    free(w.ln2_bias);

    free(w.W1);
    free(w.b1);

    free(w.W2);
    free(w.b2);

    return 0;
}