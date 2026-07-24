#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <stddef.h>

typedef struct
{
    float *ln1_weight;
    float *ln1_bias;

    float *Wq;
    float *bq;

    float *Wk;
    float *bk;

    float *Wv;
    float *bv;

    float *Wo;
    float *bo;

    float *ln2_weight;
    float *ln2_bias;

    float *W1;
    float *b1;

    float *W2;
    float *b2;

} GPT2Weights;

#endif