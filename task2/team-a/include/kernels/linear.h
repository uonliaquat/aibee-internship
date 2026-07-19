#ifndef LINEAR_H
#define LINEAR_H
#include<stddef.h>
#include<stdbool.h>

/*
linear - out=x*W + b

rows: no. of rows in x
input_features: no. of columns in x
output_features: no. of columns in W, length of b

x is input matrix, [rows * input_features]
W is the weight matrix, [input_features * output_features]
out is the resultant matrix, [rows * output_features]
b is the bias, the length of this vector is equal to the output_features

Memory layout: (row major)

x[i][j]=x[i*input_features+j]
W[i][j]=W[i*output_features+j]
output[i][j]=output[i*output_features+j]
*/

void linear(
    const float* __restrict x, //restrict for optimization so compiler can know these pointers dont overlap
    const float* __restrict W,
    const float* __restrict b,
    float* __restrict output,
    size_t rows,
    size_t input_features,
    size_t output_features,
    bool transpose_W
);

#endif