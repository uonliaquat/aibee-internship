#ifndef SOFTMAX_GELU_H
#define SOFTMAX_GELU_H

#include <stddef.h>


void softmax(float *x, int rows, int cols);


void gelu(float *x, size_t n);

#endif