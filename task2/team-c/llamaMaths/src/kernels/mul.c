#include "kernels/mul.h"

void MUL(double *a, double *b, double *output, uint32_t length)
{
    for (uint32_t i = 0; i < length; i++)
    {
        output[i] = a[i] * b[i];
    }
}