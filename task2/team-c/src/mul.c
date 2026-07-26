#include "../inc/mul.h"

void MUL(float *a, float *b, float *output, uint32_t length)
{
    for (uint32_t i = 0; i < length; i++)
    {
        output[i] = a[i] * b[i];
    }
}