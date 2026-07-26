#include "../inc/silu.h"


// void printSilu(float *Array,int n)
// {
//     printf("{");
//     for(int i=0;i<n;i++)
//     {
//        printf("%f ", Array[i]);
//     }
//     printf("}");
// }

void SILU(float *input, float *output, uint32_t length)
{
    for(uint32_t i = 0; i < length; i++)
    {
        float sigmoid = 1.0 / (1.0 + expf(-input[i]));
        output[i] = input[i] * sigmoid;
    }
}
