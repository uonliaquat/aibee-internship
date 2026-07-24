#include "../../include/kernel/softmax.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "blis.h"


void softmax( float *out,size_t row,size_t col,inc_t rs,inc_t cs)
{
	float sum=0.0f;
	float* temp=out;
	float max=-INFINITY;
	size_t offset;
	for (int r = 0; r < row; r++)
	{
		offset = rs*r;
		for (size_t c = 0; c < col; c++)
		max=fmaxf(max,(temp+offset)[c*cs]);
		for (size_t c = 0; c < col; c++)
		{
			(temp+offset)[c*cs] = expf((temp+offset)[c*cs] - max);
			sum+=(temp+offset)[c*cs];
		}
		float scale_factor = 1.0f/sum;
		bli_sscalm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, 1, col, &scale_factor, out+offset, rs, cs );
		max=-INFINITY;
		sum=0.0f;
	}
}