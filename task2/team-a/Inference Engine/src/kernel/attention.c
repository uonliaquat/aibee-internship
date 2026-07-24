#include "../../include/kernel/attention.h"
#include "../../include/kernel/softmax.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "blis.h"


void attention(const float *q, const float *k, const float *v, float *out,float *temp,
    size_t context_win_size, size_t emb_dim, size_t num_heads,inc_t rs,inc_t cs)
{
	size_t head_dim = emb_dim / num_heads;
	size_t offset;
	float infinity = -INFINITY;
	float scale_factor=1.0f/sqrt(head_dim);	
	float alpha =1,beta=0;
	inc_t temp_rs=1, temp_cs=context_win_size;
	for (size_t head = 0; head < num_heads; head++){
		offset = head_dim*head*cs;
		bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, context_win_size, context_win_size, head_dim, &alpha, q+offset, rs, cs, k+offset, rs, cs, &beta, temp, temp_rs, temp_cs);
		bli_sscalm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, context_win_size, context_win_size, &scale_factor, temp, temp_rs, temp_cs );
		bli_ssetm( BLIS_NO_CONJUGATE, 1, BLIS_NONUNIT_DIAG, BLIS_UPPER, context_win_size, context_win_size, &infinity, temp, temp_rs, temp_cs);
		softmax(temp,context_win_size,context_win_size,temp_rs,temp_cs);
		bli_sgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, context_win_size, head_dim, context_win_size, &alpha, temp, temp_rs, temp_cs, v+offset, rs, cs, &beta, out+offset, rs, cs);
	}
}