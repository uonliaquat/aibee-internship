#include "../../include/kernels/linear.h"
#include "blis/blis.h"

void linear(
    const float* __restrict x, 
    const float* __restrict W,
    const float* __restrict b,
    float* __restrict output,
    size_t rows,
    size_t input_features,
    size_t output_features,
    bool transpose_W
){
    //BLIS gemm formula is C = beta*C + alpha*A*B
    float alpha=1.0f;
    float beta=0.0f;
    // BLIS object for A, stores the metadata describing the x matrix
    obj_t A;
    bli_obj_create_with_attached_buffer(
        BLIS_FLOAT,(dim_t)rows,(dim_t)input_features,
        (void*)x,
        (inc_t)input_features,(inc_t)1,
        &A
    );
    // BLIS object for B, stores the metadata describing the W matrix
    obj_t B;
    // handle whether the W matrix is stored normally or transpose
    if(!transpose_W){
        // W is [in_features × out_features]
        bli_obj_create_with_attached_buffer(
            BLIS_FLOAT,(dim_t)input_features,(dim_t)output_features,
            (void*)W,
            (inc_t)output_features,(inc_t)1,
            &B
        );
    }else{
        // W is [out_features × in_features]
        bli_obj_create_with_attached_buffer(
            BLIS_FLOAT,(dim_t)output_features,(dim_t)input_features,
            (void*)W,
            (inc_t)input_features,(inc_t)1,
            &B
        );
        bli_obj_set_conjtrans(BLIS_TRANSPOSE,&B); //Treat the W matrix as transpose during GEMM without copying
    }
    obj_t C;
    bli_obj_create_with_attached_buffer(
        BLIS_FLOAT,(dim_t)rows,(dim_t)output_features,
        (void*)output,
        (inc_t)output_features,(inc_t)1,
        &C
    );

};
