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
    float beta;

    // Beta trick: pre-fill output with bias, then let BLIS keep it via beta=1.0
    if (b != NULL) {
        for (size_t i = 0; i < rows; i++)
            memcpy(output + i * output_features, b, output_features * sizeof(float));
        beta = 1.0f;   // C = A*B + 1.0*C → preserves bias
    } else {
        beta = 0.0f;   // C = A*B + 0.0*C → ignores C
    }

    // ... rest stays exactly the same (A, B, C objects, bli_gemm) ...

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

    // creating the scalar objects for alpha and beta because the gemm function does not accept C floats
    obj_t Alpha,Beta;
    bli_obj_create_1x1(BLIS_FLOAT,&Alpha);
    bli_obj_create_1x1(BLIS_FLOAT,&Beta);
    bli_setsc(alpha,0.0f,&Alpha); // the 0.0f is the imaginary part
    bli_setsc(beta,0.0f,&Beta);

    //C = alpha * A * B + beta * C
    bli_gemm(&Alpha,&A,&B,&Beta,&C);

}


