#include "../../inc/kernels/add.h"

// naive element by element add
void add(
    const float* a,
    const float* b,
    float* out,
    const size_t n
) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

// // slightly optimized version
// void add(
//     const float* a,
//     const float* b,
//     float* out,
//     const size_t n
// ) {
//     size_t i = 0;
//     for (; i + 7 < n; i += 8) {
//         out[i + 0] = a[i + 0] + b[i + 0];
//         out[i + 1] = a[i + 1] + b[i + 1];
//         out[i + 2] = a[i + 2] + b[i + 2];
//         out[i + 3] = a[i + 3] + b[i + 3];
//         out[i + 4] = a[i + 4] + b[i + 4];
//         out[i + 5] = a[i + 5] + b[i + 5];
//         out[i + 6] = a[i + 6] + b[i + 6];
//         out[i + 7] = a[i + 7] + b[i + 7];
//     }
//     for (; i < n; i++) {
//         out[i] = a[i] + b[i];
//     }
// }