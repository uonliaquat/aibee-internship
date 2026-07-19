#include "../../include/kernels/linear.h"
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    size_t rows = 1024;
    size_t in_f = 768;
    size_t out_f = 3072;
    int runs = 50;

    float* x   = malloc(rows * in_f  * sizeof(float));
    float* W   = malloc(in_f * out_f * sizeof(float));
    float* b   = malloc(out_f * sizeof(float));
    float* out = malloc(rows * out_f * sizeof(float));

    // Random values
    srand(42);
    for (size_t i = 0; i < rows * in_f; i++)
        x[i] = (float)rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < in_f * out_f; i++)
        W[i] = (float)rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < out_f; i++)
        b[i] = (float)rand() / RAND_MAX - 0.5f;

    // Warmup (first run is always slow — cache cold)
    linear(x, W, b, out, rows, in_f, out_f, false);

    // Benchmark
    double start = get_time();
    for (int i = 0; i < runs; i++)
        linear(x, W, b, out, rows, in_f, out_f, false);
    double end = get_time();

    double avg_ms = (end - start) / runs * 1000.0;
    printf("linear %zu×%zu @ %zu×%zu\n", rows, in_f, in_f, out_f);
    printf("Avg time: %.3f ms  (%d runs)\n", avg_ms, runs);
    printf("out[0] = %f  (sanity check)\n", out[0]);

    free(x); free(W); free(b); free(out);
    return 0;
}
