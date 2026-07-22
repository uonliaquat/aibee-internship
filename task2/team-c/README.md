# Team C – LLaMA Math Functions in C

This project implements core mathematical kernels used in LLaMA models and verifies their correctness against Hugging Face's TinyLlama. The implementation includes Rotary Positional Embeddings (RoPE) and the complete SwiGLU feed-forward block using BLIS for matrix multiplication.

## Implemented Components

- `linear_nobias` (Jazim) – Matrix multiplication without bias (BLIS backend)
- `SiLU` (Kiren)– Sigmoid Linear Unit activation
- `mul` (Kiren)– Element-wise multiplication
- `RoPE` (Shajia)– Rotary Positional Embedding
- `SwiGLU` (Shajia)– `linear(silu(linear(x)) * linear(x))`

## How to Build

```bash
make clean
make
```

## Run Tests

```bash
python3 test/test_linear_nobias.py
python3 test/test_rope.py
python3 test/test_swiglu.py
```

or simply

```bash
make test
```

## Expected Output

```
[PASS] kernel_linear_nobias_cpu_f32_forward
[PASS] RoPE
[PASS] kernel_swiglu_cpu_f32_forward
```

---

## What is RoPE?

Rotary Positional Embedding (RoPE) encodes token positions by rotating pairs of values in the Query and Key vectors instead of adding learned positional embeddings. Each pair is rotated by an angle determined by the token position and embedding dimension. This preserves relative positional information while allowing the model to generalize to longer sequences. LLaMA and other modern LLMs use the **rotate-half** formulation, which matches the Hugging Face implementation. This improves attention over long sequences while remaining computationally efficient.

---

## Project Structure

```
inc/
    linear_nobias.h
    mul.h
    silu.h
    rope.h
    swiglu.h

src/
    linear_nobias.c
    mul.c
    silu.c
    rope.c
    swiglu.c

test/
    test_linear_nobias.py
    test_rope.py
    test_swiglu.py

build/
    libkernels.so
```

## Verification
The kernels are validated against equivalent PyTorch implementations using randomly generated inputs and weights. The implementation is designed to match Hugging Face TinyLlama's mathematical operations.