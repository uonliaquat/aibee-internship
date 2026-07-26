# Team C – LLaMA Math Functions in C

This project implements core mathematical kernels used in LLaMA models and verifies their correctness against Hugging Face's TinyLlama. The implementation includes Rotary Positional Embeddings (RoPE) and the complete SwiGLU feed-forward block using BLIS for matrix multiplication.

## Implemented Components

- `linear_nobias` (Jazim) – Matrix multiplication without bias (BLIS backend)
- `SiLU` (Kiren)– Sigmoid Linear Unit activation
- `mul` (Kiren)– Element-wise multiplication
- `RoPE` (Shajia)– Rotary Positional Embedding
- `SwiGLU` (Shajia)– `linear(silu(linear(x)) * linear(x))`
- `RMSNorm` (Hanan)– Root mean squared normalization

## How to Build

```bash
make clean
make
```

## Run Tests

Create and activate the pthon environment
For a single test
```bash
python3 test/filename.py
```

or simply for all tests

```bash
make test
```

To Extract TinyLlama's inputs and weights:
```bash
python generate_answer_key.py
```

To run on TinyLlama's inputs and weights:
```bash
make output
```

## Expected Output

```
[PASS] kernalName
```

---

## What is RoPE?

Rotary Positional Embedding (RoPE) encodes token positions by rotating pairs of values in the Query and Key vectors instead of adding learned positional embeddings. Each pair is rotated by an angle determined by the token position and embedding dimension. This preserves relative positional information while allowing the model to generalize to longer sequences. LLaMA and other modern LLMs use the **rotate-half** formulation, which matches the Hugging Face implementation. This improves attention over long sequences while remaining computationally efficient.


## Verification
The kernels are validated against equivalent PyTorch implementations using randomly generated and also TinyLlama's actual inputs and weights. The implementation is designed to match Hugging Face TinyLlama's mathematical operations.