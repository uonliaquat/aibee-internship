# LLaMA Math Kernels

## Implemented Kernels

This module contains implementations of mathematical kernels used in the LLaMA architecture. The kernels are written in C and validated against PyTorch using Python unit tests.

### SiLU (Sigmoid Linear Unit)

- Computes the SiLU activation function: `SiLU(x) = x * sigmoid(x)`.
- Processes each input element independently.
- Validated against `torch.nn.functional.silu` over 1000 random test cases.

### Element-wise Multiplication (MUL)

- Performs element-wise multiplication of two input tensors.
- Computes `output[i] = a[i] * b[i]` for every element.
- Validated against NumPy element-wise multiplication over 1000 random test cases.