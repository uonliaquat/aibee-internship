"""
Answer-key test for the linear kernel.

Loads the C shared library, runs linear(x, W, b) for many random inputs,
and compares against PyTorch's F.linear (the oracle / answer key).

Tests both the normal case (W not transposed) and the transposed case.
"""
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import sys

# ---------- load shared library ----------
lib = ctypes.CDLL("./build/libkernels.so")

lib.linear.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.POINTER(ctypes.c_float),  # W
    ctypes.POINTER(ctypes.c_float),  # b
    ctypes.POINTER(ctypes.c_float),  # out
    ctypes.c_size_t,                 # rows
    ctypes.c_size_t,                 # in_features
    ctypes.c_size_t,                 # out_features
    ctypes.c_bool,                   # transpose_W
]
lib.linear.restype = None


def ptr(arr):
    """Get a ctypes float pointer from a numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run_one_test(rows, in_feat, out_feat, transpose_W, atol=1e-4):
    """
    Run one linear test case.

    PyTorch's F.linear computes:  out = x @ W^T + b
    so its weight is stored as [out_features × in_features].

    Our C function with transpose_W=False expects W as [in_features × out_features].
    Our C function with transpose_W=True  expects W as [out_features × in_features].
    """
    # random inputs
    x = np.random.randn(rows, in_feat).astype(np.float32)
    b = np.random.randn(out_feat).astype(np.float32)

    if transpose_W:
        # W stored as [out_features × in_features]  (same as PyTorch)
        W = np.random.randn(out_feat, in_feat).astype(np.float32)
    else:
        # W stored as [in_features × out_features]  (column layout for matmul)
        W = np.random.randn(in_feat, out_feat).astype(np.float32)

    out_c = np.zeros((rows, out_feat), dtype=np.float32)

    # --- run C kernel ---
    lib.linear(ptr(x), ptr(W), ptr(b), ptr(out_c),
               rows, in_feat, out_feat, transpose_W)

    # --- run PyTorch oracle ---
    x_t = torch.tensor(x)
    b_t = torch.tensor(b)

    if transpose_W:
        # W is [out × in], same layout PyTorch expects
        W_t = torch.tensor(W)
    else:
        # W is [in × out], PyTorch wants [out × in]
        W_t = torch.tensor(W.T.copy())

    out_torch = F.linear(x_t, W_t, b_t).numpy()

    # --- compare ---
    np.testing.assert_allclose(out_c, out_torch, atol=atol, rtol=1e-5)


# ---------- test suite ----------
num_tests = 50
passed = 0
failed = 0

# GPT-2 realistic sizes
test_configs = [
    # (rows, in_features, out_features, transpose_W)
    (1024, 768,  768,  False),  # Q, K, V projections
    (1024, 768,  768,  True),   # same but W transposed
    (1024, 768,  3072, False),  # MLP up-projection
    (1024, 3072, 768,  False),  # MLP down-projection
    (1024, 768,  3072, True),   # MLP up-projection, transposed
    (1024, 3072, 768,  True),   # MLP down-projection, transposed
    (1,    768,  768,  False),  # single token
    (1,    768,  768,  True),   # single token, transposed
    (64,   768,  768,  False),  # small batch
    (256,  768,  3072, True),   # medium batch, transposed
]

for i in range(num_tests):
    cfg = test_configs[i % len(test_configs)]
    rows, in_f, out_f, trans = cfg
    label = f"linear({rows}×{in_f} @ {'T' if trans else ''}{in_f}×{out_f})"
    try:
        run_one_test(rows, in_f, out_f, trans)
        passed += 1
    except AssertionError as e:
        failed += 1
        print(f"  \033[91m[FAIL]\033[0m {label}: {e}")
    except Exception as e:
        failed += 1
        print(f"  \033[91m[FAIL]\033[0m {label}: {e}")

if passed == num_tests:
    print(
        f"\033[92m[PASS]\033[0m linear "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(0)
else:
    print(
        f"\033[91m[FAIL]\033[0m linear "
        f"({passed}/{num_tests} tests passed, {failed} failed)"
    )
    sys.exit(1)
