import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import sys

lib = ctypes.CDLL("./build/libattention.so")

lib.layernorm.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_float
]

lib.layernorm.restype = None

num_tests = 100
passed = 0
for _ in range(num_tests):
    embed_dim = 786
    ctx_win = 1024
    eps = 1e-5
    # n = np.random.randint(1, ctx_win)
    embeddings = np.random.randn(ctx_win, embed_dim).astype(np.float32)
    weights = np.random.randn(embed_dim).astype(np.float32)
    bias = np.random.randn(embed_dim).astype(np.float32)

    out = np.zeros(ctx_win*embed_dim, dtype=np.float32)

    lib.layernorm(
        embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctx_win,
        embed_dim,
        eps
    )   
    try:
        y = F.layer_norm(
            torch.tensor(embeddings, dtype=torch.float32),
            normalized_shape=(embed_dim,),
            weight=torch.tensor(weights, dtype=torch.float32),
            bias=torch.tensor(bias, dtype=torch.float32),
            eps=eps,
        )
        np.testing.assert_allclose(out, y.flatten().numpy(), rtol=1e-6, atol=1e-6)
        passed +=1
    except AssertionError as e:
        print(e)
        pass

if passed == num_tests:
    print(
        f"\033[92m[PASS]\033[0m layernorm "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(0)
else:
    print(
        f"\033[91m[FAIL]\033[0m layernorm "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(1)
