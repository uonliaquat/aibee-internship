import ctypes
import platform

import numpy as np
import torch

_EXT = {
    "Linux":".so",
    "Darwin":".dylib",
    "Windows":".dll"
}

lib = ctypes.CDLL(
    f"build/libkernels{_EXT[platform.system()]}"
)

lib.kernel_rope_cpu_f32_forward.argtypes = [

    ctypes.POINTER(ctypes.c_float),

    ctypes.POINTER(ctypes.c_float),

    ctypes.POINTER(ctypes.c_float),

    ctypes.POINTER(ctypes.c_float),

    ctypes.c_size_t,

    ctypes.c_size_t
]

lib.kernel_rope_cpu_f32_forward.restype = None

seq_len = 16
head_dim = 64

q = np.random.randn(seq_len, head_dim).astype(np.float32)
k = np.random.randn(seq_len, head_dim).astype(np.float32)

angles = np.random.randn(seq_len, head_dim).astype(np.float32)

cos = np.cos(angles).astype(np.float32)
sin = np.sin(angles).astype(np.float32)

q_original = q.copy()
k_original = k.copy()

lib.kernel_rope_cpu_f32_forward(

    q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    cos.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    sin.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    seq_len,
    head_dim
)

def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

q_t = torch.tensor(q_original)
k_t = torch.tensor(k_original)

cos_t = torch.tensor(cos)
sin_t = torch.tensor(sin)

q_expected = q_t * cos_t + rotate_half(q_t) * sin_t
k_expected = k_t * cos_t + rotate_half(k_t) * sin_t

np.testing.assert_allclose(
    q,
    q_expected.numpy(),
    atol=1e-4,
    rtol=1e-4
)

np.testing.assert_allclose(
    k,
    k_expected.numpy(),
    atol=1e-4,
    rtol=1e-4
)

print("[PASS] RoPE")

