import ctypes
import platform
import sys

import numpy as np
import torch
import torch.nn.functional as F

_EXT = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}
lib = ctypes.CDLL(f"build/libkernels{_EXT[platform.system()]}")

lib.kernel_swiglu_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # gate_weight
    ctypes.POINTER(ctypes.c_float),  # up_weight
    ctypes.POINTER(ctypes.c_float),  # down_weight
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_size_t,                 # batch
    ctypes.c_size_t,                 # in_dim
    ctypes.c_size_t,                 # hidden_dim
    ctypes.c_size_t                  # out_dim
]
lib.kernel_swiglu_cpu_f32_forward.restype = None

num_tests = 100
passed = 0
max_error = 0.0

for test_idx in range(num_tests):
    batch = int(np.random.choice([1, 2, 4, 8, 16]))
    in_dim = int(np.random.choice([8, 16, 32, 64]))
    hidden_dim = int(np.random.choice([16, 32, 64, 128]))
    out_dim = int(np.random.choice([8, 16, 32, 64]))

    x = np.random.randn(batch, in_dim).astype(np.float32)
    gate_weight = np.random.randn(hidden_dim, in_dim).astype(np.float32)
    up_weight = np.random.randn(hidden_dim, in_dim).astype(np.float32)
    down_weight = np.random.randn(out_dim, hidden_dim).astype(np.float32)
    out = np.zeros((batch, out_dim), dtype=np.float32)

    lib.kernel_swiglu_cpu_f32_forward(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        up_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        down_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        batch,
        in_dim,
        hidden_dim,
        out_dim
    )

    x_t = torch.tensor(x, dtype=torch.float32)
    gate_t = torch.tensor(gate_weight, dtype=torch.float32)
    up_t = torch.tensor(up_weight, dtype=torch.float32)
    down_t = torch.tensor(down_weight, dtype=torch.float32)

    expected = F.linear(F.silu(F.linear(x_t, gate_t)) * F.linear(x_t, up_t), down_t)
    expected_np = expected.detach().numpy()

    abs_error = np.abs(out - expected_np)
    max_abs_error = float(np.max(abs_error))
    max_error = max(max_error, max_abs_error)

    try:
        np.testing.assert_allclose(out, expected_np, rtol=1e-4, atol=1e-3)
        passed += 1
    except AssertionError:
        print(
            f"FAIL test {test_idx + 1}: batch={batch}, "
            f"in={in_dim}, hidden={hidden_dim}, out={out_dim}"
        )
        print(f"  max_err={max_abs_error:.6e}")

if passed == num_tests:
    print(f"\033[92m[PASS]\033[0m kernel_swiglu_cpu_f32_forward ({passed}/{num_tests} tests passed, max_err={max_error:.2e})")
    sys.exit(0)

print(f"\033[91m[FAIL]\033[0m kernel_swiglu_cpu_f32_forward ({passed}/{num_tests} tests passed, max_err={max_error:.2e})")
sys.exit(1)
