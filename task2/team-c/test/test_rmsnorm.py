import ctypes
import numpy as np
import torch
import torch.nn as nn
import sys
import platform

_EXT = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}
lib = ctypes.CDLL(f"build/libkernels{_EXT[platform.system()]}")

lib.kernel_rmsnorm_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_float
]
lib.kernel_rmsnorm_cpu_f32_forward.restype = None

num_tests = 100
passed = 0
max_error = 0.0

for test_idx in range(num_tests):
    embed_dim = np.random.choice([512, 768, 1024, 2048, 4096])
    seq_len = np.random.randint(1, 512)
    eps = 1e-5

    embeddings = np.random.randn(seq_len, embed_dim).astype(np.float32)
    weights = np.random.randn(embed_dim).astype(np.float32)

    if test_idx % 3 == 0:
        embeddings *= 100.0
    elif test_idx % 3 == 1:
        embeddings *= 0.01

    out = np.zeros(seq_len * embed_dim, dtype=np.float32)

    lib.kernel_rmsnorm_cpu_f32_forward(
        embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        seq_len,
        embed_dim,
        eps
    )

    out_reshaped = out.reshape(seq_len, embed_dim)

    try:
        x_torch = torch.tensor(embeddings, dtype=torch.float32)
        w_torch = torch.tensor(weights, dtype=torch.float32)
        rms_norm = nn.RMSNorm(
            normalized_shape=x_torch.shape[-1],
            eps=eps,
            elementwise_affine=False,
        )

        y_torch = rms_norm(x_torch) * w_torch
        y_numpy = y_torch.numpy()
        abs_error = np.abs(out_reshaped - y_numpy)
        max_abs_error = np.max(abs_error)
        mean_abs_error = np.mean(abs_error)
        max_error = max(max_error, max_abs_error)

        try:
            np.testing.assert_allclose(out_reshaped, y_numpy, rtol=1e-5, atol=1e-5)
            passed += 1
        except AssertionError:
            if max_abs_error < 1e-4:
                passed += 1
            else:
                print(f"FAIL test {test_idx+1}: dim={embed_dim}, seq={seq_len}")
                print(f"  max_err={max_abs_error:.6e}, mean_err={mean_abs_error:.6e}")
                worst_idx = np.unravel_index(np.argmax(abs_error), abs_error.shape)
                print(f"  worst at {worst_idx}: C={out_reshaped[worst_idx]:.10f}, PyTorch={y_numpy[worst_idx]:.10f}")

    except Exception as e:
        print(f"ERROR test {test_idx+1}: {e}")

if passed == num_tests:
    print(f"\033[92m[PASS]\033[0m kernel_rmsnorm_cpu_f32_forward ({passed}/{num_tests} tests passed, max_err={max_error:.2e})")
    sys.exit(0)
else:
    print(f"\033[91m[FAIL]\033[0m kernel_rmsnorm_cpu_f32_forward ({passed}/{num_tests} tests passed, max_err={max_error:.2e})")
    sys.exit(1)