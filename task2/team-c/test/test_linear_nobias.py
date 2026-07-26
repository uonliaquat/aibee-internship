import ctypes
import numpy as np
import torch
import torch.nn as nn
import sys
import platform

_EXT = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}
lib = ctypes.CDLL(f"build/libkernels{_EXT[platform.system()]}")

lib.kernel_linear_nobias_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # weight
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_size_t,                 # seq_len
    ctypes.c_size_t,                 # in_features
    ctypes.c_size_t                  # out_features
]
lib.kernel_linear_nobias_cpu_f32_forward.restype = None

num_tests = 100
passed = 0
max_error = 0.0

for test_idx in range(num_tests):
    in_features = int(np.random.choice([512, 768, 1024, 2048, 4096]))
    out_features = int(np.random.choice([512, 768, 1024, 2048, 4096]))
    seq_len = np.random.randint(1, 512)

    embeddings = np.random.randn(seq_len, in_features).astype(np.float32)
    weight = np.random.randn(out_features, in_features).astype(np.float32)

    if test_idx % 3 == 0:
        embeddings *= 100.0
    elif test_idx % 3 == 1:
        embeddings *= 0.01

    out = np.zeros(seq_len * out_features, dtype=np.float32)

    lib.kernel_linear_nobias_cpu_f32_forward(
        embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        seq_len,
        in_features,
        out_features
    )

    out_reshaped = out.reshape(seq_len, out_features)

    try:
        x_torch = torch.tensor(embeddings, dtype=torch.float32)
        w_torch = torch.tensor(weight, dtype=torch.float32)

        linear = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            linear.weight.copy_(w_torch)

        y_torch = linear(x_torch)
        y_numpy = y_torch.detach().numpy()

        try:
            np.testing.assert_allclose(out_reshaped, y_numpy, rtol=1e-4, atol=1e-2)
            passed += 1
        except AssertionError:
            print(f"FAIL test {test_idx+1}: seq={seq_len}, in={in_features}, out={out_features}")

    except Exception as e:
        print(f"ERROR test {test_idx+1}: {e}")

if passed == num_tests:
    print(f"\033[92m[PASS]\033[0m kernel_linear_nobias_cpu_f32_forward ({passed}/{num_tests} tests passed, max_err={max_error:.2e})")
    sys.exit(0)
else:
    print(f"\033[91m[FAIL]\033[0m kernel_linear_nobias_cpu_f32_forward ({passed}/{num_tests} tests passed, max_err={max_error:.2e})")
    sys.exit(1)
