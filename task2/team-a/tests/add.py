import ctypes
import numpy as np
import torch
import sys

lib = ctypes.CDLL("./build/libkernels.so")

lib.add.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t
]

lib.add.restype = None

num_tests = 100
passed = 0

for _ in range(num_tests):
    # Randomize the array size for each test to ensure robust bounds checking
    n = np.random.randint(1024, 1024 * 10) 
    
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)

    lib.add(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n
    )   

    try:
        t_a = torch.tensor(a, dtype=torch.float32)
        t_b = torch.tensor(b, dtype=torch.float32)
        y = t_a + t_b

        np.testing.assert_allclose(out, y.numpy(), rtol=1e-6, atol=1e-6)
        passed += 1
    except AssertionError as e:
        print(e)
        pass

if passed == num_tests:
    print(
        f"\033[92m[PASS]\033[0m add "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(0)
else:
    print(
        f"\033[91m[FAIL]\033[0m add "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(1)