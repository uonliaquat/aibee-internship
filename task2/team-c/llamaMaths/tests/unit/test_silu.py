import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import platform
import os

_EXT = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}

lib = ctypes.CDLL(f"build/libkernels{_EXT[platform.system()]}")

lib.SILU.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_uint32
]

lib.SILU.restype = None

num_test = 1000
length = 100

passed = 0


os.makedirs("tests/unit/results", exist_ok=True)

with open("tests/unit/results/silu_results.txt", "w") as file:

    file.write("=" * 60 + "\n")
    file.write("SiLU (Sigmoid Linear Unit) Test Results\n")
    file.write("=" * 60 + "\n\n")

    for test in range(num_test):

        a = np.random.rand(length)
        b = np.zeros_like(a)

        x = torch.tensor(a)
        expected = F.silu(x).numpy()

        lib.SILU(a, b, length)

        result = np.allclose(b, expected, rtol=1e-9, atol=1e-12)

        if result:
            passed += 1

        # Save first 10 test cases
        if test < 10:

            file.write(f"Test Case {test + 1}\n")
            file.write("-" * 60 + "\n")

            file.write(f"Input:\n{a}\n\n")

            file.write(f"C Output:\n{b}\n\n")

            file.write(f"PyTorch Output:\n{expected}\n\n")

            file.write(f"Status: {'PASS' if result else 'FAIL'}\n")
            file.write("\n" + "=" * 60 + "\n\n")

    file.write("\nSUMMARY\n")
    file.write("=" * 60 + "\n")
    file.write(f"Total Tests : {num_test}\n")
    file.write(f"Passed      : {passed}\n")
    file.write(f"Failed      : {num_test - passed}\n")
    file.write(f"Accuracy    : {(passed / num_test) * 100:.2f}%\n")

print(f"Passed: {passed}/{num_test}")
print("Detailed results saved in tests/unit/results/silu_results.txt")