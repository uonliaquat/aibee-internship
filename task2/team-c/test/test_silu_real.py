"""
test_silu_real.py
-----------------
Loads real TinyLlama SiLU input and correct output (produced by
generate_answer_key.py) and checks that our C kernel's SiLU output
matches TinyLlama's real output within the required tolerance.
"""

import ctypes
import numpy as np
import platform
import sys
import os

_EXT = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}
LIB_PATH = f"build/libkernels{_EXT[platform.system()]}"
RESULTS_DIR = "test/weights"
TOLERANCE = 1e-4

input_path = os.path.join(RESULTS_DIR, "silu_input.npy")
output_path = os.path.join(RESULTS_DIR, "silu_output.npy")

real_input = np.load(input_path).astype(np.float32)
real_output = np.load(output_path).astype(np.float32)

total_len = real_input.size

print(f"Loaded real TinyLlama data:")
print(f"  input shape: {real_input.shape}")
print(f"  output shape: {real_output.shape}")
print(f"  total elements: {total_len}")

lib = ctypes.CDLL(LIB_PATH)

lib.SILU.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint32
]
lib.SILU.restype = None

c_output = np.zeros(total_len, dtype=np.float32)

lib.SILU(
    real_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    c_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    total_len
)

c_output = c_output.reshape(real_output.shape)

abs_error = np.abs(c_output - real_output)
max_abs_error = np.max(abs_error)
mean_abs_error = np.mean(abs_error)

print(f"\nMax absolute error : {max_abs_error:.8f}")
print(f"Mean absolute error: {mean_abs_error:.8f}")
print(f"Required tolerance : {TOLERANCE}")

if max_abs_error < TOLERANCE:
    print(f"\n\033[92m[PASS]\033[0m silu matches real TinyLlama output within {TOLERANCE}")
    sys.exit(0)
else:
    worst_idx = np.unravel_index(np.argmax(abs_error), abs_error.shape)
    print(f"\n\033[91m[FAIL]\033[0m silu does NOT match within {TOLERANCE}")
    print(f"  Worst mismatch at {worst_idx}: C={c_output[worst_idx]:.8f}, TinyLlama={real_output[worst_idx]:.8f}")
    sys.exit(1)
