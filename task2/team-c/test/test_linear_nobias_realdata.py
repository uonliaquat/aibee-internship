"""
test_linear_nobias_realdata.py
--------------------------------
Loads the REAL TinyLlama input, weight, and correct output (produced by
generate_answer_key.py) and checks that our C kernel's output matches
TinyLlama's real output within the required tolerance.
"""

import ctypes
import numpy as np
import platform
import sys
import os

_EXT = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}
LIB_PATH = f"build/libkernels{_EXT[platform.system()]}"
RESULTS_DIR = "test/results"
TOLERANCE = 1e-4

input_path  = os.path.join(RESULTS_DIR, "linear_nobias_input.npy")
weight_path = os.path.join(RESULTS_DIR, "linear_nobias_weight_qproj.npy")
output_path = os.path.join(RESULTS_DIR, "linear_nobias_output_qproj.npy")

real_input  = np.load(input_path).astype(np.float32)
real_weight = np.load(weight_path).astype(np.float32)
real_output = np.load(output_path).astype(np.float32)

seq_len, in_features = real_input.shape
out_features = real_weight.shape[0]

print(f"Loaded real TinyLlama data:")
print(f"  input  shape: {real_input.shape}")
print(f"  weight shape: {real_weight.shape}")
print(f"  output shape: {real_output.shape}")

lib = ctypes.CDLL(LIB_PATH)

lib.kernel_linear_nobias_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t
]
lib.kernel_linear_nobias_cpu_f32_forward.restype = None

c_output = np.zeros(seq_len * out_features, dtype=np.float32)

lib.kernel_linear_nobias_cpu_f32_forward(
    real_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    real_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    c_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    seq_len,
    in_features,
    out_features
)

c_output = c_output.reshape(seq_len, out_features)

abs_error = np.abs(c_output - real_output)
max_abs_error = np.max(abs_error)
mean_abs_error = np.mean(abs_error)

print(f"\nMax absolute error : {max_abs_error:.8f}")
print(f"Mean absolute error: {mean_abs_error:.8f}")
print(f"Required tolerance : {TOLERANCE}")

if max_abs_error < TOLERANCE:
    print(f"\n\033[92m[PASS]\033[0m linear_nobias matches real TinyLlama output within {TOLERANCE}")
    sys.exit(0)
else:
    worst_idx = np.unravel_index(np.argmax(abs_error), abs_error.shape)
    print(f"\n\033[91m[FAIL]\033[0m linear_nobias does NOT match within {TOLERANCE}")
    print(f"  Worst mismatch at {worst_idx}: C={c_output[worst_idx]:.8f}, TinyLlama={real_output[worst_idx]:.8f}")
    sys.exit(1)