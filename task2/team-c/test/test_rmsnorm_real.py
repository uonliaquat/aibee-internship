"""
test_rmsnorm_real.py
--------------------
Loads real TinyLlama input, weight, eps, and correct output (produced by
generate_answer_key.py) and checks that our C kernel's RMSNorm output
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

hidden_states_path = os.path.join(RESULTS_DIR, "layer0_input_hidden_states.npy")
weight_path = os.path.join(RESULTS_DIR, "rmsnorm_weight.npy")
eps_path = os.path.join(RESULTS_DIR, "rmsnorm_eps.npy")
output_path = os.path.join(RESULTS_DIR, "rmsnorm_output.npy")

hidden_states = np.load(hidden_states_path).astype(np.float32)
weight = np.load(weight_path).astype(np.float32)
eps = np.load(eps_path).astype(np.float32).item()
real_output = np.load(output_path).astype(np.float32)

seq_len, embed_dim = hidden_states.shape

print(f"Loaded real TinyLlama data:")
print(f"  hidden_states shape: {hidden_states.shape}")
print(f"  weight shape: {weight.shape}")
print(f"  eps: {eps}")
print(f"  output shape: {real_output.shape}")

lib = ctypes.CDLL(LIB_PATH)

lib.kernel_rmsnorm_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_float
]
lib.kernel_rmsnorm_cpu_f32_forward.restype = None

c_output = np.zeros(seq_len * embed_dim, dtype=np.float32)

lib.kernel_rmsnorm_cpu_f32_forward(
    hidden_states.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    c_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    seq_len,
    embed_dim,
    eps
)

c_output = c_output.reshape(seq_len, embed_dim)

abs_error = np.abs(c_output - real_output)
max_abs_error = np.max(abs_error)
mean_abs_error = np.mean(abs_error)

print(f"\nMax absolute error : {max_abs_error:.8f}")
print(f"Mean absolute error: {mean_abs_error:.8f}")
print(f"Required tolerance : {TOLERANCE}")

if max_abs_error < TOLERANCE:
    print(f"\n\033[92m[PASS]\033[0m rmsnorm matches real TinyLlama output within {TOLERANCE}")
    sys.exit(0)
else:
    worst_idx = np.unravel_index(np.argmax(abs_error), abs_error.shape)
    print(f"\n\033[91m[FAIL]\033[0m rmsnorm does NOT match within {TOLERANCE}")
    print(f"  Worst mismatch at {worst_idx}: C={c_output[worst_idx]:.8f}, TinyLlama={real_output[worst_idx]:.8f}")
    sys.exit(1)
