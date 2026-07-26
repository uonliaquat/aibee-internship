"""
test_rope_real.py
-----------------
Loads real TinyLlama RoPE inputs (Q, K, cos, sin) and correct rotated
outputs (produced by generate_answer_key.py) and checks that our C
kernel's RoPE output matches TinyLlama's real output within the
required tolerance.

TinyLlama uses GQA: 32 Q heads and 4 KV heads, each with head_dim=64.
We test the first head (index 0) by slicing the full tensors down to
(seq_len, head_dim).
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

HEAD_DIM = 64

q_input_path = os.path.join(RESULTS_DIR, "rope_input_q.npy")
k_input_path = os.path.join(RESULTS_DIR, "rope_input_k.npy")
cos_path = os.path.join(RESULTS_DIR, "rope_cos.npy")
sin_path = os.path.join(RESULTS_DIR, "rope_sin.npy")
q_output_path = os.path.join(RESULTS_DIR, "rope_output_q.npy")
k_output_path = os.path.join(RESULTS_DIR, "rope_output_k.npy")

full_q = np.load(q_input_path).astype(np.float32)
full_k = np.load(k_input_path).astype(np.float32)
full_cos = np.load(cos_path).astype(np.float32)
full_sin = np.load(sin_path).astype(np.float32)
full_q_expected = np.load(q_output_path).astype(np.float32)
full_k_expected = np.load(k_output_path).astype(np.float32)

seq_len = full_q.shape[0]
num_q_heads = full_q.shape[1] // HEAD_DIM
num_k_heads = full_k.shape[1] // HEAD_DIM

q = full_q.reshape(seq_len, num_q_heads, HEAD_DIM)[:, 0, :].copy()
k = full_k.reshape(seq_len, num_k_heads, HEAD_DIM)[:, 0, :].copy()
cos = full_cos.copy()
sin = full_sin.copy()
q_expected = full_q_expected.reshape(seq_len, num_q_heads, HEAD_DIM)[:, 0, :]
k_expected = full_k_expected.reshape(seq_len, num_k_heads, HEAD_DIM)[:, 0, :]

print(f"Loaded real TinyLlama data (head 0 of Q={num_q_heads}, K={num_k_heads}):")
print(f"  q shape: {q.shape}")
print(f"  k shape: {k.shape}")
print(f"  cos shape: {cos.shape}")
print(f"  sin shape: {sin.shape}")

lib = ctypes.CDLL(LIB_PATH)

lib.kernel_rope_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t
]
lib.kernel_rope_cpu_f32_forward.restype = None

lib.kernel_rope_cpu_f32_forward(
    q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    cos.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    sin.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    seq_len,
    HEAD_DIM
)

q_abs_error = np.abs(q - q_expected)
k_abs_error = np.abs(k - k_expected)
q_max_error = np.max(q_abs_error)
k_max_error = np.max(k_abs_error)
mean_q_error = np.mean(q_abs_error)
mean_k_error = np.mean(k_abs_error)
max_abs_error = max(q_max_error, k_max_error)

print(f"\nQ - Max error: {q_max_error:.8f}, Mean error: {mean_q_error:.8f}")
print(f"K - Max error: {k_max_error:.8f}, Mean error: {mean_k_error:.8f}")
print(f"Required tolerance: {TOLERANCE}")

if max_abs_error < TOLERANCE:
    print(f"\n\033[92m[PASS]\033[0m rope matches real TinyLlama output within {TOLERANCE}")
    sys.exit(0)
else:
    if q_max_error > k_max_error:
        worst_idx = np.unravel_index(np.argmax(q_abs_error), q_abs_error.shape)
        print(f"\n\033[91m[FAIL]\033[0m rope does NOT match within {TOLERANCE}")
        print(f"  Worst Q mismatch at {worst_idx}: C={q[worst_idx]:.8f}, TinyLlama={q_expected[worst_idx]:.8f}")
    else:
        worst_idx = np.unravel_index(np.argmax(k_abs_error), k_abs_error.shape)
        print(f"\n\033[91m[FAIL]\033[0m rope does NOT match within {TOLERANCE}")
        print(f"  Worst K mismatch at {worst_idx}: C={k[worst_idx]:.8f}, TinyLlama={k_expected[worst_idx]:.8f}")
    sys.exit(1)
