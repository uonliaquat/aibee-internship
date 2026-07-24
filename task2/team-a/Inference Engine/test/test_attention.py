import ctypes
import numpy as np
import torch
import math
import time

lib = ctypes.CDLL("../build/libattention.so")


lib.attention.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # q
    ctypes.POINTER(ctypes.c_float),   # k
    ctypes.POINTER(ctypes.c_float),   # v
    ctypes.POINTER(ctypes.c_float),   # out
    ctypes.POINTER(ctypes.c_float),   # temp
    ctypes.c_size_t,                  # context
    ctypes.c_size_t,                  # embed
    ctypes.c_size_t,                  # heads
    ctypes.c_long,                    # rs
    ctypes.c_long                     # cs
]

lib.attention.restype = None


def attention_python(q, k, v, heads):

    ctx = q.shape[0]
    embed = q.shape[1]
    head_dim = embed // heads

    output = np.zeros_like(q)

    for h in range(heads):

        cols = slice(h * head_dim, (h + 1) * head_dim)

        qh = torch.tensor(q[:, cols])
        kh = torch.tensor(k[:, cols])
        vh = torch.tensor(v[:, cols])

        scores = (qh @ kh.T) / math.sqrt(head_dim)
        mask = torch.triu(
            torch.ones(ctx, ctx),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, -float('inf'))

        probs = torch.softmax(scores, dim=-1)

        output[:, cols] = (probs @ vh).numpy()

    return output



ctx = 1024
embed = 768
heads = 12

head_dim = embed // heads

q = np.random.randn(ctx, embed).astype(np.float32)
k = np.random.randn(ctx, embed).astype(np.float32)
v = np.random.randn(ctx, embed).astype(np.float32)

c_out = np.zeros((ctx, embed), dtype=np.float32)

temp = np.zeros((ctx, ctx), dtype=np.float32)


ts_c = time.perf_counter()

lib.attention(

    q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    temp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),

    ctx,
    embed,
    heads,

    embed,
    1
)

te_c = time.perf_counter()


ts_p = time.perf_counter()

p_out = attention_python(q, k, v, heads)

te_p = time.perf_counter()


# print("C OUTPUT\n")
# print(c_out)
#
# print("PYTORCH\n")
# print(p_out)

diff = np.abs(c_out - p_out)


# print("\nMax Error :", diff.max())

try:

    np.testing.assert_allclose(
        c_out,
        p_out,
        rtol=1e-4,
        atol=1e-5
    )

    print("\nPASS")

except AssertionError as e:

    print("\nFAIL")
    # print(e)

print("\nC Time       :", (te_c - ts_c) * 1000, "ms")
print("PyTorch Time :", (te_p - ts_p) * 1000, "ms")
