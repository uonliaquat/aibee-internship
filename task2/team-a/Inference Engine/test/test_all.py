
import time
import ctypes
import numpy as np
import torch


SEQ = 128
EMB = 768
HEADS = 12
FF = EMB * 4
EPS = 1e-5

np.random.seed(0)
torch.manual_seed(0)


x = np.random.randn(SEQ, EMB).astype(np.float32)

weights = {
    "ln1_weight": np.random.randn(EMB).astype(np.float32),
    "ln1_bias": np.random.randn(EMB).astype(np.float32),

    "Wq": np.random.randn(EMB, EMB).astype(np.float32),
    "bq": np.random.randn(EMB).astype(np.float32),

    "Wk": np.random.randn(EMB, EMB).astype(np.float32),
    "bk": np.random.randn(EMB).astype(np.float32),

    "Wv": np.random.randn(EMB, EMB).astype(np.float32),
    "bv": np.random.randn(EMB).astype(np.float32),

    "Wo": np.random.randn(EMB, EMB).astype(np.float32),
    "bo": np.random.randn(EMB).astype(np.float32),

    "ln2_weight": np.random.randn(EMB).astype(np.float32),
    "ln2_bias": np.random.randn(EMB).astype(np.float32),

    "W1": np.random.randn(EMB, FF).astype(np.float32),
    "b1": np.random.randn(FF).astype(np.float32),

    "W2": np.random.randn(FF, EMB).astype(np.float32),
    "b2": np.random.randn(EMB).astype(np.float32),
}


class Timer:
    def __init__(self):
        self.times = {}

    def measure(self, name, fn):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        self.times[name] = (t1 - t0) * 1000.0
        return out

    def report(self, title):
        print("\\n" + "=" * 60)
        print(title)
        print("=" * 60)
        total = 0.0
        for k, v in self.times.items():
            total += v
            print(f"{k:20s}: {v:10.4f} ms")
        print("-" * 60)
        print(f"{'TOTAL':20s}: {total:10.4f} ms")



def layernorm(x, w, b):
    m = x.mean(axis=1, keepdims=True)
    v = x.var(axis=1, keepdims=True)
    y = (x - m) / np.sqrt(v + EPS)
    return y * w + b

def linear(x, W, b):
    return x @ W + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(
        np.sqrt(2/np.pi)*(x + 0.044715*x**3)
    ))

def attention(q,k,v):
    head_dim = EMB // HEADS

    q = q.reshape(SEQ, HEADS, head_dim)
    k = k.reshape(SEQ, HEADS, head_dim)
    v = v.reshape(SEQ, HEADS, head_dim)

    outs=[]

    for h in range(HEADS):
        score=q[:,h]@k[:,h].T
        score/=np.sqrt(head_dim)

        mask=np.triu(np.ones_like(score),1).astype(bool)
        score[mask]=-1e30

        score=np.exp(score-score.max(axis=1,keepdims=True))
        score/=score.sum(axis=1,keepdims=True)

        outs.append(score@v[:,h])

    return np.concatenate(outs,axis=1)


def numpy_block():
    T=Timer()

    ln1=T.measure("LayerNorm1",
        lambda:layernorm(x,weights["ln1_weight"],weights["ln1_bias"]))

    q=T.measure("Linear Q",
        lambda:linear(ln1,weights["Wq"],weights["bq"]))

    k=T.measure("Linear K",
        lambda:linear(ln1,weights["Wk"],weights["bk"]))

    v=T.measure("Linear V",
        lambda:linear(ln1,weights["Wv"],weights["bv"]))

    att=T.measure("Attention",
        lambda:attention(q,k,v))

    proj=T.measure("Projection",
        lambda:linear(att,weights["Wo"],weights["bo"]))

    res1=T.measure("Residual1",
        lambda:x+proj)

    ln2=T.measure("LayerNorm2",
        lambda:layernorm(res1,weights["ln2_weight"],weights["ln2_bias"]))

    ff1=T.measure("FF1",
        lambda:linear(ln2,weights["W1"],weights["b1"]))

    ff1=T.measure("GELU",
        lambda:gelu(ff1))

    ff2=T.measure("FF2",
        lambda:linear(ff1,weights["W2"],weights["b2"]))

    out=T.measure("Residual2",
        lambda:res1+ff2)

    T.report("NumPy")
    return out

def torch_block():

    T=Timer()

    tx=torch.from_numpy(x)

    tw={k:torch.from_numpy(v) for k,v in weights.items()}

    ln1=T.measure("LayerNorm1",
        lambda:(tx-tx.mean(1,keepdim=True))/torch.sqrt(tx.var(1,unbiased=False,keepdim=True)+EPS)*tw["ln1_weight"]+tw["ln1_bias"])

    q=T.measure("Linear Q",lambda:ln1@tw["Wq"]+tw["bq"])
    k=T.measure("Linear K",lambda:ln1@tw["Wk"]+tw["bk"])
    v=T.measure("Linear V",lambda:ln1@tw["Wv"]+tw["bv"])

    def torch_attention():
        hd=EMB//HEADS
        qq=q.reshape(SEQ,HEADS,hd).transpose(0,1)
        kk=k.reshape(SEQ,HEADS,hd).transpose(0,1)
        vv=v.reshape(SEQ,HEADS,hd).transpose(0,1)
        s=(qq@kk.transpose(1,2))/np.sqrt(hd)
        mask=torch.triu(torch.ones_like(s),1).bool()
        s.masked_fill_(mask,-1e30)
        s=torch.softmax(s,-1)
        o=s@vv
        return o.transpose(0,1).reshape(SEQ,EMB)

    att=T.measure("Attention",torch_attention)
    proj=T.measure("Projection",lambda:att@tw["Wo"]+tw["bo"])
    res1=T.measure("Residual1",lambda:tx+proj)

    ln2=T.measure("LayerNorm2",
        lambda:(res1-res1.mean(1,keepdim=True))/torch.sqrt(res1.var(1,unbiased=False,keepdim=True)+EPS)*tw["ln2_weight"]+tw["ln2_bias"])

    ff1=T.measure("FF1",lambda:ln2@tw["W1"]+tw["b1"])
    ff1=T.measure("GELU",lambda:torch.nn.functional.gelu(ff1,approximate="tanh"))
    ff2=T.measure("FF2",lambda:ff1@tw["W2"]+tw["b2"])
    out=T.measure("Residual2",lambda:res1+ff2)

    T.report("PyTorch")
    return out.numpy()

lib = ctypes.CDLL("libattention.so")
lib.layernorm.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # input
    ctypes.POINTER(ctypes.c_float),   # gamma
    ctypes.POINTER(ctypes.c_float),   # beta
    ctypes.POINTER(ctypes.c_float),   # output
    ctypes.c_size_t,                  # seq_len
    ctypes.c_size_t,                  # embed_dim
    ctypes.c_float                    # eps
]

lib.layernorm.restype = None

lib.linear.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # x
    ctypes.POINTER(ctypes.c_float),   # W
    ctypes.POINTER(ctypes.c_float),   # bias
    ctypes.POINTER(ctypes.c_float),   # output
    ctypes.c_size_t,                  # rows
    ctypes.c_size_t,                  # input_features
    ctypes.c_size_t,                  # output_features
    ctypes.c_bool                     # transpose
]

lib.linear.restype = None

lib.attention.argtypes = [
    ctypes.POINTER(ctypes.c_float),   # Q
    ctypes.POINTER(ctypes.c_float),   # K
    ctypes.POINTER(ctypes.c_float),   # V
    ctypes.POINTER(ctypes.c_float),   # out
    ctypes.POINTER(ctypes.c_float),   # temp
    ctypes.c_size_t,                  # seq
    ctypes.c_size_t,                  # emb
    ctypes.c_size_t,                  # heads
    ctypes.c_size_t,                  # rs
    ctypes.c_size_t                   # cs
]

lib.attention.restype = None

lib.gelu.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t
]

lib.gelu.restype = None

lib.add.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t
]

lib.add.restype = None


def c_block():
    print("\n================ C =================")
    T= Timer()
    start = time.perf_counter()

    lib.layernorm(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ln_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        SEQ,
        EMB,
        ctypes.c_float(1e-5)
    )

    end = time.perf_counter()

    print(f"LayerNorm : {(end-start)*1000:.3f} ms")
    start = time.perf_counter()

    lib.linear(
        ln_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Wq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        SEQ,
        EMB,
        EMB,
        False
    )

    end = time.perf_counter()

    print(f"Linear Q : {(end-start)*1000:.3f} ms")

    start = time.perf_counter()

    lib.attention(
        Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        K.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        attn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        temp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        SEQ,
        EMB,
        HEADS,
        EMB,
        1
    )

    end = time.perf_counter()

    print(f"Attention : {(end-start)*1000:.3f} ms")

    ff = np.random.randn(SEQ, FF).astype(np.float32)

    start = time.perf_counter()

    lib.gelu(
        ff.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ff.size
    )

    end = time.perf_counter()

    print(f"GELU : {(end-start)*1000:.3f} ms")

    out = np.zeros_like(x)

    start = time.perf_counter()

    lib.add(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.size
    )

    end = time.perf_counter()

    print(f"Add : {(end-start)*1000:.3f} ms")

    return np.zeros((SEQ,EMB),dtype=np.float32)


if __name__=="__main__":

    c_out=c_block()

    np_out=numpy_block()

    torch_out=torch_block()

    print("\nComparison")
    print("="*60)

    print("NumPy vs Torch")
    print("Max Error :",np.max(np.abs(np_out-torch_out)))
    print("Mean Error:",np.mean(np.abs(np_out-torch_out)))

    print("\nC vs NumPy")
    print("Max Error :",np.max(np.abs(c_out-np_out)))

    print("\nC vs Torch")
    print("Max Error :",np.max(np.abs(c_out-torch_out)))
