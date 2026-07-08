# Team A — Build one GPT-2 "block" in C

**Team:** Muhammad Ahmad (lead) · M. Waize · Ibrahim Butt · Hamza Abbas

## What you are building (the big picture)

GPT-2 is a text-generating model. Inside, it is just 12 identical "blocks"
stacked on top of each other. One block = a fixed recipe of math steps on
a 2D array of floats.

Your team builds **one complete block in C**. Input: a matrix of floats.
Output: a matrix of floats. If your output matches Hugging Face's output
for the same input, you have re-built a real piece of GPT-2. That's the task.

The recipe (in order):

```
input x  (matrix of size 1024 x 768  =  1024 tokens, 768 numbers each)
  1. layernorm(x)                 -> normalize every row
  2. linear: Q = x·Wq+bq, K = x·Wk+bk, V = x·Wv+bv   (matrix multiplies)
  3. attention(Q,K,V)             -> the "which words look at which words" step
  4. linear again (projection), then ADD the original x back ("residual")
  5. layernorm again
  6. linear (768 -> 3072), GELU, linear (3072 -> 768), ADD residual again
output: matrix 1024 x 768
```

You already know how to multiply matrices with BLIS from Task 1.
Every "linear" above is exactly that.

## Who does what

| Member | Your C function(s) | In plain words |
|---|---|---|
| Ibrahim | `layernorm`, `add` | For each row: subtract the mean, divide by the standard deviation, then multiply by weight `w` and add bias `b`. `add` = add two matrices element by element. |
| M. Waize | `linear` | `out = x·W + b` using BLIS sgemm (your Task 1 skill). Careful: sometimes W must be transposed. Everyone else calls YOUR function, so make it clean. |
| Hamza | `softmax`, `gelu` | softmax: per row, `exp(v-max)/sum` (subtract the max first or you get inf!). gelu: one formula applied to every number — we give you the formula. |
| Ahmad (lead) | `attention` + putting it all together | Split Q,K,V into 12 "heads", compute `scores = Q·K^T / sqrt(64)`, hide the future (causal mask), softmax, multiply by V, merge heads back. Then chain everyone's functions into the full block. |

## How we check your work

We give you a Python script `dump_gpt2.py`. It saves to files:
the real GPT-2 weights for block 0, a test input, and the CORRECT
output of every step (after layernorm, after Q/K/V, after attention...).

Your C program loads those files, runs your functions, and compares
step by step. First step that doesn't match = that's where your bug is.
This makes debugging much easier — use it.

## Checklist

- [ ] Week 1 (Fri 17): each member's function passes its own answer-key test
- [ ] Week 2 (Fri 24): the full chained block matches Hugging Face (0.0001)
- [ ] README: how to build/run + which step was hardest and why
- [ ] Bonus: time your block. What % of time is matrix multiply? (Expect ~90%)

## Warning: the hard part

The hard part is **attention's head splitting** — turning a 1024×768 matrix
into 12 separate 1024×64 pieces and back, using the right index arithmetic.
Everyone gets the indexing wrong the first time. Budget 2–3 days for it.
Draw the memory layout on paper BEFORE writing code.

## Help

- Watch: Karpathy, "Let's build GPT" (YouTube) — watch the attention part twice.
- The answer-key files make every bug findable. Match step by step, in order.
