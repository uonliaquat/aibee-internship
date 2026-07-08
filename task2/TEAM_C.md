# Team C — Build LLaMA's math functions in C

**Team:** Shajia Siddiqa · Hanan CH (lead) · Kiren Saleem · Jazim Iqbal

> Jazim: create a GitHub account on day 1 — it's required.

## What you are building (the big picture)

Team A is building GPT-2's math. But GPT-2 is from 2019. Modern models
(LLaMA, Qwen, Mistral — the ones people actually use today) replaced
some of GPT-2's math functions with better ones. Your team builds those
**modern replacements** in C, and proves each one correct against a real
model called TinyLlama.

| Old (GPT-2) | New (LLaMA) — YOUR task |
|---|---|
| LayerNorm | **RMSNorm** (simpler: no mean subtraction, no bias) |
| learned position table | **RoPE** (rotates pairs of numbers to encode word position) |
| GELU | **SiLU** (a different smooth curve) |
| plain MLP | **SwiGLU** (two linears multiplied together like a gate) |

This work matters a lot: these exact functions are the next thing the
main zerograd engine needs. Your team is building the future parts.

## Who does what

| Member | Your C function(s) | In plain words |
|---|---|---|
| Hanan | `rmsnorm` | Per row: divide by sqrt(mean of squares + eps), multiply by weight. No mean subtraction, no bias. IMPORTANT: `eps` comes from the model's config file — never type the number yourself. |
| Kiren | `silu`, `mul` | silu: `x * sigmoid(x)` on every number. mul: multiply two matrices element by element (this is the "gate"). |
| Jazim | `linear_nobias` + answer-key script | LLaMA's matrix multiplies have NO bias — wrap BLIS for that. Also own the Python script that dumps TinyLlama's layer-0 weights + correct answers to files for the whole team. |
| Shajia | `rope` + putting it together | RoPE: for each position, rotate pairs of numbers in Q and K by an angle that depends on the position. Then chain silu/mul/linear into the full SwiGLU: `out = linear( silu(linear(x)) * linear(x) )` and match TinyLlama. |

## How we check your work

Jazim's script uses Hugging Face's TinyLlama (a small real LLaMA) to save
inputs and correct outputs for every function. Your C code must match
within 0.0001.

## Checklist

- [ ] Week 1 (Fri 17): `rmsnorm` and `silu` pass their answer-key tests
- [ ] Week 2 (Fri 24): all functions pass + full SwiGLU chain matches TinyLlama
- [ ] README: how to run + explain in 5 lines what RoPE actually does

## Warning: the hard part

**RoPE** (Shajia's part). There are TWO ways to pair up the numbers for
rotation ("rotate-half" vs "interleaved"). Both look correct. Both run
without errors. Only one matches Hugging Face — it uses **rotate-half**
(first half pairs with second half). If your numbers are wrong but look
"randomly shuffled correct", you picked the wrong pairing. 2–3 days of
confusion here is normal and budgeted.

## Help

- Read: "Rotary Embeddings: A Relative Revolution" (EleutherAI blog) —
  pictures of what RoPE does.
- Print your first 4 values after RoPE and compare with HF's — at
  position 0, RoPE should change NOTHING (angle is zero). Great first test.
