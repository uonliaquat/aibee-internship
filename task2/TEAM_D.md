# Team D — Build a GGUF model-file reader in C

**Team:** Rana Hammad (lead) · M Moiz Ur Rehman · Abdullah Nawaz

## What you are building (the big picture)

When you download a model to run locally (llama.cpp, Ollama), you get one
big file ending in `.gguf`. Inside that single file is EVERYTHING:
the model's settings, its tokenizer, and all its weights (often compressed).

Your team builds a C program that opens a real GGUF file and reads it:

```
./gguf_dump model.gguf                     -> prints all settings + list of tensors
./gguf_dump model.gguf blk.0.attn_q.weight -> prints that tensor's first values
```

A GGUF file has 3 parts, in order:

```
[HEADER]    magic "GGUF", version, how many tensors, how many settings
[METADATA]  key-value pairs: "llama.block_count" = 22, tokenizer data, ...
[TENSORS]   first a table of contents (name, shape, type, offset),
            then the raw bytes of all the weights
```

Extra challenge: weights are often **quantized** (compressed). Type
"Q8_0" means: numbers stored in blocks of 32, each block = one small
scale factor + 32 one-byte integers. Real value = integer × scale.
You will decompress that.

Your team is 3 people (others are 4), so the scope is deliberately
slightly smaller — but your task is the most "real file format
engineering" of all four.

## Who does what

| Member | Your part | In plain words |
|---|---|---|
| Rana (lead) | Header + metadata + the CLI tool | Open the file with `mmap`, check the magic bytes, read version and counts, then read all key-value pairs (each value has a type id: int, float, string, array...). Wire everyone's parts into the final `gguf_dump` program. |
| Moiz | Tensor table + finding the bytes | Read the table of contents (each tensor's name, shape, type, offset). Then compute where each tensor's bytes REALLY start — there are alignment/padding rules (data starts at a multiple of 32). **This is the team's hard part.** |
| Abdullah | Dequantization | Given a tensor's raw Q8_0 bytes, decompress: for each block of 34 bytes = 2-byte scale (float16!) + 32 int8 values -> 32 real floats. Bonus: Q4_0 (two values packed per byte). |

## How we check your work

Python has an official `gguf` library. We give you a script that prints
the same things using that library. Your C output must match it:
same metadata values, same tensor list, same first-20 numbers of any
tensor you dequantize.

Test file: download TinyLlama Q8_0 from Hugging Face (one .gguf, ~1 GB).

## Checklist

- [ ] Week 1 (Fri 17): header + all metadata print correctly
- [ ] Week 2 (Fri 24): any tensor extractable by name; Q8_0 values match Python
- [ ] README: how to run + a drawing (ASCII is fine) of the file layout
- [ ] Bonus: Q4_0 support

## Warning: the hard part

**Alignment** (Moiz's part). The tensor bytes do NOT start right after
the table of contents — there is padding so data starts at a multiple
of 32, and each offset in the table is relative to that padded start,
not to the file start. If your first values look like garbage, you are
reading from the wrong offset. Also: the Q8_0 scale is a 16-bit float
(float16) — you must convert it to float32 yourself (one small function,
we'll give you a hint).

## Help

- Read: the GGUF specification (in the ggml GitHub repo, docs/gguf.md).
  It is long — you only need "File Structure" and the Q8_0 part.
- `xxd model.gguf | head` shows you the first bytes — you should
  literally see the letters G G U F. Start there.
