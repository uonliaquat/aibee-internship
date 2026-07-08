# Team B — Build the GPT-2 tokenizer in C

**Team:** Ayesha Noor · Hassnain Javaid (lead) · Toheed Ali · Zunair Khalid

## What you are building (the big picture)

Models don't read text. They read numbers. A **tokenizer** converts text
to numbers and back:

```
"Hello world"  ->  [15496, 995]      (encode)
[15496, 995]   ->  "Hello world"     (decode)
```

Every id is an entry in a dictionary of 50,257 known word-pieces.
Long/rare words get split into pieces: `"unbelievable"` might become
`["un", "believ", "able"]` -> 3 ids.

Your team builds GPT-2's tokenizer (called **BPE**) in C. It must produce
EXACTLY the same ids as Hugging Face's tokenizer — not close, identical.
This is the strictest test of all four teams, and also the fairest:
ids either match or they don't.

How BPE encoding works, in plain words:

1. Split the text into chunks (words, spaces, punctuation) using GPT-2's
   special splitting rules.
2. Turn each chunk into bytes; start with one token per byte.
3. Look at the file `merges.txt` — it is a ranked list of pairs that are
   allowed to merge (e.g. `h e` -> `he`). Repeatedly merge the
   BEST-ranked pair in your chunk until no more merges apply.
4. Look up each final piece in `vocab.json` to get its id.

## Who does what

| Member | Your part | In plain words |
|---|---|---|
| Hassnain | Load the dictionary files | Parse `vocab.json` (piece -> id) and `merges.txt` (pair -> rank). Build fast lookup tables (hash map) in C. Everyone else uses your tables. |
| Toheed | The text splitter (step 1) | Re-create GPT-2's splitting rules in plain C (no regex library): a word with its leading space stays together, numbers split apart, punctuation splits, etc. **This is the team's hard part.** |
| Ayesha | The merge loop (steps 2–3) + putting it together | The core algorithm: repeatedly find the best-ranked pair and merge it. Then wire everything into one `encode()` function. |
| Zunair | Decode + the test harness | ids -> pieces -> bytes -> text (careful with UTF-8/emoji). Also build the test: a Python script that tokenizes 100 sentences with HF and saves the ids; a C program that checks your team matches all 100. |

## How we check your work

Zunair's test set: 100 sentences including normal English, numbers,
punctuation, weird spacing, and emoji. Your `encode()` must match
Hugging Face on ALL of them, and `decode(encode(text)) == text`.

## Checklist

- [ ] Week 1 (Fri 17): dictionary files load; simple sentences like
      "Hello world" encode correctly
- [ ] Week 2 (Fri 24): all 100 sentences match exactly + round-trip works
- [ ] README: how to run + which sentence broke your code the longest

## Warning: the hard part

The **splitter** (Toheed's part). GPT-2 splits text with a complicated
rule, and one wrong split changes all the ids after it. Test with:
`"Hello, world!"`, `"  two   spaces"`, `"don't"`, `"123abc"`, an emoji.
Being stuck 2–3 days here is normal. Compare your splits against HF's
(`tokenizer.tokenize(text)` shows you the pieces).

## Help

- Watch: Karpathy, "Let's build the GPT Tokenizer" (YouTube) — it is
  literally this task in Python. Watch it FIRST, then write C.
