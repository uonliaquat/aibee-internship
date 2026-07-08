# Aibee Internship — Task 2 (read this first)

You are in one of 4 teams. Each team builds ONE small piece of a real
inference engine (the kind of program that runs ChatGPT-style models).
Your piece is small, but it is real — good work here goes into a real
open-source project later.

## The one rule that matters: the "answer key" test

For every piece of C code you write, there is a Python script that
produces the CORRECT answer (we call it the **oracle** — think of it as
the answer key). Your C program must produce the same answer.

- Numbers must match within 0.0001 (tiny floating-point differences are OK).
- Your code is NOT done when it compiles. It is NOT done when it "looks right".
- It is done ONLY when it matches the answer key.

This is exactly how professional engine developers work. Learn it now.

## Words you will see everywhere

| Word | Meaning |
|---|---|
| tensor | just a 2D array of floats, e.g. `float x[1024][768]` |
| kernel | a C function that does one math job on tensors (nothing to do with OS kernels) |
| weights | the numbers a trained model has learned; we load them from files |
| oracle test | run Python answer key + your C code on the same input, compare outputs |
| HF / Hugging Face | the website + Python library where models live |

## Schedule

| Date | What happens |
|---|---|
| Fri 10 Jul | Each team presents its plan (10 min): who does what |
| **Fri 17 Jul** | **Checkpoint: every member's own part passes its answer-key test** |
| **Fri 24 Jul** | **Final: team's full deliverable works + 15-min demo + README** |

## Rules

1. One GitHub repo per team. Everyone commits their OWN part (we check).
2. Being stuck for 2–3 days on the "hard part" of your task is NORMAL and expected.
   Being stuck silently is not. Ask your team lead first, then ask me.
3. AI tools (ChatGPT/Claude) are allowed for LEARNING (explain me this),
   discouraged for writing your code — the answer-key test will expose
   code you don't understand, and the demo questions definitely will.
4. Your README must answer: what did you build, how do we run it,
   what was the hardest bug.

Your team's own file (TEAM_A / B / C / D) has your full task.
