# Aibee Internship

This repo is the home of the whole internship. Every task gets its own
folder (`task2/`, `task3/`, ...) with its own brief and deadlines.
The rules on this page are permanent — they apply to every task.

## Teams

| Team | Members (lead first) | Focus |
|---|---|---|
| A | Muhammad Ahmad, M. Waize, Ibrahim Butt, Hamza Abbas | kernels / low-level |
| B | Hassnain Javaid, Ayesha Noor, Toheed Ali, Zunair Khalid | tokenizer |
| C | Hanan CH, Shajia Siddiqa, Kiren Saleem, Jazim Iqbal | LLaMA kernels |
| D | Rana Hammad, M Moiz Ur Rehman, Abdullah Nawaz | file formats / GGUF |

## The one rule that matters: the "answer key" test

For every piece of C code you write, there is a Python script that
produces the CORRECT answer (the **oracle** — the answer key). Your C
program must produce the same answer (within 0.0001 for float math).

- Your code is NOT done when it compiles, or when it "looks right".
- It is done ONLY when it matches the answer key.

This is how professional engine developers work. It applies to every
task in this internship.

## Words you will see everywhere

| Word | Meaning |
|---|---|
| tensor | just a 2D array of floats, e.g. `float x[1024][768]` |
| kernel | a C function that does one math job on tensors (nothing to do with OS kernels) |
| weights | the numbers a trained model has learned; we load them from files |
| oracle test | run Python answer key + your C code on the same input, compare outputs |
| HF / Hugging Face | the website + Python library where models live |

## How we work in this repo

1. Nobody pushes to `main`. Work on a branch named
   `team-x/yourname/what-it-is` (e.g. `team-b/toheed/pre-tokenizer`)
   and open a **pull request**. Your team lead reviews, then it gets merged.
2. Only touch your own team's folder.
3. Everyone commits their OWN part — commit history is part of your grade.
4. Being stuck 2–3 days on your task's "hard part" is normal and budgeted.
   Being stuck silently is not. Ask your lead first, then ask me.
5. AI tools (ChatGPT/Claude) are allowed for LEARNING ("explain this to me"),
   discouraged for writing your code — the answer-key test and demo
   questions will expose code you don't understand.
6. Every deliverable includes a README: what you built, how to run it,
   your hardest bug.

## Tasks

| Task | What | Where | Status |
|---|---|---|---|
| 1 | BLIS in C, FLOPS across matrix sizes | (before this repo) | done |
| 2 | One real engine component per team | [`task2/`](task2/) | **active — final Fri 24 Jul** |
