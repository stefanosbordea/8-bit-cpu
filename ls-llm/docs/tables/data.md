All sources with their licences are in `SOURCES.md`; the decisions are D-corpus-1 to 6 and D-data-1 to 8 in `DECISIONS.md`.

**Human-written code.** 152 public repositories passed the licence gate (MIT 88, BSD-3-Clause 37, Apache-2.0 18,
BSD-2-Clause 7, PSF-2.0 1, Unlicense 1; 43 others rejected, 31 of them by the gate): 27 collections of algorithms and
exercises, 24 utility libraries, the CPython standard library and 100 libraries and tools. Tarballs were read in memory,
nothing was unpacked or executed. After canonical formatting (`ast.unparse`: 4-space indentation, one statement per line,
no comments) and removal of duplicates the corpus has 161,755 functions and methods (62.49 MB). The part of
CodeSearchNet with permissive licences adds 18,126 plain functions from 989 repositories and, from its other twelve
shards, 69,781 more from 6,719 repositories.

**Overlap with the benchmarks.** A function is removed from all training data if its name equals the entry point of
a benchmark task, or its code is the reference solution of a task up to renaming (`lsllm/alpha.py`), or its docstring
equals a task prompt; the rule covers HumanEval, all of MBPP and both halves of the custom set (1,457 names, 1,560 code
fingerprints). In the corpus it removed 3,157 functions by name and 315 by code. The same rule is applied to every
generated function and again to every augmented copy.

**Generated code.** The only generator used is Qwen2.5-Coder-1.5B-Instruct (Apache-2.0, open weights, run locally with
llama.cpp), because training on its output is clearly allowed. Text written by the AI assistant that helped with this
project is not in the training data (`CLAUDE.md`, rule of 2026-11-13). The prompts name beginner topics only. A generated
function is kept if it parses, has a one-line docstring, passes the overlap rule and runs in the sandbox: the functions of
the first runs were kept if they ran on probe inputs and gave the same result twice (their assert lines record what
the function returns, so they prove that it runs, not that it is right); from 2026-11-14 on the generator also writes
three test lines per function and a function is kept only if all of them pass in the sandbox.

**Augmentation** (mechanical transformations of existing functions, D-data-5 and D-data-6): copies with renamed
parameters and variables, and copies in which one number is changed in the docstring and the code together.

**Near-copies of benchmark solutions** (D-data-8). A function is also removed if at least half of its runs of 8 tokens
occur in one benchmark solution, whatever its name: 49 human and 54 generated functions went this way (a first long run
that had seen them was stopped and thrown away).

**The final training text of the benchmark models** (D-data-7). The benchmark tasks are self-contained functions, so the
human part is the 18,444 corpus functions that use only their parameters, their own variables, built-ins and a few standard
modules, each once as written and once with renamed names. The other half of the tokens (47.8%) are 1,412 simple
generated functions with their number variants, repeated 16 times and renamed (92,407 renamed copies). That is
`data/tok/f512v2`: 150,366 functions, 14,710,295 tokens; the earlier Tier A model was trained on it. The final Tier A model
and Tier B were trained on `data/tok/f512c2`, which adds a docstring-free copy of the documented functions (14,753 human,
the 1,412 simple ones 16 times) and 61 request programs that pass all overlap rules, 300 times each (8.3% of the tokens):
206,004 functions, 19,188,086 tokens. In the loss, the tokens that a benchmark prompt already gives (signature and
docstring) count one tenth (D-train-2). The mix was chosen on the practice sets (practice-solution loss, D-eval-1), not
on the test sets.

**Training of the final Tier A model** (D-train-3, D-qat-7): 30,000 steps of 64 windows of 256 tokens, then 3,000 steps
with a term that keeps the residual stream inside the range that int8 can hold, then 3,000 steps of quantization-aware
training. The earlier model (10,000 steps) is kept in the tables because the weight-editing sessions and the board model
started from it.

**Tokenizer.** SentencePiece BPE with 512 tokens, trained on the Tier A text. Line structure is carried by tokens:
`<NL>` (new line, same indentation), `<IND>` (new line, one level deeper), `<DED1>` to `<DED5>` (new line, 1 to 5
levels back), `<LF>` and `<BLANK>` (lines inside a docstring). Brackets, quotes, `"""`, `, `, `()` and `):` are always
single tokens, which makes the syntax filter a table lookup per token. 2.16 bytes of source per token on the
validation text (2.09 on the general validation text of the shape search, where 256 tokens gave 1.76: D-tok-1 and
section 7.4).

**Validation text.** For the final models: 408 held-out functions (357 self-contained human functions, 51 generated),
79,082 tokens, 171,155 bytes (`data/tok/val2`); it picks checkpoints and gives every loss in this report except those of
the shape search, which used the earlier general validation text (501 functions, 76,491 tokens).

**The board model has its own training text.** It starts from the weights of the earlier Tier A model and continues on the same kind
of general text (`data/tok/f512p`: the clean set plus one docstring-free copy of every function) with 32 "request rows"
in every step: one program of the request suite each, from the first token of the context, exactly as the board sees a
request (D-board-2). The programs are human-written functions of the licensed corpus and, where the corpus has none that
passes the tests, programs written by the local Qwen model; none was written by an AI assistant or by the authors
(`docs/ASKS_SOURCES.md` lists every one with its source and licence). Many of them have the name or the code of a benchmark
task (`is_prime`, `heap_sort`, ...), which is why the board model is never scored on a benchmark, and why the benchmark
model never sees those.
