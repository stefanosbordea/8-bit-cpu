# LS-LLM: project rules (read before doing anything)

A tiny language model that writes Python, trained on a laptop, quantized to int8 and run by the LS 8-bit CPU
(own design, VHDL, Digilent Nexys A7-100T). Owner: Loucas (GitHub `dlukel`). Every number must be defensible in
front of a professor. Root of this project: `/Users/kixem/Desktop/LS Architecture/ls-llm` (quote the path: it has a space).

## Hard rules
1. Honest numbers only. Report zeros. A practice-set score is never called a benchmark score. Never write
   "new", "first" or "best"; at most "not found in our search".
2. Never tune on test sets (HumanEval, MBPP test split, custom-200 test). Tuning uses practice sets only.
3. Model-generated code runs ONLY through `lsllm.sandbox` (OS sandbox + timeout). Never `exec`/`eval`/import it in-process.
4. Loop counters are never named `i`. Use `g` (nested: `h`, `k`, or a descriptive name). Applies to every language.
5. Git: identity is repo-local `dlukel <ldl00@aubmed.ac.cy>` (already set). No AI/Claude attribution anywhere in
   commits. Agents do NOT commit, push, or change git config; the orchestrator commits.
6. Every external paper, model, dataset or code file that is used gets a row in `SOURCES.md` (append one table row:
   `| name | kind | URL | licence | used for |`). Check the licence at the source, do not guess it.
7. Decisions go to `DECISIONS.md` (append a block: `### D-<area>-<n>: title` + Decision / Why / Numbers / Rejected).
   Progress lines go to `LOG.md` (append `- YYYY-MM-DD HH:MM [area] text`). Append with `>>`, never rewrite these files.
8. Shell instructions written for the user (README, report): one combined code block, no comments inside it.

## Machine limits (MacBook Pro M1, 8 GB RAM, disk almost full)
- Disk: about 2 GB free. Run `df -h /` before any download or large write. Never let free space drop below 1.0 GB.
  No single download above 100 MB unless your brief allows it. Delete raw archives after processing.
- One shared virtualenv: `.venv/bin/python` (Python 3.12, torch, numpy, sentencepiece, matplotlib, pytest).
  Do not create other environments. Do not `pip install` anything; if you need a package, say so in your result.
- RAM: at most one heavy Python process per agent, under 1.5 GB. Training runs only through `tools/train_queue`
  rules given in your brief; never start a long training run on your own.
- Foreground commands are capped at 10 minutes. Keep tests and smoke runs short.
- Downloaded archives are untrusted data: extract each into its own directory under `data/raw/<name>/`,
  never execute anything from them, read them with `.venv/bin/python -I`.

## Working together
- Many agents work here in parallel. Edit only the files and folders your brief assigns to you.
  If another area needs a change, report it in your result instead of editing it.
- Write the minimum code that does the job: no speculative options, no unused abstractions, short docstrings.
  Prefer the standard library and numpy. Code must be readable by a second-year student who has to defend it.
- Tests live in `tests/test_<area>_*.py` and must pass with `.venv/bin/python -m pytest -q tests/test_<area>_*.py`.
- State what you verified and what you did not. If something failed, say so with the output.

## Training-data rule added 2026-11-13 19:05 (overrides any earlier brief)
No text written by an AI assistant may enter the training data: no code templates, no docstring sentences, no
examples written by an agent, not even when a program expands them. Reason: Anthropic's usage policy forbids using
Claude outputs to train a model without authorisation (quotes and links in `research/08_datasets_licences.md`,
section 2.5), and the project rule is "only if clearly allowed". Allowed training text: (1) human-written code
under a permissive licence, (2) mechanical transformations of it (canonical formatting, renaming identifiers with
names taken from the corpus), (3) outputs of an open-weight model whose licence clearly allows it (handled by the
orchestrator: Qwen2.5-Coder, Apache-2.0), each verified in the sandbox. AI-written tasks and tests are allowed for
evaluation only.
If your brief is the "synth" area: do NOT write a template generator. Build instead, under `lsllm/data/synth/`,
a pipeline that takes human-written functions from `data/corpus/*.jsonl`, finds the self-contained deterministic
ones, computes assert lines for them by running them in the sandbox on probe inputs (ints, lists, strings chosen
by trying), keeps only functions for which this works, and optionally adds mechanically renamed copies
(identifier names taken from the corpus). Same output files and the same benchmark-overlap exclusions as in the brief.
