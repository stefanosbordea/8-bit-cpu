# Shared interfaces (contracts between the parallel work areas)

Everything here is fixed. If you think a contract is wrong, finish your work against it and report the problem.
Run Python as `.venv/bin/python` from the project root. Package: `lsllm`. Loop counters are never `i`.

## Ownership (edit only your own paths)
| Area | Paths |
|---|---|
| core (orchestrator) | `lsllm/__init__.py`, `lsllm/canon.py`, `lsllm/config.py`, `lsllm/api.py`, `docs/`, top-level `*.md` |
| bench | `lsllm/sandbox.py`, `lsllm/bench/` (not `custom/`), `data/bench/HumanEval*`, `data/bench/mbpp*`, `tests/test_sandbox_*.py`, `tests/test_bench_*.py` |
| tasks | `lsllm/bench/custom/`, `data/bench/custom_*.json*`, `tests/test_custom_*.py` |
| corpus | `lsllm/data/corpus/`, `data/raw/`, `data/corpus/`, `tests/test_corpus_*.py` |
| synth | `lsllm/data/synth/`, `data/synth/`, `tests/test_synth_*.py` |
| tokenizer | `lsllm/structure.py`, `lsllm/tok.py`, `lsllm/data/tokenize_corpus.py`, `data/tok/`, `tests/test_tok_*.py` |
| model | `third_party/llama2c/`, `lsllm/model/`, `lsllm/train/`, `tests/test_model_*.py`, `tests/test_train_*.py` |
| ls-cpu | `lsllm/ls/`, `tests/test_ls_*.py` |
`lsllm/data/__init__.py` may be created empty by whoever needs it first.

## 1. Canonical source (`lsllm.canon.canon`)
`canon(src) -> str | None`: `ast.unparse` formatting (4 spaces, one statement per line, no comments), docstrings
dedented, no blank lines outside strings, inner lines of multi-line docstrings indented to their block, printable
ASCII plus newline only, no trailing newline. Corpus text, benchmark prompts and custom tasks are all canonical.
The only strings that span lines are docstrings.

## 2. Structure markers and tokenizer (tokenizer area)
`lsllm/structure.py` turns canonical text into "marker text" with exactly one marker per line break:
- `<NL>` line break, same indentation level; `<IND>` line break, level + 1; `<DED1>`..`<DED5>` line break, level - k;
- `<LF>` a newline inside a multi-line docstring (the following line is implicitly indented to the block level;
  an empty docstring line is `<LF><LF>`).
- No marker before the first line and none after the last. Leading indentation never appears as spaces.
- `to_markers(text) -> str`, `from_markers(marker_text, level=0) -> str` (must also work as a streaming decoder),
  round trip `from_markers(to_markers(c)) == c` for every canonical text `c`.
`lsllm/tok.py`:
```python
class Tok:
    def __init__(self, model_file): ...          # a SentencePiece model trained on marker text
    vocab_size: int
    bos_id: int; eos_id: int
    pieces: list[str]                            # marker-text string of every token id
    def encode(self, text, bos=True, eos=False) -> list[int]      # text is canonical source
    def decode(self, ids, level=0) -> str        # `level`: indentation level before the first id
    def level_after(self, ids, level=0) -> int   # indentation level after the ids
```
Tokenizers live in `data/tok/<name>/tok.model` with `meta.json`. Token streams are `train.bin` / `val.bin`
(numpy uint16, documents concatenated, each document = BOS, tokens, EOS) plus `meta.json` with at least
`vocab_size`, `train_tokens`, `val_tokens`, `val_bytes` (bytes of canonical text in the validation set,
so that loss can be reported in bits per byte, which is comparable across vocabularies).

## 3. Corpus records (corpus and synth areas)
JSON lines. One record = one document. Fields: `text` (canonical source of one function, with its docstring if it
has one), `name` (function name), `source` (dataset or repository), `licence` (SPDX id), `path`, `kind`
(`function` | `method` | `synthetic`), and for synthetic records `family` and `tests` (list of assert lines).
Files: `data/corpus/train.jsonl`, `data/corpus/val.jsonl`, `data/synth/train.jsonl`, `data/synth/val.jsonl`.

## 4. Benchmark tasks (bench and tasks areas)
```python
@dataclass
class Task:
    task_id: str            # e.g. "HumanEval/0", "mbpp/11", "custom/test/017"
    entry_point: str        # name of the function under test
    prompt: str             # canonical stub: optional imports, `def ...:` and a docstring; the model continues it
    tests: list[str]        # test statements, run one by one after the candidate code is defined
    canonical_solution: str # canonical full source of a correct solution (for harness self-checks only)
    meta: dict              # benchmark-specific extras (e.g. family, stratum)
```
Custom task files `data/bench/custom_test.jsonl` and `data/bench/custom_practice.jsonl` hold the same fields
(`meta` contains `family` and `stratum`). `data/bench/custom_families.json` lists every family with its stratum:
`near` (the synthetic training generator may use the family, but never a function with the same name or an
equivalent body as a task) or `far` (the family is kept out of synthetic training data completely).

## 5. Sandbox (bench area)
```python
from lsllm.sandbox import run_python
r = run_python(code, timeout_s=5.0, memory_mb=512)   # r.status in {"ok", "error", "timeout", "memory"}; r.stdout, r.stderr, r.seconds
```
Runs a fresh isolated interpreter under macOS `sandbox-exec`: no network, no file writes outside a private temp
directory, no reads of the user's home directory except the interpreter itself, no child processes, wall-clock
timeout, memory limit enforced. Model output is executed only through this function.

## 6. Completer (`lsllm.api.Completer`)
`complete(prompt, n, max_new_tokens, temperature, seed) -> list[str]`: `prompt + continuation` must equal the decoded
text of the whole token sequence. Generation stops at EOS, at a line break that returns to indentation level 0, or
after `max_new_tokens`. If the prompt is longer than the context allows, the completer keeps the last tokens that
fit and counts the truncation in `self.truncated`.

## 7. Evaluation and results (bench area)
```python
from lsllm.bench import load_tasks, evaluate
tasks = load_tasks("humaneval" | "mbpp" | "custom", split="test" | "practice")
report = evaluate(completer, tasks, n_samples=1, temperature=0.0, max_new_tokens=128, seed=0)
```
Outcome of one sample: `pass`, `wrong` (ran, an assert failed), `error` (exception), `timeout`, `syntax`
(prompt + completion does not parse). Metrics: `pass@1`, `pass@10` (unbiased estimator of Chen et al. 2021 when
n_samples >= 10), `syntax_ok` (share of samples that parse), `runs_ok` (share with outcome pass or wrong).
Every evaluation appends one JSON line to `results/results.jsonl` (fields: date, model, params, variant, decoding,
filter, edit, benchmark, split, n_tasks, n_samples, metrics, notes). `results/RESULTS.md` is rendered from that
file; rows with `split = practice` go into a separate, clearly labelled table.

## 8. Float model and training (model area)
`lsllm.model.build(cfg: ModelConfig) -> torch.nn.Module` (forward(tokens, targets=None) -> logits, loss).
Checkpoint `ckpt.pt` = `{"config": cfg.to_dict(), "model": state_dict, "step": int, "val_loss": float, "tok": tokenizer dir}`.
`.venv/bin/python -m lsllm.train.run --out runs/<name> --data data/tok/<name> [shape flags] [--steps N] ...`
writes `runs/<name>/ckpt.pt`, `log.txt` and `result.json` (val loss in nats per token and bits per byte, steps,
tokens seen, wall time, device). Only the orchestrator starts long runs.
