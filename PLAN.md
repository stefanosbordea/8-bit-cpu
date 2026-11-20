# PLAN: working plan and state (orchestrator's checklist)

Status marks: [ ] open, [~] running, [x] done. Details and numbers are in LOG.md and DECISIONS.md; results in REPORT.md.

## Phase 0: setup
- [x] Project folder, git identity, rules (CLAUDE.md), interface contracts (docs/INTERFACES.md)
- [x] Virtualenv on the external SSD (Python 3.12, torch, numpy, sentencepiece, matplotlib, pytest); sandbox and network checked
- [x] Core modules: canon.py, alpha.py, config.py, api.py

## Phase 1: research
- [x] Seven topics, each with sources: `research/01..09`, summary in `RESEARCH.md`, every source with licence in `SOURCES.md`

## Phase 2: foundation
- [x] Corpus with licences and overlap removal (`data/corpus`, `data/csn`, `data/csn2`), near-copy rule (D-data-8)
- [x] Benchmarks in the sandbox: HumanEval, MBPP, custom 200 (practice and test halves, near and far families)
- [x] Tokenizers 256 and 512, float model and trainer, LS assembler and cycle-exact simulator

## Phase 3: integer design
- [x] `docs/SPEC.md` v0.6: int8 layers, power-of-two scales, 24-bit accumulator bounds, tables, attention without division, sampling
- [x] Level L1 instruction set, program generator, cost model counted on the generated program
- [x] Hardware changes with speed-up and resource estimates (`docs/HARDWARE_CHANGES.md`)

## Phase 4: training
- [x] Final training text (self-contained human functions + simple generated functions), second validation text
- [x] Baselines: stories260K, n-gram, five public models through the same harness
- [x] Shape search, 41 runs of 600 steps, budget counted on the real memory image -> Tier A shape (D-shape-2)
- [x] Earlier Tier A model: float `runs/f/A2` (10,000 steps), int8 `runs/qat/A2q`
- [x] Final Tier A model: long run `runs/f/A3` (30,000 steps), stream-range fine-tune `runs/f/A3r`, int8 `runs/qat/A3rq` (D-qat-7)
- [x] Tier B float and int8 (`runs/f/B1`, `runs/qat/B1q`, 3,000 steps)
- [x] Tier A int8 from finer logit codes and a second QAT stage (`runs/qat/A3ru2`); Tier B trained on for 6,000 steps (`runs/f/B2`, `runs/qat/B2q`); D-train-4
- [ ] Not run: longer re-training of the three best shapes of the search; a window of 128 at full training length

## Phase 5: integer reference, decoding, editing
- [x] Bit-exact integer reference; equal to the QAT forward pass and to the generated LS program
- [x] Syntax filter (table driven), measured on and off; mode without docstrings for requests
- [x] Four weight-editing sessions, `EDITS.md`, held-out check (no gain: D-edit-2)
- [x] Benchmark tables with the integer reference for Tier A and Tier B

## Phase 6: speed
- [x] Cycles per token at context 16/64/256, breakdown, tokens per second, stream against classic MAC
- [x] Techniques measured: shared key/value heads, vocabulary, window, early exit, greedy against sampling
- [x] Tier B: cost model, tokens per second, what the CPU needs; Colab notebook prepared (not run)

## Phase 7: programs on request (added by the owner during the run)
- [x] Request suite: 401 requests with tests, 400 with a verified program text (human or local Qwen model), sources per request
- [x] Request resolver on the PC (names, other names, phrases, typing errors)
- [x] Board model trained on request rows, kept by exact recall; exact integer repair (D-board-2, D-board-3)
- [x] Every request checked on the simulated CPU (`tools/board_verify.py`)
- [x] Board model for the suite of 401 (`runs/f/C2`, `runs/qat/C2b`, repair `runs/qat/C2bp`, check on the simulated CPU): 400 of 401

## Phase 8: export and report
- [x] Memory images (hex, coe), VHDL package, memory map, program outline, test vectors, UART format and PC scripts
- [x] Final exports of the board model, `artifacts/`, REPORT.md regenerated from the final files

## Phase 9: alignment with the CPU (the CPU itself is built by its owners, not here)
- [x] The eight programs of the finished level-0 CPU in the model's simulator: 32 of 32 runs agree (`docs/L0_CROSSCHECK.md`, `tools/l0_crosscheck.py`)
- [x] What the model program uses and what the CPU still needs, with the numbers for the documents and their sources (`docs/CPU_ALIGNMENT.md`, `tools/cpu_alignment.py`)
- [x] Level-1 conformance programs with expected cycles, OUT byte, UART bytes and final state (`export/conformance/`, `tools/l1_conformance.py`)
- [ ] By the CPU's owners: the level-1 VHDL; then the model program in VHDL simulation against `export/board_uart/`, then the board
