# LS-LLM: a language model that writes Python, for the LS 8-bit CPU

A small transformer with int8 weights that runs on the LS CPU (own design, Nexys A7-100T, 100 MHz) with nothing but
streamed multiply-accumulates, constant shifts and 256-entry tables. The model, its key/value cache and all tables
fit 450 KB of block RAM. Every number in `REPORT.md` is read from a file that a run wrote.

| Read this | For |
|---|---|
| `REPORT.md` | results, tables, Pareto chart, cycle breakdown, comparisons |
| `DECISIONS.md` | every choice with its reason and numbers |
| `docs/SPEC.md` | the machine (level L1) and the integer arithmetic, binding for all code |
| `docs/HARDWARE_CHANGES.md` | what the CPU needs, and optional speed-ups with their cost |
| `docs/CPU_ALIGNMENT.md`, `docs/L0_CROSSCHECK.md` | the finished course CPU against the model: its eight programs in the model's simulator, what the model program uses, what the CPU still needs |
| `export/vhdl_reference/`, `docs/VHDL_REFERENCE.md` | the level-1 instruction set as a behavioural VHDL model; under nvc it runs the exported model program and sends the same bytes in the same cycles as the Python simulator, for all 401 requests |
| `export/conformance/` | ten small programs for the level-1 instructions with the results the extended CPU must show |
| `docs/VHDL_CHECK.md` | an earlier check of the first six hand-written VHDL modules (before the CPU was finished) |
| `docs/FILTER.md`, `docs/COST_TABLE.md`, `docs/TIER_B.md` | syntax filter, cycle counts of the generated program, the DDR2 model |
| `EDITS.md` | every weight edit of the editing sessions |
| `RESEARCH.md`, `research/` | what is known, with sources |
| `SOURCES.md` | every external paper, model, dataset and code file, with its licence |
| `export/` | memory images, VHDL package, memory map, program outline, test vectors (`board_uart`, `board_switches`: the board model) |
| `artifacts/` | the final models: float checkpoints, integer models, tokenizer |
| `LOG.md` | what was done and when |

## Layout
`lsllm/` Python package: `canon`, `structure`, `tok` (text and tokens), `data/` (corpus, augmentation, overlap rules),
`model/`, `train/` (float model), `qat/` (quantization-aware training), `intref/` (bit-exact integer reference),
`ls/` (LS assembler, simulator, program generator, cost model), `filter_table` (syntax filter), `bench/` and
`sandbox` (benchmarks), `asks/` (request suite), `edit/` (weight editing), `baselines/`, `export/`.
`tests/` holds the tests of every area, `tools/` the scripts that make the documents, `quick/` the data builders.

## Ask for a program
```
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
PYTHONPATH=. .venv/bin/python tools/board_ask.py artifacts/board --list
PYTHONPATH=. .venv/bin/python tools/board_ask.py artifacts/board "make me a bubble sort program" "binary search trees" --simulate
PYTHONPATH=. .venv/bin/python tools/board_ask.py artifacts/board "heap sort" --port /dev/tty.usbserial-XXXX
PYTHONPATH=. .venv/bin/python tools/ask.py artifacts/board "factorial"
```
Without the board, `tools/board_emulator.py artifacts/board` prints the name of a serial device that answers like the board
(the simulator behind a pseudo-terminal); give that name to `--port`. `tools/board_verify.py artifacts/board --port DEVICE`
sends every request of the suite, checks the answers and measures tokens per second on the PC.

`artifacts/board` is the board model (int8, 350,880 parameters). The first line lists the requests it knows, the second
runs the generated LS program in the cycle-exact simulator, the third talks to the board over the USB-UART (needs the
`RX` instruction and `pyserial`), the fourth uses the integer reference on the laptop. The PC turns the sentence into a
name (`def bubble_sort(`); the model writes the rest. It writes the programs of the request suite (`data/asks/`, sources
in `docs/ASKS_SOURCES.md`); a request outside the suite gets text that looks like Python and is rarely right
(`REPORT.md`, section 3).

## Check and reproduce
```
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
.venv/bin/python -m pytest -q tests
.venv/bin/python -m lsllm.bench.run --completer ref --bench custom --split test
.venv/bin/python -m lsllm.asks.run --completer intp:artifacts/board
PYTHONPATH=. .venv/bin/python tools/board_verify.py artifacts/board
.venv/bin/python -m lsllm.export artifacts/board export/NAME --uart-prompt --no-docstring
PYTHONPATH=. .venv/bin/python tools/report_tables.py runs/final.json
PYTHONPATH=. .venv/bin/python tools/report.py runs/final.json
```
The virtualenv lives on the external SSD (`/Volumes/CARMIX-WORK/ls-llm-work/venv`, linked as `.venv`); plug the
SSD in before running anything. Generated code is only ever executed inside `lsllm.sandbox`.
