Everything runs from the project folder with the virtualenv on the external SSD. Generated code is executed only by
the sandbox. The commands below repeat the checks, one benchmark row, the speed documents and this report.

```
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
.venv/bin/python -m pytest -q tests
.venv/bin/python -m lsllm.bench.run --completer ref --bench custom --split test
.venv/bin/python -m lsllm.bench.run --completer intf:MODEL_DIR --bench custom --split test --max-new-tokens 96
.venv/bin/python -m lsllm.intref.overflow MODEL_DIR --data DATA_DIR
.venv/bin/python -m lsllm.export MODEL_DIR export/NAME
PYTHONPATH=. .venv/bin/python tools/speed_report.py MODEL_DIR
PYTHONPATH=. .venv/bin/python tools/pareto.py runs/p,runs/q docs/SHAPE_SEARCH.md docs/pareto.png
PYTHONPATH=. .venv/bin/python tools/report_tables.py runs/final.json
PYTHONPATH=. .venv/bin/python tools/report.py runs/final.json
.venv/bin/python tools/uart_receive.py --port /dev/tty.usbserial-XXXX --tok DATA_DIR/tok.model
```

The board model (programs on request), from the text list to the check on the simulated CPU:

```
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
.venv/bin/python -m quick.asks_texts
.venv/bin/python -m quick.asks_bundle
.venv/bin/python -m quick.asks_all_texts data/asks/texts_NEW.jsonl data/asks/texts_C2.jsonl
.venv/bin/python -m lsllm.train.run --out runs/f/NAME --data data/tok/f512p --init-from FLOAT_CKPT --steps 3000 --batch-size 32 --lr 6e-4 --prompt-weight 0.1 --recall data/asks/texts_NEW.jsonl
.venv/bin/python -m lsllm.qat.train --out runs/qat/NAME --float runs/f/NAME/ckpt.pt --data data/tok/f512p --steps 5000 --kd-weight 0.5 --prompt-weight 0.1 --device mps --recall data/asks/texts_NEW.jsonl
.venv/bin/python -m lsllm.intref.polish runs/qat/NAME runs/qat/NAMEp --recall data/asks/texts_NEW.jsonl
.venv/bin/python -m lsllm.asks.run --completer intp:runs/qat/NAMEp --json results/asks_NAMEp_int.json
PYTHONPATH=. .venv/bin/python tools/board_verify.py runs/qat/NAMEp --json results/board_NAMEp.json
PYTHONPATH=. .venv/bin/python tools/board_ask.py runs/qat/NAMEp --list
PYTHONPATH=. .venv/bin/python tools/board_ask.py runs/qat/NAMEp "make me a bubble sort program" "binary search trees" --simulate
```

`MODEL_DIR` and `DATA_DIR` are the folders named in `runs/final.json` (the integer model and its tokenizer folder).
Training commands with all arguments are in `runs/jobs2.txt` and in the `result.json` of every run; longer runs for a
GPU are prepared in `notebooks/train_on_colab.ipynb` (`tools/make_colab_bundle.py`).
