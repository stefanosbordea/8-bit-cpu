# Final models

Copied by `tools/collect_artifacts.py` from the run folders named in `runs/final.json`.

| folder | float checkpoint from | integer model from | files |
|---|---|---|---|
| `artifacts/tier_a` | `runs/f/A3r/ckpt.pt` | `runs/qat/A3ru2` | float.pt, float_result.json, model.json, model.npz, result.json |
| `artifacts/tier_a_earlier` | `runs/f/A2/ckpt.pt` | `runs/qat/A2q` | float.pt, float_result.json, model.json, model.npz, result.json |
| `artifacts/tier_b` | `runs/f/B2/ckpt.pt` | `runs/qat/B2q` | float.pt, float_result.json, model.json, model.npz, result.json |
| `artifacts/board` | `runs/f/C2/ckpt.pt` | `runs/qat/C2bp` | float.pt, float_result.json, model.json, model.npz, qat_result.json, polish.json, polish.jsonl, texts.jsonl |

`artifacts/tokenizer/` holds the SentencePiece model of all of them (`tok.model`, `tok.vocab`; the same file in `data/tok/f512c2`, `data/tok/f512p`, `data/tok/f512v2`: checked).
Each `model.json` names its tokenizer folder under `data/tok/`, which is not tracked: on a fresh copy of the repository, put the two
files there (for example `data/tok/f512p/`) before running a tool on a model.

The board model is `artifacts/board`: `PYTHONPATH=. .venv/bin/python tools/board_ask.py artifacts/board --list`.
