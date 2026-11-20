"""The laptop side of the UART protocol against the board emulator on a pseudo-terminal (the simulator, not hardware)."""
import json
import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = os.path.join(ROOT, "artifacts", "board")
serial = pytest.importorskip("serial")


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, json.load(open(os.path.join(MODEL, "model.json")))["tokenizer"], "tok.model")),
                    reason="the tokenizer folder named in model.json is not on this machine")
def test_a_request_over_the_serial_device_gives_the_stored_program():
    from lsllm.ls.uart import ask, open_port
    from lsllm.tok import Tok
    spec = json.load(open(os.path.join(MODEL, "model.json")))
    tok = Tok(os.path.join(ROOT, spec["tokenizer"], "tok.model"))
    emulator = subprocess.Popen([sys.executable, "-W", "ignore", "tools/board_emulator.py", "artifacts/board", "--speed", "0"], cwd=ROOT,
                                stdout=subprocess.PIPE, text=True, env=dict(os.environ, PYTHONPATH="."))
    try:
        port = open_port(emulator.stdout.readline().strip())
        texts = {row["name"]: row["text"] for row in map(json.loads, open(os.path.join(MODEL, "texts.jsonl")))}
        for name in ("hello_world", "is_even"):
            ids = tok.encode("def %s(" % name, bos=True)
            answer, seconds = ask(port, ids, timeout=120.0)
            assert tok.decode(ids + answer[:-1]) == texts[name] and seconds > 0
    finally:
        emulator.kill()
        time.sleep(0.1)
