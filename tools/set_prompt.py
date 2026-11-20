#!/usr/bin/env python3
"""Put another prompt into the memory image of an export (`python -m lsllm.export`).

    python tools/set_prompt.py OUT_DIR/mem 'def area(w, h):\n    return'

Run it from the root of the repository. The text is canonical Python; \n in the argument stands for a line break.
It is encoded with the tokenizer named in MEM_DIR/map.json (BOS token in front) and written into the image with
lsllm.ls.gen.set_prompt at the addresses of map.json; then data.hex and data.coe are written again. Only the
prompt bytes change. The test vectors of the export (OUT_DIR/vectors) still describe the old prompt: for the
expected output of the new prompt run the export again with --prompt. Only for an export with one prompt in the
image: an export with a prompt table (--prompts, --asks) or with prompts over the UART is refused.
"""
import argparse
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, ".")
from lsllm.export import files
from lsllm.ls import gen
from lsllm.tok import Tok


def set_prompt(mem_dir, text):
    """Rewrite the prompt bytes of mem_dir/data.hex and data.coe. Returns the token ids written."""
    with open(os.path.join(mem_dir, "map.json")) as f:
        spec = json.load(f)
    if spec.get("form", "single") != "single":
        raise ValueError("this export takes its prompts from %s, not from the prompt bytes of the image; "
                         "export again with other prompts" % {"table": "a table chosen by the switches"}.get(spec["form"], "the UART"))
    b = types.SimpleNamespace(mem=spec["regions"])              # all that gen.set_prompt reads of a Build
    ids = Tok(os.path.join(spec["tokenizer"], "tok.model")).encode(text, bos=True, eos=False)
    image = files.read_hex(os.path.join(mem_dir, "data.hex")).astype(np.uint8).view(np.int8)
    address, size = b.mem["prompt.ids"]
    image[address:address + size] = 0                           # the bytes of a longer old prompt
    gen.set_prompt(image, b, ids)
    files.write_memory(mem_dir, "data", image, 2)
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Put another prompt into mem/data.hex and mem/data.coe of an export.")
    ap.add_argument("mem_dir", help="the mem directory of an export (with map.json, data.hex, data.coe)")
    ap.add_argument("prompt", help=r"canonical Python; \n stands for a line break")
    args = ap.parse_args()
    ids = set_prompt(args.mem_dir, args.prompt.replace("\\n", "\n"))
    print("%d prompt tokens written to %s: %s" % (len(ids), args.mem_dir, " ".join(map(str, ids))))
    print("vectors/ of this export still describes the old prompt; run the export again with --prompt for the expected output")
