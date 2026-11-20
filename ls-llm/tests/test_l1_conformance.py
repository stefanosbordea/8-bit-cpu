"""The stored level-1 conformance programs (export/conformance) still give their stored results in the simulator."""
import json
import os

import numpy as np
import pytest

from lsllm.ls.iss import L1

ROOT = os.path.join(os.path.dirname(__file__), "..", "export", "conformance")
NAMES = sorted(name for name in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, name)))


@pytest.mark.parametrize("name", NAMES)
def test_program_gives_its_stored_result(name):
    folder = os.path.join(ROOT, name)
    expected = json.load(open(os.path.join(folder, "expected.json")))
    code = [int(line, 16) for line in open(os.path.join(folder, "program.hex"))]
    image = np.array([int(line, 16) for line in open(os.path.join(folder, "data.hex"))], dtype=np.uint8).astype(np.int8)
    cpu = L1(code, image, sw=expected["switches"], rx=[int(v, 16) for v in expected["rx_bytes"]])
    cpu.run(limit=100_000)
    assert cpu.halted and len(code) == expected["program_words"] and len(image) == expected["data_bytes"]
    assert cpu.cycles == expected["cycles"] and cpu.outr == int(expected["out"], 16)
    assert list(cpu.tx) == [int(v, 16) for v in expected["tx_bytes"]]
    assert cpu.r == [int(v, 16) for v in expected["registers"]] and cpu.p == [int(v, 16) for v in expected["pointers"]]
    assert (cpu.len, cpu.acc, cpu.z, cpu.n) == (expected["len"], expected["acc"], expected["zero_flag"], expected["negative_flag"])


def test_all_ten_are_there():
    assert len(NAMES) == 10
