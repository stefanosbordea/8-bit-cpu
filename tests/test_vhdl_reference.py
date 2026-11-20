"""The VHDL model of the level-1 instruction set gives the stored results of the conformance programs (needs nvc)."""
import json
import os
import shutil
import subprocess

import pytest

from tools.vhdl_reference import VHDL, simulate

ROOT = os.path.join(os.path.dirname(__file__), "..", "export", "conformance")


@pytest.mark.skipif(not shutil.which("nvc"), reason="nvc is not installed")
def test_conformance_programs_in_the_vhdl_model(tmp_path):
    assert subprocess.run(["nvc", "--std=2008", "-a", VHDL], cwd=tmp_path).returncode == 0
    for name in sorted(n for n in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, n))):
        expected = json.load(open(os.path.join(ROOT, name, "expected.json")))
        got, sent, seconds = simulate(str(tmp_path), os.path.abspath(os.path.join(ROOT, name, "program.hex")), os.path.abspath(os.path.join(ROOT, name, "data.hex")),
                                      [int(v, 16) for v in expected["rx_bytes"]], expected["data_bytes"])
        assert int(got["cycles_low"]) == expected["cycles"] and int(got["out"]) == int(expected["out"], 16), name
        assert sent == [int(v, 16) for v in expected["tx_bytes"]], name
        assert [int(v) for v in got["registers"].split()] == [int(v, 16) for v in expected["registers"]], name
        assert [int(v) for v in got["pointers"].split()] == [int(v, 16) for v in expected["pointers"]], name
        assert (int(got["len"]), int(got["acc"])) == (expected["len"], expected["acc"]), name
