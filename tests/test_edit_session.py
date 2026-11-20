import json

import numpy as np
import pytest

from lsllm.edit.session import Session, summary

BASE = "runs/c/e0_int_calib"
pytestmark = pytest.mark.skipif(not __import__("os").path.exists(BASE + "/model.npz"), reason="no exported model")


def test_edit_log_undo_and_rules(tmp_path):
    s = Session.create(BASE, str(tmp_path / "s"), "test")
    w = s.get("W2_0")
    before = w.copy()
    w[3, 5] += 1
    w[0, 0] -= 2
    entry = s.put("W2_0", w, "two numbers")
    assert entry["changed"] == 2 and entry["positions"] == [0, 3 * w.shape[1] + 5]
    assert entry["old"] == [int(before[0, 0]), int(before[3, 5])] and entry["new"] == [int(w[0, 0]), int(w[3, 5])]
    assert s.put("W2_0", w, "nothing new") is None
    again = Session(str(tmp_path / "s"))
    assert np.array_equal(again.get("W2_0"), w)
    assert s.undo()["new"] == entry["old"] and np.array_equal(s.get("W2_0"), before)
    c_l = s.spec["preloads"]["c_l"]
    assert s.put_number(("preloads", "c_l"), c_l + 5, "offset")["old"] == [c_l]
    s.undo()
    assert Session(str(tmp_path / "s")).spec["preloads"]["c_l"] == c_l
    for name, bad in (("W2_0", -128), ("E", -128)):
        w = s.get(name)
        w[0, 0] = bad
        with pytest.raises(AssertionError):
            s.put(name, w, "weights are -127..127")
    exp = s.get("EXP")
    exp[0, 0] = 100
    with pytest.raises(AssertionError):
        s.put("EXP", exp, "the largest weight must stay 127")
    with pytest.raises(AssertionError):
        s.put_number(("preloads", "c_l"), 40000, "does not fit 16 bits")
    assert np.array_equal(s.get("W2_0"), before) and len(s.edits()) == 4
    assert any("two numbers" in line for line in summary(str(tmp_path / "s")))


def test_budget_is_enforced(tmp_path):
    s = Session.create(BASE, str(tmp_path / "s"), "test")
    out = s.trace([[1, 5, 9], [1, 7]], ["l0.g", "u"], "probe")
    assert out[0]["l0.g"].shape[0] == 3 and out[1]["u"].shape == (2, s.model().cfg.vocab_size)
    assert s.left() == 49
    s.budget["runs"] += [{"run": g, "kind": "trace", "note": "", "edits_so_far": 0} for g in range(49)]
    with pytest.raises(RuntimeError):
        s.trace([[1]], ["u"], "one too many")
    with pytest.raises(RuntimeError):
        s.evaluate("one too many")
