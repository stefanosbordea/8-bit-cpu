"""MAC #n clock by clock, against the instruction-level simulator.

stream_by_clock is a second, independent model of the stream: it is written from the state table and the STREAM
contract of 06-control.tex (FETCH, DECODE, STREAM x n, DRAIN1, DRAIN2; read registers XR/WR, valid bits V1/V2, CNT).
The simulator must give the same ACC, PROD and pointers, and charge the same number of cycles, n + 4.
"""
import numpy as np

from lsllm.ls.isa import encode
from lsllm.ls.iss import LS


def stream_by_clock(x, w, p0, p1, acc, n):
    """One row per cycle with the values held during it: (state, CNT, P0, V1, V2, PROD, ACC).

    The last row is the FETCH of the next instruction. XR and WR start with stale values (99).
    """
    state, cnt, xr, wr, v1, v2, prod = "FETCH", 0, 99, 99, 0, 0, 0
    rows = []
    while True:
        rows.append((state, cnt, p0, v1, v2, prod, acc))
        if state == "FETCH" and len(rows) > 1:
            return rows
        # the closing clock edge: every register loads from the values before the edge
        acc += prod if v2 else 0                         # ACC adds PROD when V2 = 1
        prod = xr * wr if v1 else prod                   # PROD loads XR x WR when V1 = 1
        v1, v2 = int(state == "STREAM"), v1
        xr, wr = x[p0 % 256], w[p1 % 256]                # the read registers load on every clock
        if state == "FETCH":
            state = "DECODE"
        elif state == "DECODE":
            state, cnt = "STREAM", n
        elif state == "STREAM":
            state = "STREAM" if cnt != 1 else "DRAIN1"
            p0, p1, cnt = p0 + 1, p1 + 1, cnt - 1
        else:
            state = "DRAIN2" if state == "DRAIN1" else "FETCH"


def test_contract_reproduces_the_mac4_table_of_the_document():
    """06-control.tex: b = 0, x = 1,2,3,4, w = 5,6,7,8, nine cycles (the ninth is the next FETCH)."""
    x, w = [1, 2, 3, 4] + [0] * 252, [5, 6, 7, 8] + [0] * 252          # the memories have 256 cells
    state, cnt, p0, v1, v2, prod, acc = zip(*stream_by_clock(x, w, 0, 0, 0, 4))
    assert state == ("FETCH", "DECODE", "STREAM", "STREAM", "STREAM", "STREAM", "DRAIN1", "DRAIN2", "FETCH")
    assert cnt[2:6] == (4, 3, 2, 1)
    assert p0 == (0, 0, 0, 1, 2, 3, 4, 4, 4)
    assert v1 == (0, 0, 0, 1, 1, 1, 1, 0, 0)
    assert v2 == (0, 0, 0, 0, 1, 1, 1, 1, 0)
    assert prod[4:] == (5, 12, 21, 32, 32)
    assert acc == (0, 0, 0, 0, 0, 5, 17, 38, 70)


def test_simulator_agrees_with_the_contract_for_every_stream_length():
    rng = np.random.default_rng(7)
    for n in range(1, 256):
        x, w = rng.integers(-128, 128, (2, 256)).tolist()
        p0, p1, acc = (int(v) for v in rng.integers(0, 3000, 3))
        rows = stream_by_clock(x, w, p0, p1, acc, n)
        cpu = LS([encode("MAC", n)], x, w)
        cpu.p, cpu.acc = [p0, p1], acc
        cpu.step()
        assert (cpu.acc, cpu.prod, cpu.p[0]) == (rows[-1][6], rows[-1][5], rows[-1][2])
        assert cpu.cycles == len(rows) - 1 == n + 4
