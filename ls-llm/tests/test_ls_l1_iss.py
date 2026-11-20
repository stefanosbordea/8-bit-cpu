"""The L1 machine of docs/SPEC.md Part A in the simulator: encodings, effects and cycles of the new and changed
instructions, one data memory, 20-bit pointers, LEN, tables, UART, LFSR. The L0 tests stay as they are."""
import numpy as np
import pytest

from lsllm.ls.isa import TABLE, TABLE_L1, decode, encode
from lsllm.ls.iss import L1


def enc(name, *args):
    return encode(name, *args, table=TABLE_L1)


def cpu_after(program, data=(), size=None, **state):
    """A CPU that has executed every instruction of `program` (a list of (mnemonic, operands...)) once, in order."""
    cpu = L1([enc(*insn) for insn in program], data, size=size)
    for name, value in state.items():
        setattr(cpu, name, value)
    for insn in program:
        cpu.step()
    return cpu


def test_encodings_of_the_spec_table():
    assert enc("SETP0", 0xABC) == 0xAABC and enc("SETP1", 0x123) == 0xB123        # unchanged
    assert [enc("MVP", p, 1, 2) for p in range(3)] == [0x3600, 0x3601, 0x3602]    # op 3, fn = p
    assert [enc("MVH", p, 3) for p in range(3)] == [0x3304, 0x3305, 0x3306]       # fn = 4 + p, r2 = Rb
    assert enc("MVL", 1, 2) == 0x3603                                             # fn = 3
    names = {(d, s): "CPP%d%d" % (d, s) for d in range(3) for s in range(3) if d != s}
    assert sorted(enc(name) & 0xF for name in names.values()) == [9, 10, 11, 13, 14, 15]
    assert all(enc(name) == 0x3000 | 8 + 3 * d + s for (d, s), name in names.items())
    assert enc("LD", 2, 0, 0) == 0x4800 and enc("LD", 2, 1, 1) == 0x4B00          # pointer id = IR[7] & IR[9]
    assert enc("LD2", 2, 0) == 0x4880 and enc("LD2", 2, 1) == 0x4980              # P2: IR[7] = 1, IR[8] = increment
    assert enc("ST", 1, 0, 1) == 0x5500 and enc("ST2", 1, 1) == 0x5580
    assert enc("LUT", 3, 0xB) == 0x0CB4 and enc("LUT", 1, 63) == 0x07F4           # op 0, fn 4, table in IR[9:4]
    assert enc("TX", 2) == 0x0805 and enc("RND", 1) == 0x0406
    assert enc("MACL") == 0x8100 and enc("MAC", 0) == 0x8000 and enc("MAC", 255) == 0x80FF
    assert enc("NRM", 1) == 0x0407                                                # op 0, fn 7
    assert enc("RQV", 0, 0, 0) == 0x9100 and enc("RQV", 3, 2, 1) == 0x9F02        # IR[8] = 1, Rs = IR[1:0], relu = IR[9]
    with pytest.raises(ValueError):
        enc("ST", 0, 1, 0)                                                        # no store through P1
    with pytest.raises(ValueError):
        enc("MVH", 3, 0)                                                          # there is no P3


def test_every_l1_word_decodes_to_one_instruction_and_back():
    valid = 0
    for word in range(1 << 16):
        try:
            insn, args = decode(word, TABLE_L1)
        except ValueError:
            continue
        valid += 1
        assert enc(insn.name, *args) == word
    assert valid > 26255                                                          # L0 has 26,255 valid words


def test_baseline_instructions_keep_their_words():
    for name, insn in TABLE.items():
        args = [0] * len(insn.args)
        assert enc(name, *args) == encode(name, *args)


def test_reset_state_and_memory_size():
    cpu = L1([], [1, -2, 3], size=8)
    assert list(cpu.d) == [1, -2, 3, 0, 0, 0, 0, 0] and cpu.view.dtype == np.int8
    assert (cpu.p, cpu.len, cpu.acc, cpu.tx, cpu.lfsr, cpu.cycles) == ([0, 0, 0], 0, 0, [], 0xACE1ACE1, 0)
    assert L1([], sw=0xACE1).lfsr == 0xACE10000 and L1([], sw=3).lfsr == 0xACE1ACE2   # the upper half keeps it away from 0
    L1([], size=1 << 19)
    with pytest.raises(ValueError):
        L1([], size=(1 << 19) + 1)


def test_setp_and_mvp_set_the_low_12_bits_only():
    cpu = cpu_after([("SETP0", 0x123), ("SETP1", 0xFFF), ("MVP", 2, 0, 1), ("MVP", 0, 0, 1)],
                    p=[0xAB000, 0xCD456, 0xEF789], r=[0x5C, 0x34, 0, 0])
    assert cpu.p == [0xABC34, 0xCDFFF, 0xEFC34]                                   # MVP: Ra[3:0] & Rb


def test_mvh_mvl_cpp():
    cpu = cpu_after([("MVH", 0, 1), ("MVH", 2, 2), ("MVL", 0, 3)], p=[0x12345, 0x6789A, 0xBCDEF], r=[0xF7, 0xAB, 0x00, 0x21])
    assert cpu.p == [0xAB345, 0x6789A, 0x00DEF] and cpu.len == 0x721
    for d in range(3):
        for s in range(3):
            if d != s:
                cpu = cpu_after([("CPP%d%d" % (d, s),)], p=[0x11111, 0x22222, 0x33333])
                expected = [0x11111, 0x22222, 0x33333]
                expected[d] = expected[s]
                assert cpu.p == expected


def test_ld_and_st_through_the_three_pointers():
    data = list(range(-64, 64))
    cpu = cpu_after([("LD", 0, 0, 1), ("LD", 1, 1, 0), ("LD2", 2, 1), ("LD2", 3, 0)], data, p=[5, 70, 126])
    assert cpu.r == [(-59) & 0xFF, 6, 62, 63] and cpu.p == [6, 70, 127]           # one memory for all pointers
    cpu = cpu_after([("ST", 0, 0, 1), ("ST2", 1, 1), ("ST2", 2, 0)], data, p=[3, 0, 9], r=[0x80, 0x7F, 0xFF, 0])
    assert (cpu.d[3], cpu.d[9], cpu.d[10]) == (-128, 127, -1) and cpu.p == [4, 0, 10]
    cpu = cpu_after([("LD", 0, 0, 1), ("ST2", 0, 1)], [7] * 8, size=1 << 19, p=[0xFFF, 0, 0x7FFFF])
    assert cpu.p == [0x1000, 0, 0x80000]                                          # full 20-bit counters
    assert cpu.d[0x7FFFF] == 0


def test_lut_reads_table_t_and_sets_the_flags():
    data = np.zeros(4096 + 16 * 256, dtype=np.int8)
    data[4096 + 256 * 5 + 200] = -3
    cpu = L1([enc("LUT", 1, 5), enc("LUT", 2, 5)], data, lut_base=4096)
    cpu.r = [0, 200, 201, 0]
    cpu.step()
    assert (cpu.r[1], cpu.z, cpu.n) == (0xFD, 0, 1)
    cpu.step()
    assert (cpu.r[2], cpu.z, cpu.n) == (0, 1, 0)


def test_nrm_counts_the_significant_bits_of_acc():
    for acc, bits in ((0, 0), (1, 1), (2, 2), (3, 2), (127, 7), (128, 8), (255, 8), (256, 9), (16383, 14), (16384, 15),
                      (32512, 15), (32767, 15), (32768, 16), (2 ** 23 - 1, 23)):
        cpu = cpu_after([("NRM", 2)], acc=acc, z=1, n=1)
        assert cpu.r == [0, 0, bits, 0] and (cpu.acc, cpu.z, cpu.n, cpu.cycles) == (acc, 1, 1, 4)


def test_rqv_is_rq_with_the_shift_taken_from_a_register():
    rng = np.random.default_rng(0)
    for acc in [0, 1, -1, 127, 128, -128, -129, 2 ** 23 - 1, -2 ** 23] + rng.integers(-2 ** 23, 2 ** 23, 40).tolist():
        for sh in (0, 1, 7, 8, 15, 16, 23):
            for relu in (0, 1):
                fixed = cpu_after([("RQ", 1, sh, relu)], acc=acc)
                cpu = cpu_after([("RQV", 1, 3, relu)], acc=acc, r=[0, 0, 0, sh | 0xE0])     # only Rs[4:0] counts
                assert cpu.r[1] == fixed.r[1] and cpu.r[3] == sh | 0xE0 and cpu.acc == acc and cpu.cycles == 4
    assert cpu_after([("RQV", 0, 0, 0)], acc=-500, r=[3, 0, 0, 0]).r[0] == (-63) & 0xFF     # Rd = Rs: the old value shifts
    words = [word for word in range(0x9000, 0xA000) if decodes(word)]
    assert len(words) == 4 * 2 * 24 + 4 * 2 * 4                                   # RQ: 24 shifts; RQV: 4 registers


def decodes(word):
    try:
        return decode(word, TABLE_L1)[0].name in ("RQ", "RQV")
    except ValueError:
        return False


def test_tx_collects_bytes_and_rnd_follows_the_lfsr_of_the_spec():
    cpu = cpu_after([("TX", 1), ("TX", 0), ("RND", 2), ("RND", 3)], r=[0x12, 0x85, 0, 0])
    assert cpu.tx == [0x85, 0x12]
    s, values = 0xACE1ACE1, []
    for g in range(16):                                                           # the step of SPEC A.2, bit by bit
        bit = (s ^ s >> 10 ^ s >> 30 ^ s >> 31) & 1
        s = s >> 1 | bit << 31
        if g % 8 == 7:
            values.append(s >> 24)
    assert cpu.r[2:] == values and cpu.lfsr == s


def test_streams_multiply_any_two_parts_of_the_memory_and_cost_n_plus_4():
    rng = np.random.default_rng(1)
    data = rng.integers(-128, 128, 6000)
    for n in (1, 2, 23, 24, 100, 255):
        cpu = cpu_after([("MAC", n)], data, p=[7, 1500, 0], acc=-1234)
        assert cpu.acc == -1234 + int(data[7:7 + n] @ data[1500:1500 + n]) and cpu.p == [7 + n, 1500 + n, 0]
        assert cpu.prod == int(data[7 + n - 1] * data[1500 + n - 1])
        assert (cpu.cycles, cpu.elems) == (n + 4, [n]) and not cpu.overflow
    cpu = cpu_after([("MAC", 0)], data, p=[7, 7, 0])                              # the classic MAC: 4 cycles
    assert cpu.acc == int(data[7]) ** 2 and cpu.p == [8, 8, 0] and (cpu.cycles, cpu.elems) == (4, [0])
    for length in (1, 300, 1000, 4095):
        cpu = cpu_after([("MACL",)], data, p=[0, 1500, 0], len=length)
        assert cpu.acc == int(data[:length] @ data[1500:1500 + length]) and not cpu.overflow
        assert cpu.cycles == length + 4 and cpu.p == [length, 1500 + length, 0]
    with pytest.raises(RuntimeError, match="LEN = 0"):
        cpu_after([("MACL",)], data)
    with pytest.raises(IndexError):
        cpu_after([("MAC", 9)], data, p=[5995, 0, 0])


def test_the_sticky_flag_sees_an_overflow_inside_a_stream():
    most = np.full(1024, -128, dtype=np.int8)                                     # every product is 16,384
    assert not cpu_after([("MACL",)], most, p=[0, 512, 0], len=511).overflow      # 8,372,224 fits
    cpu = cpu_after([("MACL",)], most, p=[0, 512, 0], len=512)                    # 2^23 does not
    assert cpu.overflow and cpu.acc == -(1 << 23)
    data = np.concatenate([np.full(600, 127), np.full(300, 127), np.full(300, -127)]).astype(np.int8)
    cpu = cpu_after([("MACL",)], data, p=[0, 600, 0], len=600, acc=8_000_000)     # leaves the range and comes back
    assert cpu.overflow and cpu.acc == 8_000_000
    assert not cpu_after([("MACL",)], data, p=[0, 900, 0], len=300, acc=8_000_000).overflow


def test_run_stops_at_halt_or_at_an_address_and_counts_per_address():
    words = [enc("LDI", 3, 3), enc("MAC", 4), enc("DJNZ", 1), enc("JMP", 0)]
    cpu = L1(words, [1] * 64)
    cpu.run(stop=0)
    assert cpu.pc == 0 and cpu.count == [1, 3, 3, 1] and cpu.elems == [0, 12, 0, 0]
    assert cpu.instructions == 8 and cpu.cycles == 4 * 8 + 12 and cpu.acc == 12
    with pytest.raises(RuntimeError, match="not finished"):
        L1([enc("JMP", 0)]).run(limit=100)
    cpu = L1([enc("HALT")])
    cpu.run()
    with pytest.raises(RuntimeError, match="halted"):
        cpu.step()
    with pytest.raises(RuntimeError, match="outside"):
        L1([enc("NOP")]).run()
