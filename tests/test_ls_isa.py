"""LS instruction encodings against the design document (05-isa.tex, figures/formats.tex, figures/opcodes.tex)."""
import itertools

import numpy as np
import pytest

from lsllm.ls.isa import FORMATS, INSNS, TABLE, decode, encode, rq, signed

# "Encoding examples" of 05-isa.tex: mnemonic, operand values in assembly order, word
EXAMPLES = [
    ("MAC", (64,), 0x8040),          # MAC #64
    ("MAC", (0,), 0x8000),           # MAC
    ("LDI", (1, 0xC0), 0x14C0),      # LDI R1, 0xC0
    ("AND", (0, 1), 0x2102),         # AND R0, R1
    ("MVP", (0, 1, 0), 0x3400),      # MVP P0, R1, R0
    ("LD", (1, 1, 1), 0x4700),       # LD R1, [P1]+
    ("LDA", (2, 3), 0x7B01),         # LDA R2, R3
    ("RQ", (0, 2, 1), 0x9202),       # RQ R0, 2, relu
    ("DJNZ", (0x009,), 0xF009),      # DJNZ 0x009
]
# figures/opcodes.tex and the function codes of 05-isa.tex: opcode -> format, mnemonics in function-code order
OPCODES = {
    0x0: ("R", "NOP HALT IN OUT"), 0x1: ("I", "LDI"), 0x2: ("R", "ADD SUB AND OR XOR MOV SHL SHR SAR"),
    0x3: ("R", "MVP"), 0x4: ("M", "LD"), 0x5: ("M", "ST"), 0x6: ("R", "MUL"), 0x7: ("R", "ADDP LDA"),
    0x8: ("N", "MAC"), 0x9: ("Q", "RQ"), 0xA: ("A", "SETP0"), 0xB: ("A", "SETP1"), 0xC: ("A", "JMP"),
    0xD: ("A", "BZ"), 0xE: ("A", "BNZ"), 0xF: ("A", "DJNZ"),
}
# figures/formats.tex read independently: the bits of each format that are not opcode and not reserved
USED_BITS = {"R": 0x0F0F, "I": 0x0CFF, "M": 0x0F00, "Q": 0x0E1F, "A": 0x0FFF, "N": 0x00FF}


@pytest.mark.parametrize("name, args, word", EXAMPLES)
def test_encoding_examples_of_the_document(name, args, word):
    assert encode(name, *args) == word
    assert decode(word) == (TABLE[name], args)


def test_opcode_map_and_function_codes():
    for op, (fmt, names) in OPCODES.items():
        rows = [insn for insn in INSNS if insn.op == op]
        assert [insn.name for insn in rows] == names.split()
        assert all(insn.fmt == fmt for insn in rows)
        if len(rows) > 1:                 # SYS, ALU and opcode 7 are told apart by the function code
            assert [insn.fixed["fn"] for insn in rows] == list(range(len(rows)))
    assert len(INSNS) == len(TABLE) == 28


def test_format_bit_fields():
    for fmt, fields in FORMATS.items():
        masks = [(1 << (high + 1)) - (1 << low) for high, low in fields.values()]
        assert sum(masks) == USED_BITS[fmt]              # no overlap (sum == or) and the reserved bits left out
        assert sum(masks) == np.bitwise_or.reduce(masks)


def test_encode_and_decode_agree_on_all_65536_words():
    valid = {}
    for insn in INSNS:
        spans = [FORMATS[insn.fmt][field] for field in insn.args]
        tops = [insn.limit.get(field, (1 << (high - low + 1)) - 1) for field, (high, low) in zip(insn.args, spans)]
        for args in itertools.product(*[range(top + 1) for top in tops]):
            word = encode(insn.name, *args)
            assert word not in valid                     # no two instructions share a word
            valid[word] = (insn, args)
    # Counted by hand from the formats: NOP 1, HALT 1, IN 4, OUT 4, LDI 4*256, six two-register ALU 6*16,
    # three shifts 3*4, MVP 2*16, LD 4*2*2, ST 4*2, MUL 16, ADDP 1, LDA 16, MAC 256, RQ 4*24*2, six A-format 6*4096.
    assert len(valid) == 26_255
    for word in range(1 << 16):
        if word in valid:
            assert decode(word) == valid[word]
        else:
            with pytest.raises(ValueError):
                decode(word)


@pytest.mark.parametrize("name, args", [
    ("LDI", (4, 0)), ("LDI", (0, 256)), ("LDI", (0, -1)), ("RQ", (0, 24, 0)), ("RQ", (0, 2, 2)), ("MAC", (256,)),
    ("ST", (0, 1, 0)), ("MVP", (2, 0, 0)), ("JMP", (4096,)), ("ADD", (0,)), ("HALT", (0,)),
])
def test_encode_rejects_values_that_do_not_fit(name, args):
    with pytest.raises(ValueError):
        encode(name, *args)


@pytest.mark.parametrize("word", [0x0004, 0x0010, 0x0102, 0x1100, 0x2009, 0x3002, 0x4001, 0x5200, 0x6001, 0x7002,
                                  0x8100, 0x9018, 0x9100])
def test_decode_rejects_undefined_words(word):
    with pytest.raises(ValueError):
        decode(word)


def test_signed():
    assert [signed(v, 8) for v in (0, 127, 128, 255, 256)] == [0, 127, -128, -1, 0]
    assert signed(0xFFFED3, 24) == -301                  # figures/rq.tex
    assert signed(0x800000, 24) == -(1 << 23) and signed(0x7FFFFF, 24) == (1 << 23) - 1


def test_rq_examples_of_the_document():
    assert rq(-301, 2, 0) == -76                         # rounds toward minus infinity, not -75
    assert rq(-301, 2, 1) == 0                           # ReLU
    assert rq(-1, 1, 0) == -1                            # 05-isa.tex: -1 >> 1 = -1


def test_rq_against_numpy_at_the_24_bit_edges():
    rng = np.random.default_rng(0)
    edges = [-(1 << 23), -(1 << 23) + 1, -32769, -32768, -257, -256, -129, -128, -127, -1, 0, 1, 127, 128, 255,
             256, 32767, 32768, (1 << 23) - 2, (1 << 23) - 1]
    accs = np.array(edges + rng.integers(-(1 << 23), 1 << 23, 2000).tolist(), dtype=np.int64)
    for sh in range(24):
        floor = np.floor_divide(accs, 1 << sh)           # division by 2^sh that rounds toward minus infinity
        for relu in (0, 1):
            want = np.clip(np.maximum(floor, 0) if relu else floor, -128, 127)
            assert [rq(int(acc), sh, relu) for acc in accs] == want.tolist()
