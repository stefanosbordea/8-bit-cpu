"""The LS assembler: the syntax of the listings in 07-programs.tex, the listing output and the error messages."""
import pytest

from lsllm.ls.asm import assemble

# "Encoding examples" of 05-isa.tex, as they are written there
EXAMPLES = {
    "MAC #64": 0x8040, "MAC": 0x8000, "LDI R1, 0xC0": 0x14C0, "AND R0, R1": 0x2102, "MVP P0, R1, R0": 0x3400,
    "LD R1, [P1]+": 0x4700, "LDA R2, R3": 0x7B01, "RQ R0, 2, relu": 0x9202, "DJNZ 0x009": 0xF009,
}


@pytest.mark.parametrize("text", EXAMPLES)
def test_encoding_examples_of_the_document(text):
    assert assemble(text).words == [EXAMPLES[text]]


def test_every_instruction_is_assembled():
    source = """
        NOP
        HALT
        IN   R2
        OUT  R3
        LDI  R3, 255
        ADD  R1, R2
        SUB  R1, R2
        AND  R1, R2
        OR   R1, R2
        XOR  R1, R2
        MOV  R1, R2
        SHL  R1
        SHR  R2
        SAR  R3
        MVP  P1, R2, R3
        LD   R2, [P0]
        LD   R2, [P0]+
        LD   R2, [P1]
        ST   R3, [P0]
        ST   R3, [P0]+
        MUL  R2, R3
        ADDP
        LDA  R0, R1
        MAC  #0
        MAC  #255
        RQ   R3, 23, 0
        SETP0 0xFFF
        SETP1 0b101
        JMP  1
        BZ   2
        BNZ  3
        DJNZ 4
    """
    assert assemble(source).words == [
        0x0000, 0x0001, 0x0802, 0x0C03, 0x1CFF, 0x2600, 0x2601, 0x2602, 0x2603, 0x2604, 0x2605, 0x2406, 0x2807,
        0x2C08, 0x3B01, 0x4800, 0x4900, 0x4A00, 0x5C00, 0x5D00, 0x6B00, 0x7000, 0x7101, 0x8000, 0x80FF, 0x9C17,
        0xA000 | 0xFFF, 0xB005, 0xC001, 0xD002, 0xE003, 0xF004,
    ]


def test_labels_constants_comments_and_listing():
    program = assemble("""; a comment line
N = 3
COUNT = N               ; a constant from an earlier constant
start:
        LDI   R3, COUNT
loop:   NOP             ; label in front of an instruction
        DJNZ  loop      ; backward reference
        BZ    done      ; forward reference
        JMP   start
done:   HALT""")
    assert program.words == [0x1C03, 0x0000, 0xF001, 0xD005, 0xC000, 0x0001]
    assert program.listing.splitlines() == [
        "           ; a comment line",
        "           N = 3",
        "           COUNT = N               ; a constant from an earlier constant",
        "           start:",
        "000  1C03          LDI   R3, COUNT",
        "001  0000  loop:   NOP             ; label in front of an instruction",
        "002  F001          DJNZ  loop      ; backward reference",
        "003  D005          BZ    done      ; forward reference",
        "004  C000          JMP   start",
        "005  0001  done:   HALT",
    ]


def test_region_marks_take_the_address_of_the_next_instruction():
    program = assemble("NOP\n.region first\nNOP\nNOP\nlast: .region second\nHALT")
    assert program.regions == {1: "first", 3: "second"}
    assert program.words == [0, 0, 0, 1]


@pytest.mark.parametrize("source, message", [
    ("NOP\nFOO R1", "line 2: unknown instruction 'FOO'"),
    ("ADD R0, 2", "line 1: wrong operands for ADD"),           # a number where a register is expected
    ("LDI R0", "wrong operands for LDI"),
    ("LDI R0, R1", "wrong operands for LDI"),
    ("MAC 64", "wrong operands for MAC"),                      # the count is written #64
    ("MVP R0, R1, R2", "wrong operands for MVP"),
    ("LD R0, P1", "wrong operands for LD"),
    ("RQ R0, 2", "wrong operands for RQ"),                     # the ReLU operand is not optional
    ("nop", "unknown instruction 'nop'"),
    ("LDI R0, 256", "imm8 = 256 is outside 0..255"),
    ("LDI R0, -1", "imm8 = -1 is outside 0..255"),
    ("RQ R0, 24, 0", "sh = 24 is outside 0..23"),
    ("RQ R0, 2, 2", "relu = 2 is outside 0..1"),
    ("MAC #256", "n = 256 is outside 0..255"),
    ("SETP0 4096", "imm12 = 4096 is outside 0..4095"),
    ("ST R0, [P1]", "p = 1 is outside 0..0"),                  # ST writes the X memory only
    ("JMP nowhere", "'nowhere' is not a number or a defined name"),
    ("LDI R4, 0", "'R4' is not a number or a defined name"),
    ("a: NOP\na: NOP", "line 2: 'a' is defined twice"),
    ("K = 1\nK = 2", "line 2: 'K' is defined twice"),
    ("relu = 0", "'relu' is defined twice"),
    ("K = L", "'L' is not a number or a defined name"),
])
def test_errors(source, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace(".", r"\.")):
        assemble(source)


def test_program_longer_than_the_program_counter_is_rejected():
    assert len(assemble("NOP\n" * 4096).words) == 4096
    with pytest.raises(ValueError, match="12-bit program counter"):
        assemble("NOP\n" * 4097)
