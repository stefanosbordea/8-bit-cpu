"""The LS simulator, instruction by instruction: effects, flags, pointers, saturation, ACC wrap-around, counters."""
import numpy as np
import pytest

from lsllm.ls.asm import assemble
from lsllm.ls.isa import INSNS, encode
from lsllm.ls.iss import LS


def one(text, x=(), w=(), **state):
    """A CPU that has executed the first instruction of `text`, starting from the given state."""
    cpu = LS(assemble(text).words, x, w)
    for name, value in state.items():
        setattr(cpu, name, value)
    cpu.step()
    return cpu


def run(text, x=(), w=(), **options):
    """A CPU that has run the program `text` to its HALT."""
    program = assemble(text)
    cpu = LS(program.words, x, w, program.regions, **options)
    cpu.run()
    return cpu


def test_reset_state():
    cpu = LS([])
    assert (cpu.r, cpu.z, cpu.n, cpu.pc, cpu.p, cpu.prod, cpu.acc) == ([0, 0, 0, 0], 0, 0, 0, [0, 0], 0, 0)
    assert (cpu.outr, cpu.halted, cpu.overflow, cpu.cycles, cpu.instructions) == (0, False, False, 0, 0)
    assert cpu.mem == [[0] * 256, [0] * 256]


def test_cycle_rule_4_per_instruction_and_n_plus_4_for_a_stream():
    for insn in INSNS:                                   # operands all 0: for MAC that is the classic MAC
        cpu = LS([encode(insn.name, *[0] * len(insn.args))])
        cpu.step()
        assert (cpu.cycles, cpu.instructions, cpu.insn_cycles) == (4, 1, {insn.name: 4})
    for n in range(1, 256):
        cpu = LS([encode("MAC", n)])
        cpu.step()
        assert (cpu.cycles, cpu.instructions) == (n + 4, 1)


def test_nop_and_halt():
    cpu = one("NOP\nHALT", r=[1, 2, 3, 4], z=1, n=1, p=[5, 6], acc=7, prod=8)
    assert (cpu.r, cpu.z, cpu.n, cpu.p, cpu.acc, cpu.prod, cpu.pc) == ([1, 2, 3, 4], 1, 1, [5, 6], 7, 8, 1)
    assert not cpu.halted
    cpu.step()
    assert cpu.halted and (cpu.cycles, cpu.instructions) == (8, 2)       # HALT costs 4 cycles too
    with pytest.raises(RuntimeError, match="halted"):
        cpu.step()
    assert (cpu.cycles, cpu.instructions) == (8, 2)                      # frozen


def test_in_out_ldi():
    assert one("IN R1", sw=0x1A5).r == [0, 0xA5, 0, 0]                   # SW7..0 only
    assert one("OUT R2", r=[1, 2, 3, 4]).outr == 3
    assert one("LDI R3, 0xC0").r == [0, 0, 0, 0xC0]


# The ALU on every pair of operands. Expected values: numpy's own 8-bit arithmetic (uint8 wraps, int8 >> is arithmetic).
A, B = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8), indexing="ij")
ALU = {
    "ADD R0, R1": A + B, "SUB R0, R1": A - B, "AND R0, R1": A & B, "OR R0, R1": A | B, "XOR R0, R1": A ^ B,
    "MOV R0, R1": B, "SHL R0": A << 1, "SHR R0": A >> 1, "SAR R0": (A.astype(np.int8) >> 1).astype(np.uint8),
}


@pytest.mark.parametrize("text", ALU)
def test_alu_all_65536_operand_pairs_with_flags(text):
    cpu = LS(assemble(text).words)
    want = ALU[text].tolist()
    for a in range(256):
        for b in range(256):
            cpu.pc, cpu.r[0], cpu.r[1] = 0, a, b
            cpu.step()
            t = want[a][b]
            assert (cpu.r[0], cpu.r[1], cpu.z, cpu.n) == (t, b, t == 0, t >= 128), (a, b)
    assert cpu.instructions == 65536


def test_alu_register_fields():
    assert one("SUB R2, R3", r=[1, 2, 10, 3]).r == [1, 2, 7, 3]          # Rd is the first operand, Rs the second
    assert one("MOV R3, R1", r=[1, 2, 10, 3]).r == [1, 2, 10, 2]
    assert one("SAR R1", r=[1, 0x82, 10, 3]).r == [1, 0xC1, 10, 3]


def test_only_alu_instructions_write_the_flags():
    others = """
        IN   R0
        LDI  R1, 0
        MVP  P0, R1, R1
        LD   R2, [P0]+
        ST   R1, [P0]+
        MUL  R1, R1
        ADDP
        LDA  R1, R1
        MAC
        MAC  #3
        RQ   R0, 0, relu
        SETP0 0
        SETP1 0
        OUT  R0
        LDI  R3, 1
        DJNZ 0
        JMP  end
end:    HALT"""
    for z, n in ((0, 0), (1, 1)):
        cpu = LS(assemble(others).words)
        cpu.z, cpu.n = z, n
        cpu.run()
        assert (cpu.z, cpu.n, cpu.instructions) == (z, n, 18)            # results of 0 did not set Z
    cpu = one("XOR R0, R0", r=[0x80, 0, 0, 0], n=1)
    assert (cpu.z, cpu.n) == (1, 0)


def test_mvp_joins_low_nibble_and_byte():
    assert one("MVP P0, R1, R2", r=[0, 0xAB, 0xCD, 0]).p == [0xBCD, 0]
    assert one("MVP P1, R2, R1", r=[0, 0xAB, 0xCD, 0]).p == [0, 0xDAB]


X = [g - 128 for g in range(256)]                        # X[a] = a - 128
W = [(g * 7) % 256 - 128 for g in range(256)]            # W[a] = 7a mod 256 - 128


@pytest.mark.parametrize("text, p, r, p_after", [
    ("LD R1, [P0]", [5, 9], [0, (5 - 128) & 255, 0, 0], [5, 9]),
    ("LD R1, [P0]+", [5, 9], [0, (5 - 128) & 255, 0, 0], [6, 9]),
    ("LD R2, [P1]", [5, 9], [0, 0, (63 - 128) & 255, 0], [5, 9]),
    ("LD R2, [P1]+", [5, 9], [0, 0, (63 - 128) & 255, 0], [5, 10]),
    ("LD R3, [P0]+", [0xFFF, 0], [0, 0, 0, 127], [0, 0]),                # the pointer wraps at 12 bits
    ("LD R3, [P1]", [0, 0x305], [0, 0, 0, (35 - 128) & 255], [0, 0x305]),   # the memory sees P1[7:0]
])
def test_ld(text, p, r, p_after):
    cpu = one(text, X, W, p=p)
    assert (cpu.r, cpu.p, cpu.mem) == (r, p_after, [X, W])


@pytest.mark.parametrize("text, p, cell, p_after", [
    ("ST R1, [P0]", [7, 3], 7, [7, 3]),
    ("ST R1, [P0]+", [7, 3], 7, [8, 3]),
    ("ST R1, [P0]+", [0xFFF, 3], 255, [0, 3]),
])
def test_st(text, p, cell, p_after):
    cpu = one(text, X, W, p=p, r=[0, 0xFB, 0, 0])
    assert cpu.mem[0] == X[:cell] + [-5] + X[cell + 1:]                  # byte 0xFB is stored as the int8 -5
    assert (cpu.mem[1], cpu.p, cpu.r) == (W, p_after, [0, 0xFB, 0, 0])


def test_stored_value_is_read_back_by_ld_and_mac():
    cpu = run("LDI R0, 0xFE\nSETP0 4\nST R0, [P0]\nLD R1, [P0]\nMAC\nHALT", w=[3])
    assert (cpu.r[1], cpu.prod, cpu.acc, cpu.p) == (0xFE, -6, -6, [5, 1])


def test_mul_all_65536_signed_pairs():
    cpu = LS(assemble("MUL R0, R1").words)
    want = (A.astype(np.int8).astype(np.int32) * B.astype(np.int8).astype(np.int32)).tolist()
    for a in range(256):
        for b in range(256):
            cpu.pc, cpu.r[0], cpu.r[1] = 0, a, b
            cpu.step()
            assert cpu.prod == want[a][b], (a, b)
    assert want[128][128] == 16384                                       # (-128)(-128) needs bit 15 of PROD
    assert (cpu.acc, cpu.instructions) == (0, 65536)                     # MUL leaves ACC alone


def test_addp_sign_extends_prod():
    assert one("ADDP", prod=16384, acc=-5).acc == 16379                  # bit 15 clear: +2^14, not -2^14
    assert one("ADDP", prod=-16256, acc=100).acc == -16156
    assert one("ADDP", prod=-1, acc=0).acc == -1


@pytest.mark.parametrize("high, low, acc", [(0x12, 0x34, 0x1234), (0x7F, 0xFF, 32767), (0x80, 0x00, -32768),
                                            (0xFF, 0x85, -123), (0x00, 0x80, 128)])
def test_lda_loads_a_sign_extended_16_bit_value(high, low, acc):
    assert one("LDA R2, R3", r=[0, 0, high, low], acc=-999).acc == acc


def test_classic_mac():
    cpu = one("MAC", X, W, p=[2, 5], acc=100)                            # X[2] = -126, W[5] = -93
    assert (cpu.prod, cpu.acc, cpu.p, cpu.cycles) == (11718, 11818, [3, 6], 4)


def test_stream_example_of_the_document():
    """06-control.tex: b = 0, x = 1,2,3,4, w = 5,6,7,8: PROD ends at 32, ACC at 70, both pointers advance by 4."""
    cpu = one("MAC #4", [1, 2, 3, 4], [5, 6, 7, 8])
    assert (cpu.prod, cpu.acc, cpu.p, cpu.cycles) == (32, 70, [4, 4], 8)


def dot(body, p0, p1, bias):
    return f"SETP0 {p0}\nSETP1 {p1}\nLDI R0, {bias >> 8 & 255}\nLDI R1, {bias & 255}\nLDA R0, R1\n{body}\nHALT"


@pytest.mark.parametrize("n", [1, 2, 3, 4, 64, 255])
@pytest.mark.parametrize("p0, p1", [(0, 0), (0xC0, 0x40), (0xFF0, 0x0F8)])       # the last pair wraps around
def test_stream_has_the_effect_of_n_classic_macs(n, p0, p1):
    rng = np.random.default_rng(n)
    x, w = rng.integers(-128, 128, (2, 256)).tolist()
    bias = int(rng.integers(-32768, 32768))
    products = [x[(p0 + k) % 256] * w[(p1 + k) % 256] for k in range(n)]
    stream, unrolled = run(dot(f"MAC #{n}", p0, p1, bias), x, w), run(dot("MAC\n" * n, p0, p1, bias), x, w)
    for cpu in (stream, unrolled):
        assert (cpu.acc, cpu.prod, cpu.overflow) == (bias + sum(products), products[-1], False)
        assert cpu.p == [(p0 + n) & 0xFFF, (p1 + n) & 0xFFF]
    assert (stream.cycles, stream.instructions, stream.insn_cycles["MAC"]) == (24 + n + 4, 7, n + 4)
    assert (unrolled.cycles, unrolled.instructions, unrolled.insn_cycles["MAC"]) == (24 + 4 * n, 6 + n, 4 * n)


def test_stream_of_64_worst_case_products():
    cpu = one("MAC #64", [-128] * 64, [-128] * 64)
    assert (cpu.acc, cpu.prod, cpu.overflow) == (1 << 20, 1 << 14, False)


@pytest.mark.parametrize("text, acc, byte", [
    ("RQ R2, 2, 0", -301, 0xB4),                         # figures/rq.tex: -301 >> 2 = -76 = 0xB4
    ("RQ R2, 2, relu", -301, 0x00),
    ("RQ R2, 0, 0", 127, 0x7F), ("RQ R2, 0, 0", 128, 0x7F), ("RQ R2, 0, relu", 1000, 0x7F),      # saturation
    ("RQ R2, 0, 0", -128, 0x80), ("RQ R2, 0, 0", -129, 0x80), ("RQ R2, 0, relu", -129, 0x00),
    ("RQ R2, 3, 0", 1023, 0x7F), ("RQ R2, 3, 0", 1015, 0x7E), ("RQ R2, 3, 0", -1025, 0x80),
    ("RQ R2, 23, 0", (1 << 23) - 1, 0x00), ("RQ R2, 23, 0", -(1 << 23), 0xFF), ("RQ R2, 16, 0", -(1 << 23), 0x80),
])
def test_rq_instruction(text, acc, byte):
    cpu = one(text, acc=acc, r=[1, 2, 3, 4])
    assert (cpu.r, cpu.acc) == ([1, 2, byte, 4], acc)


def test_pc_wraps_at_12_bits():
    cpu = LS([encode("HALT")] + [encode("NOP")] * 4095)
    cpu.pc = 0xFFF
    cpu.step()
    assert cpu.pc == 0
    cpu.step()
    assert cpu.halted


def test_setp_and_branches():
    assert one("SETP0 0xABC").p == [0xABC, 0]
    assert one("SETP1 0xFFF").p == [0, 0xFFF]
    for text, z, taken in (("JMP 0x123", 0, True), ("JMP 0x123", 1, True), ("BZ 0x123", 1, True),
                           ("BZ 0x123", 0, False), ("BNZ 0x123", 0, True), ("BNZ 0x123", 1, False)):
        cpu = one(text, z=z)
        assert (cpu.pc, cpu.cycles) == (0x123 if taken else 1, 4)        # 4 cycles, taken or not


def test_branch_on_the_flag_of_an_alu_result():
    cpu = run("LDI R0, 5\nSUB R0, R0\nBZ yes\nLDI R1, 1\nyes: BNZ no\nLDI R2, 2\nno: HALT")
    assert (cpu.r, cpu.instructions, cpu.cycles) == ([0, 0, 2, 0], 6, 24)


@pytest.mark.parametrize("r3, r3_after, taken", [(2, 1, True), (1, 0, False), (0, 255, True)])
def test_djnz(r3, r3_after, taken):
    cpu = one("DJNZ 0x123", r=[9, 8, 7, r3], z=1, n=1)
    assert (cpu.r, cpu.pc, cpu.z, cpu.n, cpu.cycles) == ([9, 8, 7, r3_after], 0x123 if taken else 1, 1, 1, 4)


def test_acc_takes_the_largest_value_and_wraps_on_the_next_product():
    """0x7FFF + 2 * 255 * 16384 is exactly 2^23 - 1, the largest ACC. One more product wraps around."""
    cpu = LS(assemble("LDI R0, 0x7F\nLDI R1, 0xFF\nLDA R0, R1\nMAC #255\nMAC #255\nMAC\nMAC\nLDA R0, R1").words,
             [-128] * 256, [-128] * 256)
    for g in range(5):
        cpu.step()
    assert (cpu.acc, cpu.overflow) == ((1 << 23) - 1, False)
    cpu.step()
    assert (cpu.acc, cpu.overflow) == (-(1 << 23) + 16383, True)
    cpu.step()
    assert (cpu.acc, cpu.overflow) == (-(1 << 23) + 32767, True)         # this sum fits: the flag is sticky
    cpu.step()
    assert (cpu.acc, cpu.overflow) == (32767, True)                      # and LDA does not clear it


def test_acc_wraps_below_the_smallest_value():
    cpu = LS(assemble("LDI R0, 0x80\nLDA R0, R1\nMAC #255\nMAC #255\nMAC #5").words, [-128] * 256, [127] * 256)
    for g in range(4):
        cpu.step()
    assert (cpu.acc, cpu.overflow) == (-32768 - 510 * 16256, False)
    cpu.step()
    assert (cpu.acc, cpu.overflow) == (-32768 - 515 * 16256 + (1 << 24), True)


@pytest.mark.parametrize("passes, acc, overflow", [(255, 510 << 14, False), (0, -(1 << 23), True)])
def test_addp_in_a_djnz_loop_reaches_the_edge_of_acc(passes, acc, overflow):
    """R3 = 0 gives 256 passes: 512 products of 2^14 are 2^23, one more than ACC can hold."""
    cpu = run(f"LDI R0, 0x80\nMUL R0, R0\nLDI R3, {passes}\nloop: ADDP\nADDP\nDJNZ loop\nHALT")
    assert (cpu.acc, cpu.overflow, cpu.instructions) == (acc, overflow, 4 + 3 * (passes or 256))


def test_counters_and_regions():
    cpu = run("NOP\n.region setup\nLDI R3, 3\n.region loop\nloop: NOP\nDJNZ loop\n.region end\nMAC #9\nHALT")
    assert (cpu.cycles, cpu.instructions) == (49, 10)
    assert cpu.insn_cycles == {"NOP": 16, "LDI": 4, "DJNZ": 12, "MAC": 13, "HALT": 4}
    assert cpu.region_cycles == {None: 4, "setup": 4, "loop": 24, "end": 17}
    assert sum(cpu.insn_cycles.values()) == sum(cpu.region_cycles.values()) == cpu.cycles


def test_memory_sizes_are_parameters():
    w = [9] + [0] * 0x2FF + [7]
    text = "SETP1 0x300\nLD R0, [P1]\nHALT"
    assert run(text, w=w, w_size=4096).r[0] == 7         # a 4096-cell memory sees all 12 pointer bits
    assert run(text, w=w[:256]).r[0] == 9                # the 256-cell memory sees P1[7:0] = 0
    assert [len(m) for m in LS([], x_size=64, w_size=1024).mem] == [64, 1024]
    assert LS([], np.array([-128, 127], dtype=np.int8)).mem[0][:3] == [-128, 127, 0]


def test_errors():
    with pytest.raises(RuntimeError, match="PC = 0x001 is outside the program"):
        run("NOP")
    cpu = LS(assemble("loop: JMP loop").words)
    with pytest.raises(RuntimeError, match="no HALT after 100 instructions"):
        cpu.run(limit=100)
    assert cpu.instructions == 100
    cpu = LS(assemble("NOP\nNOP\nHALT").words)
    cpu.run(limit=3)                                     # exactly `limit` instructions are allowed
    assert cpu.halted
    with pytest.raises(ValueError, match="0x00F0 is not an LS instruction"):
        LS([0x00F0])
    with pytest.raises(ValueError, match="12-bit program counter"):
        LS([0] * 4097)
    for bad in (dict(x=[128]), dict(w=[-129]), dict(w=[0.5]), dict(x=[0] * 257), dict(x_size=100), dict(w_size=0)):
        with pytest.raises(ValueError):
            LS([], **bad)
