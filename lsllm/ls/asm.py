"""Assembler for the LS CPU, in the syntax of the listings of the design document (07-programs.tex).

One instruction per line, for example:    loop:   LD    R1, [P1]+     ; comment
    name:           a label: the address of the next instruction (alone on a line or in front of an instruction)
    NAME = value    a named constant: a number or a name defined earlier; `relu` is predefined as 1
    .region name    the instructions from here on belong to region `name` (the simulator reports cycles per region)
Numbers are decimal, 0x hex or 0b binary. Mnemonics and register names are capitals; names are case-sensitive.
Operands: R0..R3, P0/P1, [P0] [P1] with + for post-increment, #n for the MAC count, numbers and names.
"""
import re
from typing import NamedTuple

from lsllm.ls.isa import MASK12, TABLE, encode

# How an operand field of the instruction table is written. Fields that are not listed are numbers or names.
KIND = {"r1": "reg", "r2": "reg", "rd": "reg", "r": "reg",    # R0..R3
        "fn": "ptr",                                          # P0, P1 (MVP keeps its pointer in fn)
        "p": "mem", "inc": "inc",                             # [P0], [P1], with + for post-increment
        "n": "cnt"}                                           # #n
NAME = r"([A-Za-z_]\w*)"


class Program(NamedTuple):
    words: list      # one 16-bit word per instruction; the address of a word is its index
    listing: str     # every source line with the address and the word of its instruction
    regions: dict    # address -> name, from the `.region` lines


def _value(text, symbols):
    """A number or a defined name -> int."""
    if text in symbols:
        return symbols[text]
    try:
        return int(text, 0)
    except ValueError:
        raise ValueError(f"'{text}' is not a number or a defined name") from None


def _define(symbols, name, value):
    if name in symbols:
        raise ValueError(f"'{name}' is defined twice")
    symbols[name] = value


def _operand(text, symbols):
    """One operand as written -> list of (kind, value), one pair for each instruction field it fills."""
    if m := re.fullmatch(r"R([0-3])", text):
        return [("reg", int(m[1]))]
    if m := re.fullmatch(r"P([01])", text):
        return [("ptr", int(m[1]))]
    if m := re.fullmatch(r"\[P([01])\](\+?)", text):
        return [("mem", int(m[1])), ("inc", len(m[2]))]
    if text.startswith("#"):
        return [("cnt", _value(text[1:], symbols))]
    return [("num", _value(text, symbols))]


def _word(code, symbols):
    """One instruction as written, without label and comment -> its 16-bit word."""
    name, *rest = code.split(None, 1)
    operands = [text.strip() for text in rest[0].split(",")] if rest else []
    if name not in TABLE:
        raise ValueError(f"unknown instruction '{name}'")
    if name == "MAC" and not operands:
        operands = ["#0"]                 # MAC alone is the classic MAC, n = 0
    pairs = [pair for text in operands for pair in _operand(text, symbols)]
    if [kind for kind, _ in pairs] != [KIND.get(field, "num") for field in TABLE[name].args]:
        raise ValueError(f"wrong operands for {name}")
    return encode(name, *[value for _, value in pairs])


def assemble(text):
    """Assembly text -> Program(words, listing, regions). ValueError with the line number on any error."""
    symbols, regions, lines = {"relu": 1}, {}, []    # lines: (line number, source line, instruction or "")
    address = 0
    for number, source in enumerate(text.splitlines(), 1):
        code = source.split(";")[0].strip()
        try:
            if m := re.match(NAME + ":", code):
                _define(symbols, m[1], address)
                code = code[m.end():].strip()
            if m := re.fullmatch(NAME + r"\s*=\s*(\S+)", code):
                _define(symbols, m[1], _value(m[2], symbols))
                code = ""
            elif m := re.fullmatch(r"\.region\s+(\S+)", code):
                regions[address] = m[1]
                code = ""
        except ValueError as err:
            raise ValueError(f"line {number}: {err}") from None
        lines.append((number, source, code))
        address += bool(code)
    if address > MASK12 + 1:
        raise ValueError(f"{address} words do not fit the 12-bit program counter")
    words, listing = [], []
    for number, source, code in lines:
        if not code:
            listing.append(f"{'':11}{source}".rstrip())
            continue
        try:
            words.append(_word(code, symbols))
        except ValueError as err:
            raise ValueError(f"line {number}: {err}") from None
        listing.append(f"{len(words) - 1:03X}  {words[-1]:04X}  {source}")
    return Program(words, "\n".join(listing), regions)
