"""The file formats of the export, each with its writer and its reader.

.hex   one value per line in hex digits, two's complement, row-major: 2 digits per byte, 4 per program word.
       Readable by `$readmemh` and by a VHDL `hread` loop.
.coe   the same values for the Vivado block memory generator:
           memory_initialization_radix=16;
           memory_initialization_vector=
           00,
           7F;
.vhd   one package `ls_tables_pkg` (VHDL-2008) with a constant per small table and per shift and preload.
.txt   test vectors: a line that starts with a letter is `name size size ...` (`name 1` for a single number),
       the lines after it hold the values in decimal, row-major, 16 per line.
"""
import re

import numpy as np


def digits_of(values, digits):
    mask = (1 << 4 * digits) - 1
    return ["%0*X" % (digits, int(v) & mask) for v in np.ravel(values)]


def write_hex(path, values, digits=2):
    with open(path, "w") as f:
        f.write("\n".join(digits_of(values, digits)) + "\n")


def read_hex(path):
    """The values of a .hex file as unsigned integers."""
    with open(path) as f:
        return np.array([int(line, 16) for line in f], dtype=np.int64)


def write_coe(path, values, digits=2):
    with open(path, "w") as f:
        f.write("memory_initialization_radix=16;\nmemory_initialization_vector=\n")
        f.write(",\n".join(digits_of(values, digits)) + ";\n")


def write_memory(directory, name, values, digits):
    """<name>.hex and <name>.coe of one memory."""
    write_hex("%s/%s.hex" % (directory, name), values, digits)
    write_coe("%s/%s.coe" % (directory, name), values, digits)


def read_coe(path):
    with open(path) as f:
        radix, vector = f.read().split(";")[:2]
    if radix.strip() != "memory_initialization_radix=16" or not vector.strip().startswith("memory_initialization_vector="):
        raise ValueError(path + " is not a .coe file of radix 16")
    return np.array([int(v, 16) for v in vector.split("=")[1].split(",")], dtype=np.int64)


# ---- VHDL ----

VHDL_HEAD = """-- Small tables and constants of the integer model %s.
-- Written by `python -m lsllm.export`; do not edit by hand. VHDL-2008.
-- A table of 256 entries is indexed by the UNSIGNED byte of the int8 value (docs/SPEC.md Part E):
-- index 0..127 = values 0..127, index 128..255 = values -128..-1.
-- <NAME>_TABLE holds the same numbers as weights/<NAME>.hex; the rows of a matrix are its first index.
package ls_tables_pkg is
  subtype byte_t is integer range -128 to 127;
  subtype ubyte_t is integer range 0 to 255;
  type byte_table_t is array (0 to 255) of byte_t;
  type ubyte_table_t is array (0 to 255) of ubyte_t;
  type byte_matrix_t is array (natural range <>, natural range <>) of byte_t;
  type ubyte_matrix_t is array (natural range <>, natural range <>) of ubyte_t;
  type int_vector_t is array (natural range <>) of integer;
  type int_matrix_t is array (natural range <>, natural range <>) of integer;
"""


def aggregate(value, indent="    "):
    """A VHDL aggregate, 16 numbers per line. A single element needs its index (`(0 => x)`): `(x)` is only x in brackets."""
    if value.ndim == 1:
        items = [str(int(v)) for v in value]
        parts = [", ".join(items[k:k + 16]) for k in range(0, len(items), 16)]
    else:
        parts = [aggregate(v, indent + " ") for v in value]
    return "(" + ("0 => " if len(value) == 1 else "") + (",\n" + indent).join(parts) + ")"


def constant(name, value, kind):
    """One constant declaration. kind: "byte" (int8), "ubyte" (0..255) or "int" (a number of model.json)."""
    value = np.asarray(value)
    if value.ndim == 0:
        return "  constant %s : integer := %d;" % (name, value)
    if kind != "int" and value.shape == (256,):
        return "  constant %s : %s_table_t :=\n    %s;" % (name, kind, aggregate(value))
    if kind != "int":
        value = value.reshape(len(value) if value.ndim > 1 else 1, -1)       # every other table is a matrix
    bounds = ", ".join("0 to %d" % (n - 1) for n in value.shape)
    return "  constant %s : %s_%s_t(%s) :=\n    %s;" % (name, kind, "vector" if value.ndim == 1 else "matrix", bounds, aggregate(value))


def vhdl(title, tables, numbers):
    """The text of ls_tables_pkg.vhd. tables: name -> int8 or uint8 array; numbers: name -> int or nested list."""
    lines = [constant(name.upper(), value, "int") for name, value in numbers.items()]
    lines += [constant(name + "_TABLE", value, "ubyte" if value.dtype == np.uint8 else "byte") for name, value in tables.items()]
    return VHDL_HEAD % title + "\n".join(lines) + "\nend package ls_tables_pkg;\n"


def read_vhdl(text):
    """name -> integer array of every constant of a package written by `vhdl` (a reader for that layout only)."""
    out = {}
    for name, kind, bounds, body in re.findall(r"constant (\w+) : (\w+?)(?:\(([^)]*)\))? :=(.*?);", text, re.S):
        values = np.array([int(v) for v in re.findall(r"-?\d+", body.replace("0 =>", ""))])
        shape = [int(top) + 1 for top in re.findall(r"0 to (\d+)", bounds)]
        out[name] = values[0] if kind == "integer" else values.reshape(shape or -1)
    return out


# ---- test vectors ----

def vector_text(values):
    """The .txt form of name -> integer array."""
    lines = []
    for name, value in values.items():
        value = np.asarray(value)
        flat = value.ravel().tolist()
        lines.append("%s %s" % (name, " ".join(map(str, value.shape)) or "1"))
        lines += [" ".join(map(str, flat[k:k + 16])) for k in range(0, len(flat), 16)]
    return "\n".join(lines) + "\n"


def read_vector_text(text):
    """name -> integer array (a single number comes back with shape (1,))."""
    out, name = {}, None
    for line in text.splitlines():
        if line[0].isalpha():
            name, *shape = line.split()
            out[name] = ([int(n) for n in shape], [])
        else:
            out[name][1].extend(int(v) for v in line.split())
    return {name: np.array(values, dtype=np.int64).reshape(shape) for name, (shape, values) in out.items()}
