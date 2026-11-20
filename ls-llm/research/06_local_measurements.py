"""Local measurements quoted in research/06_constrained_decoding.md (research script, not part of the pipeline).

Proxy corpus: every module-level function and method of the CPython standard library, in canonical form
(lsllm.canon). It is NOT the training corpus. Nothing here executes the code it reads or generates:
validity is checked with compile() only.

Run from the project root:  .venv/bin/python research/06_local_measurements.py 2>/dev/null
"""
import ast
import collections
import os
import random
import sys
import sysconfig
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sentencepiece as spm  # noqa: E402
from lsllm.canon import canon  # noqa: E402

warnings.simplefilter("ignore")  # SyntaxWarning from odd escapes in generated text
PAIR = {"(": ")", "[": "]", "{": "}"}
QUOTES = ("'", '"', "'''", '"""')
HEAD = {"if", "elif", "else", "for", "while", "def", "class", "try", "except", "finally", "with", "async"}
SOFT = {"match", "case"}     # soft keywords: may open a block, but are also legal names
NLCHAR = chr(0x23CE)         # stands for "\n" in SentencePiece input (one function per line)
SAME, OPEN = 0xE000, 0xE010  # chr(SAME + k): newline + k dedents.  chr(OPEN): colon + newline + one indent
BUDGET = 200                 # token budget per generated function in part E


def corpus():
    """Canonical text of the standard-library functions (decorators removed, at least 3 lines)."""
    skip = {"test", "tests", "site-packages", "idlelib", "lib2to3", "__pycache__", "turtledemo"}
    files = []
    for root, dirs, names in os.walk(sysconfig.get_paths()["stdlib"]):
        dirs[:] = [d for d in dirs if d not in skip]
        files += [os.path.join(root, n) for n in names if n.endswith(".py")]
    files = sorted(f for f in files if os.path.getsize(f) < 150_000)
    funcs = []
    for path in files:
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, ValueError):
            continue
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef):
                nodes += [n for n in cls.body if isinstance(n, ast.FunctionDef)]
        for node in nodes:
            node.decorator_list = []
            text = canon(ast.unparse(node))
            if text and text.count("\n") >= 2:
                funcs.append(text + "\n")
    return funcs, len(files)


def scan(text):
    """Character-level state machine: string mode, bracket stack, line starts.

    Returns (mode, stack, rows), or None when the text breaks a canonical-form rule (mismatched bracket,
    newline inside brackets or inside a one-line string, comment). rows holds one [start, in_string, max_depth]
    per physical line; in_string says that the line starts inside a string.
    """
    mode, stack, rows, g, n = None, [], [[0, False, 0]], 0, len(text)
    while g < n:
        c = text[g]
        if mode is None:
            if c in "'\"":
                mode = text[g:g + 3] if text[g:g + 3] in QUOTES else c
                g += len(mode)
                continue
            if c in PAIR:
                stack.append(c)
                rows[-1][2] = max(rows[-1][2], len(stack))
            elif c in ")]}":
                if not stack or PAIR[stack[-1]] != c:
                    return None
                stack.pop()
            elif c == "#" or (c == "\n" and stack):
                return None
        elif c == "\\":
            g += 2
            continue
        elif text.startswith(mode, g):
            g += len(mode)
            mode = None
            continue
        elif c == "\n" and len(mode) == 1:
            return None
        if c == "\n":
            rows.append([g + 1, mode is not None, 0])
        g += 1
    return mode, stack, rows


def code_lines(text):
    """(indent, max bracket depth, line, ends with a block colon) for every line that starts outside a string."""
    result = scan(text)
    if result is None:
        return []
    rows = result[2]
    return [(len(line) - len(line.lstrip(" ")), rows[g][2], line, line.endswith(":") and not rows[g + 1][1])
            for g, line in enumerate(text.split("\n")[:-1]) if not rows[g][1]]


def ok(src):
    """True if `src` compiles (nothing is executed)."""
    try:
        compile(src, "<generated>", "exec")
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


# ---------------------------------------------------------------- A: canonical-form invariants

def part_a(funcs):
    st, depth_hist, level_hist = collections.Counter(), collections.Counter(), collections.Counter()
    for text in funcs:
        result = scan(text)
        if result is None or result[0] is not None or result[1]:
            st["functions rejected by the character machine"] += 1
            continue
        before = None
        for indent, depth, line, colon in code_lines(text):
            st["code lines"] += 1
            depth_hist[depth] += 1
            level_hist[indent // 4] += 1
            header = line.strip().split(" ")[0].rstrip(":") in HEAD
            st["lines ending with a colon (block headers)"] += colon
            st["block keyword first but no colon at the end"] += header and not colon
            st["colon at the end but no block keyword first"] += colon and not header
            st["indentation not a multiple of 4"] += indent % 4 != 0
            if before is not None:
                st["indent step other than +4 after a colon"] += before[1] and indent != before[0] + 4
                st["indent grows without a colon before"] += not before[1] and indent > before[0]
            before = (indent, colon)
        st["lines inside multi-line strings"] += sum(row[1] for row in result[2])
    print("A. canonical-form invariants on", len(funcs), "functions,", sum(map(len, funcs)), "characters")
    for key in sorted(st):
        print(f"   {key:48s} {st[key]}")
    for name, hist in (("max bracket depth per code line", depth_hist), ("indent level per code line", level_hist)):
        total = sum(hist.values())
        print(f"   {name}:", {k: f"{100 * v / total:.3f}%" for k, v in sorted(hist.items())})


# ---------------------------------------------------------------- B: what to do when the token budget runs out

def close_blocks(kept, fix_try):
    """Make a line-complete prefix compile: `pass` after a dangling header, a handler for an open `try`."""
    lines = code_lines(kept)
    if not lines:
        return None
    out = kept
    if lines[-1][3]:
        out += " " * (lines[-1][0] + 4) + "pass\n"
    if fix_try:
        open_try = []  # indentation of every `try` that has no except/finally yet
        for indent, _, line, _ in lines:
            word = line.strip().split(" ")[0].rstrip(":")
            while open_try and (open_try[-1] > indent or (open_try[-1] == indent and word not in ("except", "finally"))):
                open_try.pop()
            if word in ("except", "finally") and open_try and open_try[-1] == indent:
                open_try.pop()
            if word == "try":
                open_try.append(indent)
        for indent in reversed(open_try):
            out += " " * indent + "except Exception:\n" + " " * (indent + 4) + "pass\n"
    return out


def repair(text):
    """Cut `text` back to the last complete code line, then close blocks. Returns None if nothing is left."""
    result = scan(text)
    if result is None:
        return None
    return close_blocks(text[:max(row[0] for row in result[2] if not row[1])], True)


def part_b(funcs):
    rng = random.Random(0)
    res = collections.Counter()
    for text in rng.sample(funcs, 3000):
        for _ in range(3):
            pre = text[:rng.randrange(text.index("\n") + 1, len(text))]
            mode, stack, rows = scan(pre)
            res["prefixes"] += 1
            res["cut in the middle of a line"] += not pre.endswith("\n")
            res["cut inside a string"] += mode is not None
            res["cut inside brackets"] += bool(stack)
            res["valid as it is"] += ok(pre)
            in_place = pre + (mode or "") + "".join(PAIR[b] for b in reversed(stack))
            in_place += "" if in_place.endswith("\n") else "\n"
            res["valid: close string and brackets in place, close blocks"] += ok(close_blocks(in_place, True) or "(")
            kept = pre[:max(row[0] for row in rows if not row[1])]
            res["valid: drop the unfinished line, add pass"] += ok(close_blocks(kept, False) or "(")
            res["valid: drop the unfinished line, add pass, close try"] += ok(close_blocks(kept, True) or "(")
    print("B. truncation at a random character (3000 functions x 3 cuts, seed 0)")
    for key, value in res.items():
        print(f"   {key:58s} {value:5d}  {100 * value / res['prefixes']:6.2f}%")


# ---------------------------------------------------------------- C/D: vocabularies

def to_structure(text):
    """Replace every newline + indentation by one marker character (SAME + dedents, or OPEN)."""
    out, mode, level, base = [], None, 0, 0
    for row, line in enumerate(text.split("\n")[:-1]):
        body = line
        if row > 0 and mode is None:
            new = (len(line) - len(line.lstrip(" "))) // 4
            if out[-1].endswith(":") and new == level + 1:
                out[-1] = out[-1][:-1]
                out.append(chr(OPEN))
            else:
                out.append(chr(SAME + level - new))
            level, body = new, line[4 * new:]
        elif row > 0:
            out.append(chr(SAME))
            body = line[base:]
        if mode is None:
            base = 4 * level
        mode = scan((mode or "") + body)[0]  # a line that starts inside a string: reopen it for the scanner
        out.append(body)
    out.append(chr(SAME + level))
    return "".join(out)


def from_structure(s):
    """Inverse of to_structure."""
    out, level, mode, base, g, n = [], 0, None, 0, 0, len(s)
    while g < n:
        c = s[g]
        if ord(c) == OPEN:
            level += 1
            base = 4 * level
            out.append(":\n" + " " * base)
        elif SAME <= ord(c) < SAME + 10 and mode is None:
            level -= ord(c) - SAME
            base = 4 * level
            out.append("\n" + (" " * base if g < n - 1 else ""))
        elif SAME <= ord(c) < SAME + 10:
            blank = g + 1 >= n or ord(s[g + 1]) >= SAME
            out.append("\n" + ("" if blank else " " * base))
        elif mode is None and c in "'\"":
            mode = s[g:g + 3] if s[g:g + 3] in QUOTES else c
            out.append(mode)
            g += len(mode) - 1
        elif mode is not None and c == "\\":
            out.append(s[g:g + 2])
            g += 1
        elif mode is not None and s.startswith(mode, g):
            out.append(mode)
            g += len(mode) - 1
            mode = None
        else:
            out.append(c)
        g += 1
    return "".join(out)


def train(folder, tag, vocab, lines, user, split_ws=False):
    """Train a SentencePiece BPE model on `lines` (one function per line); return the processor."""
    path = os.path.join(folder, tag + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    spm.SentencePieceTrainer.train(
        input=path, model_prefix=os.path.join(folder, tag), model_type="bpe", vocab_size=vocab,
        character_coverage=1.0, normalization_rule_name="identity", remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True, split_digits=True, split_by_whitespace=split_ws, add_dummy_prefix=False,
        max_sentence_length=1_000_000, num_threads=2, pad_id=-1, user_defined_symbols=user)
    return spm.SentencePieceProcessor(model_file=os.path.join(folder, tag + ".model"))


def pieces_of(sp):
    """Text of every piece (None for control and unknown tokens)."""
    return [None if sp.is_control(g) or sp.is_unknown(g) else sp.id_to_piece(g).replace(chr(0x2581), " ")
            for g in range(sp.get_piece_size())]


def part_c(funcs, folder):
    raw = [text.replace("\n", NLCHAR) for text in funcs]
    marked = [to_structure(text) for text in funcs]
    bad = sum(from_structure(marked[g]) != funcs[g] for g in range(len(funcs)))
    marks = collections.Counter(c for line in marked for c in line if ord(c) >= SAME)
    print("C. vocabularies (SentencePiece", spm.__version__, "BPE; token counts on every 4th function)")
    print("   structural line ends: round-trip failures", bad, "of", len(funcs), "| characters",
          sum(map(len, funcs)), "->", sum(map(len, marked)))
    print("   line-end tokens:", {("OPEN" if ord(k) == OPEN else f"NL-{ord(k) - SAME}"): v for k, v in sorted(marks.items())})
    structural = set("()[]{}'\"" + NLCHAR)
    brackets_quotes = list("()[]{}") + list(QUOTES)
    line_ends = [chr(SAME + k) for k in range(10)] + [chr(OPEN)]
    for vocab in (512, 256):
        variants = (
            ("free BPE, split at spaces (llama2.c settings)", raw, [], True),
            ("free BPE, pieces may span spaces", raw, [], False),
            ("brackets, quotes, newline as single tokens", raw, brackets_quotes + [NLCHAR], False),
            ("structural line ends + free BPE", marked, line_ends, False),
            ("structural line ends + brackets, quotes single", marked, line_ends + brackets_quotes, False),
        )
        base = None
        for number, (name, lines, user, split_ws) in enumerate(variants):
            sp = train(folder, f"v{vocab}_{number}", vocab, lines, user, split_ws)
            pieces = pieces_of(sp)
            bridge = [g for g, p in enumerate(pieces) if p and len(p) > 1 and p not in QUOTES and set(p) & structural]
            count = collections.Counter(t for line in lines[::4] for t in sp.encode(line))
            total = sum(count.values())
            base = base or total
            top = sorted(bridge, key=lambda g: -count[g])[:8]
            print(f"   V={vocab} {name:46s} tokens {total:7d} ({100 * (total / base - 1):+5.1f}%)  chars/token "
                  f"{sum(map(len, funcs[::4])) / total:.3f}  bridge pieces {len(bridge):2d} = "
                  f"{100 * sum(count[g] for g in bridge) / total:4.1f}% of tokens",
                  [pieces[g].replace(NLCHAR, "\n") for g in top] if number == 0 else "")
    return train(folder, "final", 512, marked, line_ends + brackets_quotes), marked


# ---------------------------------------------------------------- E: the token-level filter

class Filter:
    """The proposed filter, for a vocabulary whose brackets, quotes and line ends are single tokens."""

    def __init__(self, pieces, header_rule):
        self.pieces, self.header_rule = pieces, header_rule
        self.mode, self.esc, self.stack, self.level, self.line, self.done = None, False, [], 0, "", False
        self.special = sum(p is None for p in pieces)
        self.checked = [g for g, p in enumerate(pieces) if p and (p in ")]}" or ord(p[0]) >= SAME)]

    def allowed(self, tok):
        p = self.pieces[tok]
        if p is None:
            return False
        if len(p) == 1 and ord(p) >= SAME:                       # a line-end token
            if self.mode is not None:
                return len(self.mode) == 3 and ord(p) == SAME    # plain newline inside a docstring only
            if self.stack:
                return False
            first = self.line.split(" ")[0]
            if ord(p) == OPEN:
                return first in HEAD or first in SOFT or not self.header_rule
            return ord(p) - SAME <= self.level and not (self.header_rule and first in HEAD)
        if self.mode is None and p in ")]}":
            return bool(self.stack) and PAIR[self.stack[-1]] == p
        return True

    def masked(self):
        """Number of tokens the filter forbids in the current state (control tokens included)."""
        return self.special + sum(not self.allowed(t) for t in self.checked)

    def push(self, tok):
        p = self.pieces[tok]
        if len(p) == 1 and ord(p) >= SAME:
            if self.mode is None:
                self.level += 1 if ord(p) == OPEN else SAME - ord(p)
                self.line = ""
                self.done = self.level <= 0
            self.esc = False
        elif self.mode is None:
            if p in PAIR:
                self.stack.append(p)
            elif p in ")]}" and self.stack:
                self.stack.pop()
            elif p in QUOTES:
                self.mode = p
            self.line += p
        elif p == self.mode and not self.esc:
            self.mode = None
        else:                                                    # string content: track a pending backslash
            run = len(p) - len(p.rstrip("\\"))
            self.esc = (run - (self.esc and run == len(p))) % 2 == 1


def generate(counts, pieces, rng, use_mask, header_rule, power):
    """Sample one function from the trigram counts. Returns (tokens, finished, masked probability per step, back-offs)."""
    flt = Filter(pieces, header_rule)
    a = b = -1
    out, mass, backoff = [], [], 0
    while not flt.done and len(out) < BUDGET:
        for context in ((a, b), (b,), ()):
            cand = [(t, c ** power) for t, c in counts[context].items()]
            good = [(t, w) for t, w in cand if not use_mask or flt.allowed(t)]
            if context == (a, b) and cand:
                mass.append(1 - sum(w for _, w in good) / sum(w for _, w in cand))
            if good:
                break
            backoff += 1
        tok = rng.choices([t for t, _ in good], [w for _, w in good])[0]
        flt.push(tok)
        out.append(tok)
        a, b = b, tok
    return out, flt.done, mass, backoff


def part_e(sp, marked):
    pieces = pieces_of(sp)
    seqs = [sp.encode(line) for line in marked]
    rejected, masked_rows, steps = 0, 0, 0
    for number, seq in enumerate(seqs):
        flt = Filter(pieces, True)
        for tok in seq:
            if not flt.allowed(tok):
                rejected += 1
                break
            if number % 4 == 0:
                masked_rows += flt.masked()
                steps += 1
            flt.push(tok)
        else:
            rejected += not flt.done
    print("E. token-level filter with the V=512 'structural line ends + brackets, quotes single' vocabulary")
    print(f"   corpus functions rejected by the filter (header rule on): {rejected} of {len(seqs)} "
          f"({sum(map(len, seqs))} tokens, mean {sum(map(len, seqs)) / len(seqs):.0f} per function)")
    print(f"   output rows the filter forbids, average over corpus states: {masked_rows / steps:.1f} of {len(pieces)}")
    counts = collections.defaultdict(collections.Counter)
    for seq in seqs:
        a = b = -1
        for tok in seq:
            counts[(a, b)][tok] += 1
            counts[(b,)][tok] += 1
            counts[()][tok] += 1
            a, b = b, tok
    print("   token-trigram stand-in generator (NOT our model), 2000 samples per row, budget", BUDGET, "tokens")
    for power, label in ((1.0, "temperature 1.0"), (2.0, "temperature 0.5")):
        for name, use_mask, header_rule in (("no filter", False, False), ("filter", True, False),
                                            ("filter + header rule", True, True)):
            rng = random.Random(0)
            res, errors, mass_sum, mass_n = collections.Counter(), collections.Counter(), 0.0, 0
            for _ in range(2000):
                out, finished, mass, backoff = generate(counts, pieces, rng, use_mask, header_rule, power)
                text = from_structure("".join(pieces[t] for t in out))
                res["finished"] += finished
                res["valid"] += ok(text)
                fixed = text if finished else repair(text)
                res["valid after repair"] += bool(fixed) and ok(fixed)
                res["backoff"] += backoff
                res["tokens"] += len(out)
                mass_sum += sum(mass)
                mass_n += len(mass)
                if fixed and not ok(fixed):
                    try:
                        compile(fixed, "<generated>", "exec")
                    except (SyntaxError, ValueError) as err:
                        errors[str(getattr(err, "msg", err))[:38]] += 1
            print(f"   {label}, {name:21s} finished {res['finished'] / 20:5.1f}%  compiles {res['valid'] / 20:5.1f}%  "
                  f"compiles after truncation repair {res['valid after repair'] / 20:5.1f}%  "
                  f"probability on masked tokens {100 * mass_sum / mass_n:4.1f}%  "
                  f"steps where every trigram candidate was masked {100 * res['backoff'] / res['tokens']:.2f}%")
            print("      most frequent remaining errors:", errors.most_common(4))


if __name__ == "__main__":
    functions, n_files = corpus()
    print("proxy corpus:", len(functions), "functions from", n_files, "standard-library files, Python", sys.version.split()[0])
    part_a(functions)
    part_b(functions)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        processor, marked_lines = part_c(functions, tmp)
        part_e(processor, marked_lines)
