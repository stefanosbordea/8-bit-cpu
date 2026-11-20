"""Canonical form of Python source. Used for the corpus, the benchmark prompts and the custom tasks,
so that the model always sees one formatting style.

canon(src) parses the code and writes it back with `ast.unparse` (4-space indentation, one statement per line,
no comments), cleans docstrings, removes blank lines outside strings and indents the inner lines of multi-line
docstrings to their block. It returns None when the source cannot be used.
"""
import ast
import inspect
import io
import tokenize

_DOC_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_ALLOWED = set(chr(c) for c in range(32, 127)) | {"\n"}


def _clean_docstrings(tree):
    """Dedent every docstring and strip trailing spaces from its lines."""
    for node in ast.walk(tree):
        if not isinstance(node, _DOC_OWNERS) or not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            lines = inspect.cleandoc(first.value.value).split("\n")
            first.value.value = "\n".join(line.rstrip() for line in lines)


def _layout(text):
    """Drop blank lines outside strings; indent inner lines of multi-line strings to their block."""
    lines = text.split("\n")
    inner = {}  # physical line index -> indentation of the line where the string starts
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
            start = lines[tok.start[0] - 1]
            base = " " * (len(start) - len(start.lstrip(" ")))
            for row in range(tok.start[0], tok.end[0]):
                inner[row] = base
    out = []
    for row, line in enumerate(lines):
        if row in inner:
            out.append(inner[row] + line if line.strip() else "")
        elif line.strip():
            out.append(line.rstrip())
    return "\n".join(out)


def canon(src):
    """Return the canonical form of `src`, or None if it does not parse or is not plain ASCII."""
    try:
        tree = ast.parse(src)
        _clean_docstrings(tree)
        out = _layout(ast.unparse(tree))
        ast.parse(out)
    except (SyntaxError, ValueError, RecursionError, MemoryError, tokenize.TokenError, IndentationError):
        return None
    if not out or not set(out) <= _ALLOWED:
        return None
    return out
