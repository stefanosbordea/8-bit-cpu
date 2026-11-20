"""Turn one Python source file into corpus records: one record per short function or method, in canonical form."""
import ast
import re

from lsllm.canon import canon

MIN_LINES, MAX_LINES, MAX_CHARS = 2, 40, 1500
MARKERS = ("<NL>", "<IND>", "<DED", "<LF>")   # structure markers of the tokenizer, must not occur in corpus text
PY2_NAMES = {"xrange", "raw_input", "unicode", "basestring", "unichr"}  # builtins that only Python 2 has
PY2_ATTRS = {"iteritems", "itervalues", "iterkeys", "has_key"}          # dict methods that only Python 2 has
_TESTS = re.compile(r"(^|[/_.])(tests?|testing|conftest)([/_.]|$)")
_VENDORED = re.compile(r"(^|/)_?(vendor|vendored|third_party|thirdparty|extern|external|externals)/")


def skip_path(path):
    """Why the file at `path` is not read: "test" file, "vendored" copy of another project, or None (read it)."""
    path = path.lower()
    if _TESTS.search(path):
        return "test"
    if _VENDORED.search(path):
        return "vendored"       # code copied in from another project may carry another licence
    return None


def stable_canon(src):
    """canon(src), applied again until the text no longer changes; None if canon refuses it or it never settles.

    One pass is not always enough: some docstrings (first line indented deeper than the rest, or a last line of
    spaces only) change once more on a second pass. Corpus text must satisfy canon(text) == text.
    """
    text = canon(src)
    for _ in range(2):
        again = canon(text) if text else None
        if again == text:
            return text
        text = again
    return None


def _is_python2(node):
    """True if the function uses a builtin or a dict method that only Python 2 has (such code can still parse)."""
    for part in ast.walk(node):
        if isinstance(part, ast.Name) and part.id in PY2_NAMES or isinstance(part, ast.Attribute) and part.attr in PY2_ATTRS:
            return True
    return False


def _check(node):
    """Return (verdict, canonical text) of one function; the verdict is "extracted" or the reason for dropping it."""
    if node.name.startswith("test_"):
        return "test_function", None
    if _is_python2(node):
        return "python2", None
    if sum(isinstance(part, ast.stmt) for part in ast.walk(node)) > MAX_LINES:
        return "lines", None                # every statement takes at least one line, so canon need not run
    try:
        text = stable_canon(ast.unparse(node))
    except RecursionError:
        text = None
    if text is None:
        return "not_canonical", None        # canon() refuses it, almost always because of non-ASCII characters
    if not MIN_LINES <= text.count("\n") + 1 <= MAX_LINES:
        return "lines", None
    if len(text) > MAX_CHARS:
        return "chars", None
    if any(marker in text for marker in MARKERS):
        return "marker", None
    return "extracted", text


def extract(source, counts):
    """Return the records of one source file (text or bytes) and add what happened to each function to `counts`.

    Top-level functions give kind "function"; methods of top-level classes give kind "method", written as a
    standalone `def` at column 0. Nested functions stay inside the text of their parent. `async def` is not taken.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        counts["file_not_python_3_12"] += 1
        return []
    found = [("function", node) for node in tree.body]
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            found += [("method", node) for node in cls.body]
    records = []
    for kind, node in found:
        if isinstance(node, ast.AsyncFunctionDef):
            counts["async"] += 1
        if not isinstance(node, ast.FunctionDef):
            continue
        verdict, text = _check(node)
        counts[verdict] += 1
        if text:
            records.append({"text": text, "name": node.name, "kind": kind, "n_lines": text.count("\n") + 1,
                            "has_docstring": ast.get_docstring(node) is not None})
    return records
