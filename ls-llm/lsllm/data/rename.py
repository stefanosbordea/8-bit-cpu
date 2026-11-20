"""Consistent renaming of the parameters and local variables of a function (data augmentation).

A mechanical transformation of existing code: the behaviour stays the same, only names change, in the code and
(as whole words) in the docstring. A model trained on several renamings of one function cannot remember which
names a body uses: it has to read them from the signature and the docstring.

    pool = name_pool(texts)                   # names that functions of the corpus really use, with weights
    new = rename(text, random.Random(0), pool)    # None if the function is left alone

The result is checked: it must parse, and its name-independent fingerprint (lsllm.alpha) must equal the one of
the original, so a renaming that would change the code structure is never returned.
"""
import ast
import builtins
import keyword
import re
from collections import Counter

from lsllm.alpha import alpha_key
from lsllm.canon import canon

# English function words: a parameter with such a name is not renamed when the docstring uses the word,
# because replacing it there would rewrite the sentence, and such a word is never chosen as a new name
STOP = set("a i an as at be by do he in is it me my no of on or so to up us we the and for not all any are but can "
           "has its one out use was if else from with than then that this each every two have into self cls".split())
RESERVED = set(keyword.kwlist) | set(keyword.softkwlist) | set(dir(builtins)) | STOP
UNSAFE_CALLS = {"locals", "globals", "vars", "eval", "exec", "dir", "getattr", "setattr", "delattr", "super"}
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _function(tree):
    return tree.body[0] if len(tree.body) == 1 and isinstance(tree.body[0], FUNCTIONS) else None


def _candidates(fn):
    """(parameters and local variables that may be renamed, every identifier the function uses) or None."""
    params, stored, fixed, used = [], [], {fn.name}, {fn.name}
    for node in ast.walk(fn):
        if isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.Match)):
            return None
        if isinstance(node, ast.arg):
            params.append(node.arg)
        elif isinstance(node, ast.Name):
            used.add(node.id)
            if isinstance(node.ctx, ast.Store):
                stored.append(node.id)
            elif node.id in UNSAFE_CALLS:
                return None
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            stored.append(node.name)
        elif isinstance(node, FUNCTIONS) and node is not fn:
            fixed.add(node.name)
        elif isinstance(node, ast.alias):
            fixed.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.keyword) and node.arg:
            fixed.add(node.arg)             # a keyword names a parameter of the called function: keep it as it is
    names = [n for n in dict.fromkeys(params + stored) if n not in fixed and n not in ("self", "cls") and not n.startswith("__")]
    return names, used | fixed | set(params) | set(stored)


def names_of(text):
    """(parameter names, local variable names) of the function in `text`; ([], []) if it cannot be read."""
    try:
        fn = _function(ast.parse(text))
    except (SyntaxError, ValueError, RecursionError):
        fn = None
    if fn is None:
        return [], []
    params = [node.arg for node in ast.walk(fn) if isinstance(node, ast.arg)]
    local = [node.id for node in ast.walk(fn) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)]
    return params, [n for n in dict.fromkeys(local) if n not in params]


def name_pool(texts, least=20, power=0.75):
    """(names, weights): the parameter and variable names used by at least `least` of the functions `texts`,
    weighted by count^power. Keywords, builtins and English function words are left out."""
    count = Counter()
    for text in texts:
        params, local = names_of(text)
        count.update(set(params) | set(local))
    names = sorted(n for n, c in count.items() if c >= least and n not in RESERVED and n.isidentifier() and not n.startswith("_"))
    return names, [count[n] ** power for n in names]


def rename(text, rng, pool, share=1.0):
    """A copy of the function `text` with new names for its parameters and local variables, or None.

    rng: random.Random. pool: (names, weights) from name_pool. share: probability that a given name is replaced.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    fn = _function(tree)
    found = fn and _candidates(fn)
    if not found or not found[0]:
        return None
    names, used = found
    doc = ast.get_docstring(fn, clean=False) or ""
    words = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", doc))
    mapping, taken = {}, set(used) | words
    for old in names:
        if (old in STOP and old in words) or rng.random() >= share:
            continue
        for attempt in range(20):
            new = rng.choices(pool[0], pool[1])[0]
            if new not in taken:
                mapping[old] = new
                taken.add(new)
                break
    if not mapping:
        return None
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            node.id = mapping.get(node.id, node.id)
        elif isinstance(node, ast.arg):
            node.arg = mapping.get(node.arg, node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            node.name = mapping.get(node.name, node.name)
    if doc:
        swap = lambda m: mapping.get(m.group(0), m.group(0))
        fn.body[0].value.value = re.sub(r"(?<![A-Za-z_0-9])[A-Za-z_][A-Za-z_0-9]*(?![A-Za-z_0-9])", swap, doc)
    try:
        out = canon(ast.unparse(tree))
    except (SyntaxError, ValueError, RecursionError):
        return None
    if not out or out == text or alpha_key(out) != alpha_key(text):
        return None
    return out
