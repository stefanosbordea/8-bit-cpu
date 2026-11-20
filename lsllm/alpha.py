"""Name-independent fingerprint of a function, used to detect overlap between training data and benchmarks.

Two functions get the same key when they are the same code up to renaming the function, its arguments and its
local variables, and ignoring the docstring. Example: `def add(a, b): return a + b` and
`def plus(x, y): return x + y` share one key.
"""
import ast
import hashlib


def _bound_names(fn):
    """Names bound inside the function: arguments, assigned variables, loop and comprehension variables."""
    args = fn.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    names += [a.arg for a in (args.vararg, args.kwarg) if a]
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.append(node.id)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
            names.append(node.name)
    order = {}
    for name in names:
        order.setdefault(name, "v%d" % len(order))
    return order


def alpha_key(src):
    """Return the fingerprint of the first function defined in `src`, or None if there is none."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError):
        return None
    fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return None
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        fn.body = body[1:] or [ast.Pass()]
    rename = _bound_names(fn)
    rename[fn.name] = "f"
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id in rename:
            node.id = rename[node.id]
        elif isinstance(node, ast.arg):
            node.arg = rename.get(node.arg, node.arg)
            node.annotation = None
        elif isinstance(node, ast.ExceptHandler) and node.name:
            node.name = rename.get(node.name, node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.name = rename.get(node.name, node.name)
            node.returns = None
            node.decorator_list = []
    return hashlib.sha1(ast.dump(fn).encode()).hexdigest()
