"""Is a function self-contained? It uses only its parameters, its own variables, Python built-ins and a few
standard-library modules, like the functions of the benchmarks. Used to choose training functions.

    free_names(text)      names the function reads that it does not define itself
    self_contained(text)  True if all of them are built-ins or allowed modules, and nothing ties it to a project
"""
import ast
import builtins

BUILTINS = set(dir(builtins))
MODULES = {"math", "re", "string", "itertools", "functools", "collections", "operator", "heapq", "bisect", "random",
           "cmath", "fractions", "decimal", "statistics", "copy", "typing"}
TYPING = {"List", "Dict", "Tuple", "Set", "Optional", "Any", "Union", "Iterable", "Iterator", "Sequence", "Callable",
          "Generator", "Mapping", "Deque", "Counter", "defaultdict", "deque", "OrderedDict", "namedtuple", "reduce",
          "lru_cache", "sqrt", "floor", "ceil", "gcd", "pi", "inf", "log", "factorial", "isqrt", "Fraction", "Decimal",
          "permutations", "combinations", "product", "accumulate", "chain", "groupby", "heappush", "heappop", "heapify"}


def free_names(text):
    """Names read by the first function of `text` that are not its parameters, variables or nested functions.
    None if the text is not exactly one top-level function."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    fn = tree.body[0]
    bound, read = {fn.name}, set()
    for node in ast.walk(fn):
        if isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name):
            (read if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not fn:
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
    return read - bound


def self_contained(text, modules=MODULES):
    """True for one top-level function without decorators, global statements, classes or imports inside, whose
    free names are all built-ins, allowed modules or common names imported from them."""
    names = free_names(text)
    if names is None:
        return False
    fn = ast.parse(text).body[0]
    if fn.decorator_list:
        return False
    for node in ast.walk(fn):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.Import, ast.ImportFrom, ast.AsyncFunctionDef, ast.Await,
                             ast.Yield, ast.YieldFrom)):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("print", "input", "open", "exec", "eval",
                                                                                            "globals", "locals", "__import__"):
            return False
    return all(name in BUILTINS or name in modules or name in TYPING for name in names)
