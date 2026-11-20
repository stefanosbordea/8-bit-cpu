"""Request suite ("asks"): classic programs a person asks for by name, each with tests.

data/asks/asks.jsonl       one ask per line: id, request, requests, name, kind, tests, category, notes, aliases
data/asks/solutions.jsonl  one verified human-written implementation per ask (written by lsllm.asks.mine)
data/asks/missing.jsonl    asks for which the corpus has no implementation that passes the tests

This is a recall test of programs the board model was trained on. It is not a benchmark.
"""
import ast
import builtins
import json
import re
from pathlib import Path

from lsllm.canon import canon

DATA = Path(__file__).resolve().parents[2] / "data" / "asks"
ASKS, SOLUTIONS, MISSING = DATA / "asks.jsonl", DATA / "solutions.jsonl", DATA / "missing.jsonl"
PHRASE_FILE = DATA / "phrases.jsonl"            # per ask: sentences people use for it (id, name, phrases); read by Resolver

# Words of a request that carry no meaning for the name. The list is fixed: a request maps to one name only.
FILLER = {"make", "me", "a", "an", "the", "please", "write", "program", "function", "code", "for", "that", "to",
          "give", "create", "implement", "in", "python", "can", "you", "i", "want", "need"}


def request_to_prompt(request):
    """The prompt the model continues for a request line: lower-case, split into words of letters and digits,
    drop the FILLER words, join the rest with underscores, and return "def <name>(".

    "Make me a bubble sort program" -> "def bubble_sort("

    The opening parenthesis ends the name: without it the model keeps extending the name.
    """
    words = [word for word in re.findall("[a-z0-9]+", request.lower()) if word not in FILLER]
    if not words:
        raise ValueError("the request %r has no word besides filler words" % request)
    return "def " + "_".join(words) + "("


class _NoHints(ast.NodeTransformer):
    """Removes parameter and return annotations; `x: int = 0` becomes `x = 0`, a bare `x: int` is dropped."""

    def visit_FunctionDef(self, node):
        node.returns = None
        return self.generic_visit(node)

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_AnnAssign(self, node):
        return ast.Assign(targets=[node.target], value=node.value, lineno=node.lineno) if node.value else None


def plain(text):
    """The form in which a corpus function is tested and in which the board model has to write it, made by
    mechanical steps only: the docstring is removed; the type hints are removed if they use a name that is not
    a built-in (`List[int]` without its import does not run alone; `list[int]` stays); canonical formatting.
    Returns (text, hints_removed); text is None if nothing but the docstring is left or it is not plain ASCII.
    The function is not renamed here (see rename)."""
    tree = ast.parse(text)
    function = tree.body[0]
    if ast.get_docstring(function, clean=False) is not None:
        function.body = function.body[1:]
        if not function.body:
            return None, False
    hints = [node.annotation for node in ast.walk(tree) if isinstance(node, (ast.arg, ast.AnnAssign)) and node.annotation]
    hints += [node.returns for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.returns]
    names = {node.id for hint in hints for node in ast.walk(hint) if isinstance(node, ast.Name)}
    hints_removed = not names <= set(dir(builtins))
    if hints_removed:
        tree = _NoHints().visit(tree)
    try:
        return canon(ast.unparse(tree)), hints_removed
    except (AttributeError, ValueError):           # a body left empty by a dropped bare annotation
        return None, hints_removed


def rename(text, own_name, name):
    """`text` with the identifier `own_name` (the function, calls of itself, a variable of the same name) replaced
    by `name`. Attributes are not touched: `array.pop()` inside a function called `pop` stays as it is."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == own_name:
            node.name = name
        elif isinstance(node, ast.Name) and node.id == own_name:
            node.id = name
        elif isinstance(node, ast.arg) and node.arg == own_name:
            node.arg = name
    return canon(ast.unparse(tree))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]



def _words(request):
    """The words of a request as the prompt rule sees them, with camelCase split first (isPrime -> is prime)."""
    request = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", request)
    return [word for word in re.findall("[a-z0-9]+", request.lower()) if word not in FILLER]


def _one_off(typed, word):
    """One letter missing, added, replaced or swapped with its neighbour, in words of 3 letters or more."""
    if min(len(typed), len(word)) < 3 or abs(len(typed) - len(word)) > 1:
        return False
    if len(typed) == len(word):
        differ = [g for g in range(len(word)) if typed[g] != word[g]]
        return len(differ) == 1 or (len(differ) == 2 and differ[1] == differ[0] + 1 and typed[differ[0]] == word[differ[1]] and typed[differ[1]] == word[differ[0]])
    short, long = sorted((typed, word), key=len)
    return any(long[:g] + long[g + 1:] == short for g in range(len(long)))


# Phrases that name a data structure, not a function: the ask that builds it. Written by hand; extend with care.
PHRASES = {"binary_search_tree": "bst_insert", "bst": "bst_insert", "binary_tree": "bst_insert", "stack": "stack_push", "queue": "enqueue"}
# Words that may stand around a name in a longer sentence without changing what is asked. Written by hand.
AROUND = {"list", "lists", "array", "arrays", "number", "numbers", "string", "strings", "integer", "integers", "value", "values",
          "element", "elements", "word", "words", "text", "sentence", "my", "of", "with", "using", "use", "algorithm", "two", "given",
          "and", "from", "it", "is", "if", "check", "find", "get", "compute", "calculate", "return", "sort", "do", "on", "by", "this",
          "some", "simple", "basic", "how", "whether", "all", "one", "which", "what", "show", "print", "example", "script",
          "generate", "display", "determine", "convert", "are", "user", "input", "entered", "takes", "any", "tell", "there", "does",
          "method", "task", "problem", "solution", "solve", "implementation", "version", "easy", "short", "small", "quick", "little"}
# One-word requests that the other names of the corpus send to an unusual ask: the usual meaning, by hand.
WORDS = {"add": "add_numbers", "addition": "add_numbers", "multiply": "multiply_numbers", "search": "linear_search", "find": "linear_search",
         "reverse": "reverse_string", "prime": "is_prime", "sqrt": "square_root", "factors": "divisors", "even_odd": "even_or_odd",
         "palindrome": "is_palindrome", "anagram": "is_anagram", "sorting": "bubble_sort", "sort": "bubble_sort", "heap": "build_min_heap",
         "tree": "bst_insert", "max": "find_max", "min": "find_min", "sum": "sum_list", "swap": "swap_numbers", "square": "square"}


def _typo(typed, word, changed=True):
    """True if `typed` is `word` with two neighbouring letters swapped (words of 4 letters or more), or with one
    letter missing, added or (if `changed`) replaced (5 letters or more: in a short word one changed letter gives
    another word)."""
    if typed == word or min(len(typed), len(word)) < 4 or abs(len(typed) - len(word)) > 1:
        return False
    if len(typed) == len(word):
        differ = [g for g in range(len(word)) if typed[g] != word[g]]
        swapped = len(differ) == 2 and differ[1] == differ[0] + 1 and typed[differ[0]] == word[differ[1]] and typed[differ[1]] == word[differ[0]]
        return swapped or (len(differ) == 1 and len(word) >= 5 and changed)
    short, long = sorted((typed, word), key=len)
    return len(short) >= 5 and any(long[:g] + long[g + 1:] == short for g in range(len(long)))


class Resolver:
    """Which ask a request line means, decided on the PC before anything is sent to the CPU (the model always gets
    `def <name>(` with the name of the ask). Rules, in this order:
      name    the words of the request are the name of an ask                   "make me a bubble sort program"
      alias   they are another name listed for an ask (if several asks list     "bubblesort", "binary search tree"
              it, the one that comes first in the suite), or a phrase of
              PHRASES
      part    a name stands inside the request, word by word (other names       "sort my list with bubble sort"
              count if they have two words or more), and every other word is
              one of AROUND; the longest name wins, a tie chooses nothing
      near    typing errors: the same number of words as a name or other name,  "buble sort", "bubble srot",
              each word equal or with one typing error (_typo); or, in a        "quik sort"
              request of several words, exactly one word one letter off
              (_one_off); and only one ask fits
    Each rule is tried on the words as typed and then with plural endings removed, before the next rule. A request
    that fits no rule is not in the suite; its words go to the model as they are."""

    def __init__(self, asks=None):
        asks = load_jsonl(ASKS) if asks is None else asks
        own = {ask["name"] for ask in asks}
        claims = {}                                     # other name -> the asks that list it, in the order of the suite
        for ask in asks:
            for alias in ask.get("aliases", []):
                key = "_".join(_words(alias.replace("_", " ")))
                if key and key not in own and ask["name"] not in claims.setdefault(key, []):
                    claims[key].append(ask["name"])
        self.names = {key: names[0] for key, names in claims.items()}
        self.ambiguous = sorted(key for key, names in claims.items() if len(names) > 1)
        self.names.update({key: name for key, name in PHRASES.items() if name in own})
        if PHRASE_FILE.exists():                        # ways people ask for a request (data/asks/phrases.jsonl): above the other names
            for row in load_jsonl(PHRASE_FILE):
                for phrase in row["phrases"]:
                    key = "_".join(_words(phrase))
                    if key and row["name"] in own:
                        self.names[key] = row["name"]
        self.names.update({key: name for key, name in WORDS.items() if name in own})
        self.names.update({name: name for name in own})
        self.parts = {tuple(key.split("_")): name for key, name in self.names.items() if "_" in key or key in own}
        self.by_length = {}                             # number of words -> [(words, name of the ask)]
        for key, name in self.names.items():
            self.by_length.setdefault(key.count("_") + 1, []).append((key.split("_"), name))

    def _exact(self, words):
        joined = "_".join(words)
        if joined in self.names:
            return self.names[joined], "name" if self.names[joined] == joined else "alias"

    def _part(self, words):
        for size in range(len(words) - 1, 0, -1):       # the longest run of words that is a name
            found = set()
            for start in range(len(words) - size + 1):
                name = self.parts.get(tuple(words[start:start + size]))
                if name and all(word in AROUND for word in words[:start] + words[start + size:]):
                    found.add(name)
            if found:
                return (found.pop(), "part") if len(found) == 1 else ("", None)

    def _near(self, words):
        # a request of one word may not differ by a replaced letter: `headers` is not `leaders`
        found = {name for key, name in self.by_length.get(len(words), [])
                 if all(a == b or _typo(a, b, len(words) > 1) for a, b in zip(words, key))}
        if not found and len(words) > 1:                # `quik sort`: one letter off in a short word, the other words as listed
            found = {name for key, name in self.by_length.get(len(words), [])
                     if sum(a != b for a, b in zip(words, key)) == 1 and all(a == b or _one_off(a, b) for a, b in zip(words, key))}
        if len(found) == 1:
            return found.pop(), "near"

    def suggest(self, request, n=5):
        """Names of asks whose name or phrases look like the request (for a request that fits no rule)."""
        import difflib
        joined = "_".join(_words(request))
        close = difflib.get_close_matches(joined, list(self.names), n=4 * n, cutoff=0.6)
        words = set(_words(request))
        close += [key for key in self.names if words & set(key.split("_")) and key not in close][:4 * n]
        out = []
        for key in close:
            if self.names[key] not in out:
                out.append(self.names[key])
        return out[:n]

    def __call__(self, request):
        """(name of the ask, rule) or (the words joined with underscores, None)."""
        words = _words(request)
        if not words:
            raise ValueError("the request %r has no word besides filler words" % request)
        singular = [word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word for word in words]
        for rule in (self._exact, self._part, self._near):
            for form in (words, singular):
                found = rule(form)
                if found:
                    return found if found[1] else ("_".join(words), None)
        return "_".join(words), None
