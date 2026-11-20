"""Change one whole number of a documented function in its docstring and in its code together (data augmentation).

A mechanical transformation of an existing function: "Return n multiplied by 10." / `return n * 10` becomes
"Return n multiplied by 7." / `return n * 7`. Only a number that is written exactly once in the docstring (as
digits) and exactly once in the code is changed, so the sentence and the code keep saying the same thing. A model
trained on such variants has to copy the number from the docstring instead of remembering it.

    new = renumber(text, random.Random(0))        # None if the function has no such number

If the name of the function states the number (`add_five`, `larger_than_10`), the name changes with it; a function
whose name is tied to a number in another way (`double`, `first`, a unit, a second number) is left alone. The caller must check the result against the benchmark overlap rules again: a new number can
turn a function into the solution of a benchmark task.
"""
import ast
import re

from lsllm.canon import canon

CARDINALS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
             11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
             19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
             100: "hundred", 1000: "thousand"}
PLURALS = {"six": "sixes", "twenty": "twenties", "thirty": "thirties", "forty": "forties", "fifty": "fifties", "sixty": "sixties"}
# words in a function name that tie it to a number in a way a changed constant would break
NUMBER_WORDS = re.compile(r"\d|" + "|".join(sorted(CARDINALS.values(), key=len, reverse=True)) +
                          r"|million|dozen|double|triple|twice|thrice|half|halve|quarter|square|cube|cubic|"
                          r"percent|pair|single|first|second|third|fourth|fifth|last|even|odd|binary|decimal|week|minute|hour|"
                          r"day|year|month|inch|feet|foot|mile|meter|metre|gram|pound|ounce|celsius|fahrenheit|kelvin|byte|bit", re.I)
STANDALONE = re.compile(r"(?<![\w.])\d+(?!\w|\.\d)")
CHOICES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 50, 100]


def plural(word):
    return PLURALS.get(word, word + "s")


def rename_number(name, old, new):
    """The function name with the number `old` replaced by `new` where a part of the name states it, as digits or
    as a word (`add_five`, `larger_than_10`, `count_fives`). None if the name is tied to a number in another way
    (another number, `double`, `first`, a unit) and would no longer fit."""
    parts, rest = [], []
    for part in name.split("_"):
        word = CARDINALS.get(old)
        if part == str(old):
            parts.append(str(new))
        elif word and part.lower() == word:
            parts.append(CARDINALS[new])
        elif word and part.lower() == plural(word):
            parts.append(plural(CARDINALS[new]))
        else:
            parts.append(part)
            rest.append(part)
    return None if NUMBER_WORDS.search("_".join(rest)) else "_".join(parts)


def renumber(text, rng):
    """A copy of the function `text` with one shared whole number replaced in docstring and code, or None."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    fn = tree.body[0]
    doc = ast.get_docstring(fn, clean=False)
    if not doc:
        return None
    in_doc = [int(m) for m in STANDALONE.findall(doc)]
    nodes = [node for node in ast.walk(fn) if isinstance(node, ast.Constant) and type(node.value) is int and node is not fn.body[0].value]
    in_code = [node.value for node in nodes]
    shared = [v for v in set(in_doc) if in_doc.count(v) == 1 and in_code.count(v) == 1 and 0 <= v <= 1000]
    if not shared:
        return None
    old = rng.choice(sorted(shared))
    new = rng.choice([v for v in CHOICES if v != old and v not in in_doc and v not in in_code])
    name = rename_number(fn.name, old, new)
    if name is None:
        return None
    for node in ast.walk(fn):                       # the function may call itself
        if isinstance(node, ast.Name) and node.id == fn.name:
            node.id = name
    fn.name = name
    for node in nodes:
        if node.value == old:
            node.value = new
    fn.body[0].value.value = STANDALONE.sub(lambda m: str(new) if int(m.group(0)) == old else m.group(0), doc)
    try:
        out = canon(ast.unparse(tree))
    except (SyntaxError, ValueError, RecursionError):
        return None
    return out if out and out != text else None
