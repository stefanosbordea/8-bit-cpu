from lsllm.alpha import alpha_key


def test_same_up_to_names_and_docstring():
    a = 'def add(a, b):\n    """Sum."""\n    return a + b'
    b = "def plus(x: int, y: int) -> int:\n    return x + y"
    assert alpha_key(a) == alpha_key(b)


def test_different_code_differs():
    assert alpha_key("def f(a, b):\n    return a + b") != alpha_key("def f(a, b):\n    return a - b")
    assert alpha_key("def f(a, b):\n    return a + b") != alpha_key("def f(a, b):\n    return b + a")


def test_locals_recursion_builtins():
    a = "def fact(n):\n    if n < 2:\n        return 1\n    r = n * fact(n - 1)\n    return r"
    b = "def factorial(k):\n    if k < 2:\n        return 1\n    out = k * factorial(k - 1)\n    return out"
    assert alpha_key(a) == alpha_key(b)
    assert alpha_key("def f(xs):\n    return len(xs)") != alpha_key("def f(xs):\n    return sum(xs)")


def test_no_function():
    assert alpha_key("x = 1") is None
    assert alpha_key("def (") is None
