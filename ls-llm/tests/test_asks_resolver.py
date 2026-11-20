"""The request resolver of the PC side (lsllm.asks.Resolver): every phrasing and every unambiguous other name of the
suite leads to its own ask, and the rules behave as documented."""
from lsllm.asks import ASKS, Resolver, _words, load_jsonl, request_to_prompt

SUITE = load_jsonl(ASKS)
RESOLVE = Resolver(SUITE)


def test_every_main_request_gives_the_prompt_of_the_fixed_rule():
    for ask in SUITE:
        assert "def %s(" % RESOLVE(ask["request"])[0] == request_to_prompt(ask["request"]) == "def %s(" % ask["name"]


def test_every_phrasing_and_other_name_finds_its_ask():
    from lsllm.asks import PHRASE_FILE, PHRASES, WORDS
    own, checked, first = {ask["name"] for ask in SUITE}, 0, {}
    by_hand = set(PHRASES) | set(WORDS)                 # keys decided by hand or by the phrase file rank above the other names
    if PHRASE_FILE.exists():
        by_hand |= {"_".join(_words(phrase)) for row in load_jsonl(PHRASE_FILE) for phrase in row["phrases"]}
    for ask in SUITE:
        for line in ask["requests"] + ask.get("aliases", []):
            key = "_".join(_words(line.replace("_", " ")))
            if (key in own and key != ask["name"]) or (key in by_hand and line not in ask["requests"]):
                continue                                # the name of another ask, or a key that a phrase decides
            first.setdefault(key, ask["name"])          # listed for several asks: the first of the suite wins
            assert RESOLVE(line.replace("_", " "))[0] == first[key], line
            checked += 1
    assert checked > 4000


def test_common_words_have_their_usual_meaning():
    for typed, name in (("add", "add_numbers"), ("multiply", "multiply_numbers"), ("search", "linear_search"), ("reverse", "reverse_string"),
                        ("prime", "is_prime"), ("sqrt", "square_root"), ("even odd", "even_or_odd"), ("generate prime numbers", "sieve_of_eratosthenes"),
                        ("make me binary search trees", "bst_insert"), ("heap sort", "heap_sort")):
        assert RESOLVE(typed)[0] == name, typed
    assert RESOLVE("headers")[1] is None                # one changed letter of a single word is not a typing error of `leaders`


def test_rules():
    assert RESOLVE("Make me a bubble sort program") == ("bubble_sort", "name")
    assert RESOLVE("bubblesort") == ("bubble_sort", "alias")
    assert RESOLVE("make me binary search trees") == ("bst_insert", "alias")        # plural removed, then a phrase
    assert RESOLVE("binary search") == ("binary_search", "name")
    assert RESOLVE("sort my list with bubble sort please") == ("bubble_sort", "part")
    assert RESOLVE("buble sort") == ("bubble_sort", "near")
    assert RESOLVE("snake game") == ("snake_game", None)                            # not in the suite: passed on as typed
    assert RESOLVE("bubble srot") == ("bubble_sort", "near")                        # two letters swapped
    assert RESOLVE("quik sort") == ("quick_sort", "near")                           # one letter off in a short word, next to a listed word
    assert "quick_sort" in RESOLVE.suggest("kwik sort") and RESOLVE("kwik sort")[1] is None
    assert RESOLVE("sort") == ("bubble_sort", "alias")                              # a word decided by hand (WORDS)
    assert RESOLVE("qsor") == ("qsor", None)                                        # not `qsort`: too short for a missing letter
    assert RESOLVE("heal") == ("heal", None)                                        # not `leap` or `heap`: one changed letter in a short word
    assert RESOLVE("zigzag traversal")[1] is None                                   # another word, not a typing error of `inorder`
    assert RESOLVE("bubble sort descending")[1] is None                             # a word that changes what is asked
    assert RESOLVE("check if a number is prime") == ("is_prime", "part")
