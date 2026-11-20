The request suite (`data/asks/asks.jsonl`) has 401 programs with tests (numbers 105, lists 82, strings 63, classics 52, structures 30, sorting 25, dictionaries 14, matrices 14, bits 9, searching 7). A request such as "make me a bubble sort
program" becomes the prompt `def bubble_sort(`; the model writes the rest with the syntax filter on and no docstring; the
result is run against the tests of the request in the sandbox.
**This is a recall test of programs that are in the training text of the board model, not a benchmark.** Where the texts
come from: 233 requests have a human-written function from the licensed corpus that passes the tests (renamed to the
request's name, docstring removed; in 3 cases the helper functions of its file were moved inside; 2 need more than
230 tokens); for 167 more the local model Qwen2.5-Coder-1.5B wrote a program that passes the tests; 1 has no text and
is not trained. Every source and licence is listed in `docs/ASKS_SOURCES.md`. The board model is never used for a
benchmark score (D-board-1).

| model | requests passed | of those with a training text | texts written out exactly | bits | classics | dictionaries | lists | matrices | numbers | searching | sorting | strings | structures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tier A board model, float (`runs/f/C2`) | 398 of 401 | 398 of 400 | 398 of 400 | 9 of 9 | 51 of 52 | 13 of 14 | 82 of 82 | 14 of 14 | 105 of 105 | 7 of 7 | 25 of 25 | 63 of 63 | 29 of 30 |
| Tier A board model, int8 after QAT (`runs/qat/C2b`) | 341 of 401 | 341 of 400 | 322 of 400 | 9 of 9 | 42 of 52 | 12 of 14 | 75 of 82 | 11 of 14 | 94 of 105 | 4 of 7 | 15 of 25 | 57 of 63 | 22 of 30 |
| Tier A board model, int8 after QAT and exact repair (`runs/qat/C2bp`): what the CPU computes | 400 of 401 | 400 of 400 | 400 of 400 | 9 of 9 | 51 of 52 | 14 of 14 | 82 of 82 | 14 of 14 | 105 of 105 | 7 of 7 | 25 of 25 | 63 of 63 | 30 of 30 |

"Texts written out exactly": the float and QAT rows are the trainer's count (every other token at least one code below the
wanted one at every position); the repaired model is counted from its written files with the rule of the CPU (the smallest
index among the largest logits wins), by `lsllm/intref/polish.py`. A request passes if the program passes its tests; a
program can pass without being the stored text.

Tier A board model, float (`runs/f/C2`): not passed: 1 syntax, 1 error, 1 wrong. Trained but not passed: build_min_heap, sort_dict_by_key.
Tier A board model, int8 after QAT (`runs/qat/C2b`): not passed: 26 error, 11 syntax, 21 wrong, 2 timeout. Trained but not passed: merge_sort, heap_sort, is_perfect_number, roman_int, is_anagram, capitalize, average, remove_duplicates, is_bst, pascal_triangle, longest_common_subsequence, bfs, comb_sort, radix_sort, pancake_sort, cycle_sort, odd_even_sort, jump_search, move_zeros, majority_element, plus_one, is_abundant, compound_interest, quadratic_roots, totient, is_happy_number, trailing_zeros, add_binary, group_by_first_letter, determinant, rock_paper_scissors, coin_change, group_anagrams, postorder_traversal, is_same_tree, linked_list_length, has_cycle, heap_push, heap_pop, build_min_heap, is_cyclic, tic_tac_toe_winner, is_valid_date, sort_dict_by_key, zigzag_conversion, valid_palindrome, find_substring, longest_common_substring, next_greater_element, equilibrium_index, is_strong_number, npr, search_insert, ternary_search, stooge_sort, pigeonhole_sort, sort_by_frequency, is_magic_square, min_path_sum.
Tier A board model, int8 after QAT and exact repair (`runs/qat/C2bp`): what the CPU computes: not passed: 1 error. Trained but not passed: none.

**How the board model is made** (D-board-2, D-board-3, D-board-4). It starts from the earlier Tier A model and is trained further on
general text plus request rows: one request program per row, from the first token of the context, as on the board; the run keeps
the weights that write the most texts exactly. The same is done with the integers (QAT, two stages). What is still wrong
then is mended exactly: a logit is `rq(y . E[v] + c_l, sh_l)`, and an entry of E can move up to 7 codes without changing the
input embedding (the input reads `rqr(E[v], 3)`), so the effect of such a change on every logit of every text is known by
integer arithmetic alone. 78 repairs moved 2,481 of the 81,920 entries of E; no input embedding changed; 400 of 400 texts are
exact afterwards. The price is paid on general text (loss on the body tokens of the validation text, nats per token):
the model it started from, float 1.7915; board model, float 1.8252; int8 after QAT 1.9288; after the repair 1.9432.

**The same requests on the simulated CPU** (Tier A board model, int8 after QAT and exact repair (`runs/qat/C2bp`): what the CPU computes; `tools/board_verify.py`: the generated LS program with the UART prompt
protocol and the syntax filter, in the cycle-exact simulator, one request after the other like the board):

| | |
|---|---|
| requests sent | 401 |
| answers that pass the tests of their request | 400 |
| answers equal to the integer reference, token by token | 401 of 401 |
| tokens sent | 29,135 |
| cycles | 17,347,450,624 |
| time per request at 100 MHz: median, longest | 0.33 s, 1.54 s (`cycle_sort`, 241 tokens) |
| all requests at 100 MHz | 173.5 s |
| tokens per second over all requests (reading the prompt included) | 167.9 |

Three answers exactly as the simulated CPU sent them (requests typed as "make me a bubblesort program", "make me binary search
trees", "shortest path"; the PC turned them into `def bubble_sort(`, `def bst_insert(`, `def dijkstra(`):

```python
def bubble_sort(array):
    n = len(array)
    for i in range(n):
        for j in range(0, n - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = (array[j + 1], array[j])
    return array
```
81 tokens, 46,611,752 cycles = 0.47 s at 100 MHz; passes the 7 tests of its request.

```python
def bst_insert(root, key):
    if root is None:
        return Node(key)
    if key < root.val:
        root.left = bst_insert(root.left, key)
    else:
        root.right = bst_insert(root.right, key)
    return root
```
89 tokens, 50,344,740 cycles = 0.50 s at 100 MHz; passes the 6 tests of its request. (`Node` is the node class of the tests: a tree program needs one.)

```python
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    visited = set()
    while visited != set(graph):
        nearest = min(((distances[node], node) for node in graph if node not in visited))[1]
        visited.add(nearest)
        for neighbour, weight in graph[nearest].items():
            distances[neighbour] = min(distances[neighbour], distances[nearest] + weight)
    return distances
```
202 tokens, 123,867,032 cycles = 1.24 s at 100 MHz; passes the 6 tests of its request.


**Requests the model was not trained on.** An earlier board model (`runs/qat/C1p`, trained and repaired the same way) had the texts of 284 requests; of the other 117 requests of the suite it
passed 1 (heap_sort). A model of this size writes the programs it was trained on; a request outside the list gets text that
looks like Python but is almost never the right program. This is the measured limit, not a setting. (The pass of heap_sort
was the model's shell sort written under the asked name: the tests check that a list gets sorted, not how.)

**From a sentence to the name** (`lsllm.asks.Resolver`, on the PC, before anything is sent): the words of the request are
the name of a request ("make me a bubble sort program"), a phrase or another name listed for a request ("bubblesort",
"primality test", "shortest path"), a name inside a longer sentence ("sort my list with bubble sort"), or a typing
error of one ("buble sort"). Of the 10,060 request lines, phrases and other names of the suite, 9,900 lead to their own request
(`tests/test_asks_resolver.py`, `tests/test_asks_phrases.py`); the rest are names that several requests share, which go
to one of them by a fixed rule. A sentence that fits no rule is not sent: `tools/board_ask.py` says that the model does not
know this request and names the closest ones it does know (`--anyway` sends it as typed). The CPU always receives
`def <name>(` as tokens.

How a request reaches the CPU: the program takes one of up to 64 stored prompts by the switches, or reads a typed prompt from
the UART (`tools/board_ask.py`; this needs the proposed `RX` instruction).
