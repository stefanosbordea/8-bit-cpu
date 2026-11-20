"""The phrases people use for the asks (data/asks/phrases.jsonl): every phrase leads to its own ask, and to that ask alone."""
from lsllm.asks import ASKS, PHRASE_FILE, Resolver, _words, load_jsonl

SUITE = load_jsonl(ASKS)
ROWS = load_jsonl(PHRASE_FILE)
RESOLVE = Resolver(SUITE)

# Requests as people type them -> the ask they must lead to (by a phrase, or by a phrase inside a longer sentence).
TYPED = """
prime checker -> is_prime
write a python program to check whether a number is prime or not -> is_prime
print all prime numbers -> sieve_of_eratosthenes
prime numbers in a range -> primes_between
find the largest number in a list -> find_max
largest of three numbers -> max_of_three
maximum of two numbers -> max_of_two
check whether a number is even or odd -> even_or_odd
is the number even -> is_even
temperature converter -> celsius_fahrenheit
convert celsius to fahrenheit -> celsius_fahrenheit
c to f -> celsius_fahrenheit
kadane -> max_subarray_sum
levenshtein distance -> edit_distance
hello world -> hello_world
simple calculator -> calculator
multiplication table -> multiplication_table
add two numbers -> add_numbers
swap two numbers -> swap_numbers
swap two elements in a list -> swap
print fibonacci series -> fibonacci_sequence
factorial using recursion -> factorial
check whether a number is a palindrome or not -> is_palindrome_number
check whether a string is palindrome or not -> is_palindrome
check whether a year is a leap year or not -> is_leap_year
sort a list of numbers -> bubble_sort
sort numbers in descending order -> sort_descending
search an element in a list -> linear_search
remove duplicates from a string -> remove_duplicate_chars
remove duplicates from a list -> remove_duplicates
most common character -> most_common_char
count substrings -> count_substring
string compression -> compress_string
run length decoding -> run_length_decoding
zigzag -> zigzag_conversion
vowels and consonants -> count_vowels_consonants
dijkstra's algorithm -> dijkstra
shortest path in a graph -> dijkstra
depth first search -> dfs
topological sort -> topological_sort
0/1 knapsack -> knapsack
union find -> union_sets
linked list -> insert_at_end
detect a loop in a linked list -> has_cycle
detect cycle in a graph -> is_cyclic
priority queue -> heap_push
heapify -> build_min_heap
preorder traversal -> preorder_traversal
postorder traversal -> postorder_traversal
level order traversal -> level_order_traversal
gray code -> gray_sequence
binary to gray -> binary_gray
strong number -> is_strong_number
binary gap -> binary_gap
climbing stairs -> climb_stairs
amicable numbers -> is_amicable
harmonic sum -> harmonic_sum
pairwise swap -> pairwise_swap
leaders in an array -> leaders
equilibrium index -> equilibrium_index
first occurrence -> first_occurrence
last occurrence -> last_index
insert position -> search_insert
stooge sort -> stooge_sort
sort by frequency -> sort_by_frequency
seconds to hh:mm:ss -> seconds_time
day of year -> day_of_year
sum of natural numbers -> triangular_number
area of a square -> square_area
convert kg to pounds -> kg_pounds
grade calculator -> letter_grade
tic tac toe -> tic_tac_toe_winner
number guessing game -> check_guess
number of combinations -> binomial_coefficient
all combinations of a list -> combinations
number of permutations -> npr
permutations of a string -> string_permutations
"""


def test_one_row_per_ask_with_enough_phrases():
    assert [(row["id"], row["name"]) for row in ROWS] == [(ask["id"], ask["name"]) for ask in SUITE]
    assert all(5 <= len(row["phrases"]) <= 15 for row in ROWS)


def test_no_key_is_claimed_twice():
    own, claimed = {ask["name"] for ask in SUITE}, {}
    for row in ROWS:
        for phrase in row["phrases"]:
            key = "_".join(_words(phrase))
            assert key and key not in own, phrase               # never the name of an ask, its own included
            assert key not in claimed, (phrase, claimed[key])   # never said twice, by this ask or by another
            claimed[key] = row["name"]


def test_every_phrase_finds_its_ask():
    for row in ROWS:
        for phrase in row["phrases"]:
            assert RESOLVE(phrase) == (row["name"], "alias"), phrase


def test_requests_people_type():
    for line in TYPED.strip().split("\n"):
        request, name = line.split(" -> ")
        assert RESOLVE(request)[0] == name and RESOLVE(request)[1] is not None, request
