# The prompts of the switch table

Set the switches SW5..SW0 to the number of a prompt and press reset: the program reads the switches once
(`IN`), copies that prompt into its prompt buffer, answers over the UART and halts. A number beyond the table
(64 and up) gives prompt 0. The switches above SW5 do not choose a prompt; all 16 switches also seed the random
generator, which matters only for a program that samples. In a prompt `\n` stands for a line break; the token
count includes the BOS token in front.

| SW5..SW0 | number | prompt | tokens | token ids |
|---|---|---|---|---|
| `000000` | 0 | `def bubble_sort(` | 10 | 1 53 192 449 449 51 150 39 431 12 |
| `000001` | 1 | `def insertion_sort(` | 9 | 1 53 29 172 122 150 39 431 12 |
| `000010` | 2 | `def selection_sort(` | 9 | 1 53 52 51 385 150 39 431 12 |
| `000011` | 3 | `def merge_sort(` | 9 | 1 53 445 31 73 150 39 431 12 |
| `000100` | 4 | `def quick_sort(` | 9 | 1 53 162 436 119 150 39 431 12 |
| `000101` | 5 | `def shell_sort(` | 9 | 1 53 184 95 438 150 39 431 12 |
| `000110` | 6 | `def heap_sort(` | 8 | 1 53 170 169 150 39 431 12 |
| `000111` | 7 | `def counting_sort(` | 9 | 1 53 219 431 61 150 39 431 12 |
| `001000` | 8 | `def linear_search(` | 9 | 1 53 360 43 441 52 43 75 12 |
| `001001` | 9 | `def binary_search(` | 10 | 1 53 449 29 43 200 52 43 75 12 |
| `001010` | 10 | `def gcd(` | 6 | 1 53 448 442 439 12 |
| `001011` | 11 | `def lcm(` | 6 | 1 53 438 442 445 12 |
| `001100` | 12 | `def is_prime(` | 7 | 1 53 275 444 144 99 12 |
| `001101` | 13 | `def sieve_of_eratosthenes(` | 16 | 1 53 79 429 121 441 437 382 31 35 437 32 446 45 269 12 |
| `001110` | 14 | `def prime_factors(` | 10 | 1 53 444 144 99 441 212 321 435 12 |
| `001111` | 15 | `def factorial(` | 7 | 1 53 212 321 436 44 12 |
| `010000` | 16 | `def fibonacci(` | 8 | 1 53 70 449 38 231 244 12 |
| `010001` | 17 | `def fibonacci_sequence(` | 13 | 1 53 70 449 38 231 244 441 52 162 45 69 12 |
| `010010` | 18 | `def power(` | 6 | 1 53 155 454 31 12 |
| `010011` | 19 | `def sum_of_digits(` | 11 | 1 53 303 441 437 382 77 448 163 435 12 |
| `010100` | 20 | `def count_digits(` | 9 | 1 53 219 62 77 448 163 435 12 |
| `010101` | 21 | `def decimal_binary(` | 11 | 1 53 36 244 445 44 356 29 43 451 12 |
| `010110` | 22 | `def binary_decimal(` | 11 | 1 53 449 29 43 200 36 244 445 44 12 |
| `010111` | 23 | `def is_perfect_number(` | 9 | 1 53 275 248 312 442 62 177 12 |
| `011000` | 24 | `def is_armstrong_number(` | 11 | 1 53 275 43 445 78 38 448 441 177 12 |
| `011001` | 25 | `def is_even(` | 6 | 1 53 275 420 45 12 |
| `011010` | 26 | `def is_power_of_two(` | 12 | 1 53 275 155 454 384 437 382 431 454 437 12 |
| `011011` | 27 | `def is_perfect_square(` | 12 | 1 53 275 248 312 442 62 435 162 434 28 12 |
| `011100` | 28 | `def divisors(` | 8 | 1 53 77 452 91 39 435 12 |
| `011101` | 29 | `def int_roman(` | 8 | 1 53 29 62 65 445 48 12 |
| `011110` | 30 | `def roman_int(` | 8 | 1 53 65 445 48 441 158 12 |
| `011111` | 31 | `def binomial_coefficient(` | 16 | 1 53 449 29 437 445 436 44 441 58 429 443 70 244 107 12 |
| `100000` | 32 | `def count_set_bits(` | 10 | 1 53 219 62 52 62 449 163 435 12 |
| `100001` | 33 | `def sum_of_squares(` | 11 | 1 53 303 441 437 443 150 162 434 129 12 |
| `100010` | 34 | `def digital_root(` | 11 | 1 53 77 448 163 44 441 65 437 431 12 |
| `100011` | 35 | `def reverse_number(` | 7 | 1 53 28 120 394 177 12 |
| `100100` | 36 | `def reverse_string(` | 7 | 1 53 28 120 394 182 12 |
| `100101` | 37 | `def is_palindrome(` | 10 | 1 53 275 444 44 29 439 65 99 12 |
| `100110` | 38 | `def is_anagram(` | 8 | 1 53 275 48 336 432 54 12 |
| `100111` | 39 | `def count_vowels(` | 10 | 1 53 219 62 452 437 454 95 435 12 |
| `101000` | 40 | `def count_words(` | 8 | 1 53 219 62 300 439 435 12 |
| `101001` | 41 | `def word_frequency(` | 11 | 1 53 300 97 443 28 162 45 442 451 12 |
| `101010` | 42 | `def char_frequency(` | 11 | 1 53 254 441 443 28 162 45 442 451 12 |
| `101011` | 43 | `def capitalize(` | 8 | 1 53 442 169 163 363 217 12 |
| `101100` | 44 | `def caesar_cipher(` | 11 | 1 53 318 269 43 441 244 444 446 31 12 |
| `101101` | 45 | `def run_length_encoding(` | 14 | 1 53 432 66 441 145 448 40 441 45 58 439 61 12 |
| `101110` | 46 | `def is_pangram(` | 9 | 1 53 275 444 48 448 432 54 12 |
| `101111` | 47 | `def is_isogram(` | 9 | 1 53 275 91 437 448 432 54 12 |
| `110000` | 48 | `def remove_vowels(` | 12 | 1 53 28 128 121 441 452 437 454 95 435 12 |
| `110001` | 49 | `def reverse_words(` | 9 | 1 53 28 120 394 300 439 435 12 |
| `110010` | 50 | `def longest_word(` | 9 | 1 53 438 38 73 380 300 439 12 |
| `110011` | 51 | `def hamming_distance(` | 11 | 1 53 446 54 445 61 441 77 32 240 12 |
| `110100` | 52 | `def edit_distance(` | 9 | 1 53 429 77 62 77 32 240 12 |
| `110101` | 53 | `def camel_snake(` | 10 | 1 53 442 86 438 150 433 434 139 12 |
| `110110` | 54 | `def longest_common_prefix(` | 13 | 1 53 438 38 73 380 419 38 441 220 70 453 12 |
| `110111` | 55 | `def balanced_parentheses(` | 12 | 1 53 449 44 240 97 180 227 400 52 435 12 |
| `111000` | 56 | `def find_max(` | 8 | 1 53 443 29 97 116 453 12 |
| `111001` | 57 | `def find_min(` | 7 | 1 53 443 29 97 288 12 |
| `111010` | 58 | `def sum_list(` | 6 | 1 53 303 441 100 12 |
| `111011` | 59 | `def average(` | 6 | 1 53 434 120 333 12 |
| `111100` | 60 | `def median(` | 6 | 1 53 99 77 48 12 |
| `111101` | 61 | `def mode(` | 4 | 1 53 343 12 |
| `111110` | 62 | `def remove_duplicates(` | 13 | 1 53 28 128 121 441 439 133 60 442 102 435 12 |
| `111111` | 63 | `def flatten(` | 7 | 1 53 443 438 35 357 12 |

64 prompts, 599 tokens, 1,198 bytes in the table (MEMORY_MAP.md, section The prompts).
