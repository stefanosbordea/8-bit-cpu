"""Test tasks of the far families about numbers (format: see ../build.py)."""

family = "gcd_primes"

def gcd_of_two(a, b):
    """Return the largest integer that divides both of the positive integers a and b."""
    for d in range(min(a, b), 0, -1):
        if a % d == 0 and b % d == 0:
            return d
assert gcd_of_two(12, 18) == 6
assert gcd_of_two(7, 5) == 1
assert gcd_of_two(9, 9) == 9
assert gcd_of_two(1, 10) == 1
assert gcd_of_two(20, 5) == 5
assert gcd_of_two(6, 9) == 3

def lcm_of_two(a, b):
    """Return the smallest positive integer that is a multiple of both of the positive integers a and b."""
    multiple = a
    while multiple % b != 0:
        multiple += a
    return multiple
assert lcm_of_two(4, 6) == 12
assert lcm_of_two(3, 5) == 15
assert lcm_of_two(7, 7) == 7
assert lcm_of_two(1, 9) == 9
assert lcm_of_two(10, 5) == 10

def is_prime_number(n):
    """Return True if the integer n is prime (at least 2 and divisible only by 1 and itself), otherwise False."""
    if n < 2:
        return False
    for d in range(2, n):
        if n % d == 0:
            return False
    return True
assert is_prime_number(2) == True
assert is_prime_number(9) == False
assert is_prime_number(4) == False
assert is_prime_number(1) == False
assert is_prime_number(0) == False
assert is_prime_number(17) == True

def list_divisors(n):
    """Return the list of the positive integers that divide the positive integer n, in increasing order."""
    return [d for d in range(1, n + 1) if n % d == 0]
assert list_divisors(6) == [1, 2, 3, 6]
assert list_divisors(1) == [1]
assert list_divisors(7) == [1, 7]
assert list_divisors(12) == [1, 2, 3, 4, 6, 12]
assert list_divisors(9) == [1, 3, 9]

def smallest_factor(n):
    """Return the smallest integer greater than 1 that divides the integer n, where n is at least 2."""
    d = 2
    while n % d != 0:
        d += 1
    return d
assert smallest_factor(6) == 2
assert smallest_factor(7) == 7
assert smallest_factor(9) == 3
assert smallest_factor(2) == 2
assert smallest_factor(35) == 5
assert smallest_factor(49) == 7

family = "classification"

def sign_word(n):
    """Return 'negative' if n is less than zero, 'zero' if n is zero, otherwise 'positive'."""
    if n < 0:
        return 'negative'
    if n == 0:
        return 'zero'
    return 'positive'
assert sign_word(5) == 'positive'
assert sign_word(-3) == 'negative'
assert sign_word(0) == 'zero'
assert sign_word(0.5) == 'positive'
assert sign_word(-0.1) == 'negative'

def fizz_or_buzz(n):
    """Return 'Fizz' if n is divisible by 3, otherwise 'Buzz' if n is divisible by 5, otherwise n written as a string."""
    if n % 3 == 0:
        return 'Fizz'
    if n % 5 == 0:
        return 'Buzz'
    return str(n)
assert fizz_or_buzz(3) == 'Fizz'
assert fizz_or_buzz(5) == 'Buzz'
assert fizz_or_buzz(15) == 'Fizz'
assert fizz_or_buzz(7) == '7'
assert fizz_or_buzz(10) == 'Buzz'
assert fizz_or_buzz(1) == '1'

def grade_letter(score):
    """Return 'A' if score is at least 90, 'B' if score is at least 80 but less than 90, otherwise 'C'."""
    if score >= 90:
        return 'A'
    if score >= 80:
        return 'B'
    return 'C'
assert grade_letter(95) == 'A'
assert grade_letter(90) == 'A'
assert grade_letter(89) == 'B'
assert grade_letter(80) == 'B'
assert grade_letter(79) == 'C'
assert grade_letter(0) == 'C'

def comparison_word(a, b):
    """Return 'less' if a is smaller than b, 'equal' if a equals b, otherwise 'greater'."""
    if a < b:
        return 'less'
    if a == b:
        return 'equal'
    return 'greater'
assert comparison_word(1, 2) == 'less'
assert comparison_word(2, 2) == 'equal'
assert comparison_word(3, 2) == 'greater'
assert comparison_word(-1, -2) == 'greater'
assert comparison_word(0, 0.5) == 'less'

def even_or_odd_word(n):
    """Return the word 'even' if the integer n is even, otherwise the word 'odd'."""
    if n % 2 == 0:
        return 'even'
    return 'odd'
assert even_or_odd_word(4) == 'even'
assert even_or_odd_word(7) == 'odd'
assert even_or_odd_word(0) == 'even'
assert even_or_odd_word(-1) == 'odd'
assert even_or_odd_word(-2) == 'even'

family = "digits"

def digit_sum(n):
    """Return the sum of the digits of the integer n, which is not negative."""
    return sum(int(ch) for ch in str(n))
assert digit_sum(123) == 6
assert digit_sum(0) == 0
assert digit_sum(9) == 9
assert digit_sum(1000) == 1
assert digit_sum(99) == 18

def number_of_digits(n):
    """Return how many digits the integer n has, where n is not negative (0 has one digit)."""
    return len(str(n))
assert number_of_digits(123) == 3
assert number_of_digits(0) == 1
assert number_of_digits(9) == 1
assert number_of_digits(1000) == 4
assert number_of_digits(10) == 2

def reverse_digits(n):
    """Return the integer made of the digits of n in reverse order, where n is not negative (120 gives 21)."""
    return int(str(n)[::-1])
assert reverse_digits(123) == 321
assert reverse_digits(5) == 5
assert reverse_digits(120) == 21
assert reverse_digits(0) == 0
assert reverse_digits(1001) == 1001

def list_of_digits(n):
    """Return the list of the digits of the integer n from left to right, where n is not negative (0 gives [0])."""
    return [int(ch) for ch in str(n)]
assert list_of_digits(123) == [1, 2, 3]
assert list_of_digits(0) == [0]
assert list_of_digits(7) == [7]
assert list_of_digits(1000) == [1, 0, 0, 0]

def tens_digit(n):
    """Return the digit in the tens place of the integer n, where n is not negative (0 if n is less than 10)."""
    return n // 10 % 10
assert tens_digit(123) == 2
assert tens_digit(5) == 0
assert tens_digit(10) == 1
assert tens_digit(99) == 9
assert tens_digit(1000) == 0
assert tens_digit(47) == 4

family = "sequences"

def fibonacci_number(n):
    """Return Fibonacci number n, where numbers 0 and 1 are 0 and 1 and every later one is the sum of the two before it."""
    a, b = 0, 1
    for g in range(n):
        a, b = b, a + b
    return a
assert fibonacci_number(0) == 0
assert fibonacci_number(1) == 1
assert fibonacci_number(2) == 1
assert fibonacci_number(5) == 5
assert fibonacci_number(10) == 55

def fibonacci_list(n):
    """Return the list of the first n Fibonacci numbers: 0, 1, and then every number is the sum of the two before it."""
    result = []
    a, b = 0, 1
    for g in range(n):
        result.append(a)
        a, b = b, a + b
    return result
assert fibonacci_list(0) == []
assert fibonacci_list(1) == [0]
assert fibonacci_list(2) == [0, 1]
assert fibonacci_list(6) == [0, 1, 1, 2, 3, 5]

def collatz_steps(n):
    """Return how many steps take the positive integer n to 1, where a step halves an even number and turns an odd number k into 3 * k + 1."""
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps
assert collatz_steps(1) == 0
assert collatz_steps(2) == 1
assert collatz_steps(3) == 7
assert collatz_steps(6) == 8
assert collatz_steps(4) == 2

def count_halvings(n):
    """Return how many times the positive integer n must be halved, dropping remainders, until it is 1."""
    count = 0
    while n > 1:
        n //= 2
        count += 1
    return count
assert count_halvings(1) == 0
assert count_halvings(2) == 1
assert count_halvings(8) == 3
assert count_halvings(9) == 3
assert count_halvings(100) == 6

def power_of_two_above(n):
    """Return the smallest of the numbers 1, 2, 4, 8, 16 and so on that is greater than n."""
    power = 1
    while power <= n:
        power *= 2
    return power
assert power_of_two_above(0) == 1
assert power_of_two_above(1) == 2
assert power_of_two_above(5) == 8
assert power_of_two_above(8) == 16
assert power_of_two_above(100) == 128

family = "binary"

def to_binary_string(n):
    """Return the binary digits of the integer n as a string, where n is not negative (5 gives '101' and 0 gives '0')."""
    return bin(n)[2:]
assert to_binary_string(5) == '101'
assert to_binary_string(0) == '0'
assert to_binary_string(1) == '1'
assert to_binary_string(8) == '1000'
assert to_binary_string(255) == '11111111'

def binary_to_number(bits):
    """Return the integer whose binary digits are the characters of the string bits, so '101' gives 5."""
    return int(bits, 2)
assert binary_to_number('101') == 5
assert binary_to_number('0') == 0
assert binary_to_number('1') == 1
assert binary_to_number('1000') == 8
assert binary_to_number('11111111') == 255

def count_one_bits(n):
    """Return how many digits 1 the integer n has when it is written in binary, where n is not negative."""
    return bin(n).count('1')
assert count_one_bits(5) == 2
assert count_one_bits(0) == 0
assert count_one_bits(1) == 1
assert count_one_bits(8) == 1
assert count_one_bits(255) == 8
assert count_one_bits(7) == 3

def binary_length(n):
    """Return how many digits the positive integer n has when it is written in binary, so 5 (binary 101) gives 3."""
    return len(bin(n)) - 2
assert binary_length(1) == 1
assert binary_length(2) == 2
assert binary_length(5) == 3
assert binary_length(8) == 4
assert binary_length(255) == 8
assert binary_length(256) == 9

def flip_bits(bits):
    """Return the string bits with every '0' changed to '1' and every '1' changed to '0'."""
    return ''.join('1' if ch == '0' else '0' for ch in bits)
assert flip_bits('101') == '010'
assert flip_bits('') == ''
assert flip_bits('0') == '1'
assert flip_bits('1111') == '0000'
assert flip_bits('0011') == '1100'
