"""Practice tasks of the far families about numbers (format: see ../build.py)."""

family = "gcd_primes"

def common_divisors(a, b):
    """Return the list of the positive integers that divide both of the positive integers a and b, in increasing order."""
    return [d for d in range(1, min(a, b) + 1) if a % d == 0 and b % d == 0]
assert common_divisors(12, 18) == [1, 2, 3, 6]
assert common_divisors(7, 5) == [1]
assert common_divisors(8, 8) == [1, 2, 4, 8]
assert common_divisors(1, 9) == [1]
assert common_divisors(20, 30) == [1, 2, 5, 10]

def common_multiples_up_to(a, b, limit):
    """Return the list of the positive integers up to limit that are multiples of both of the positive integers a and b, in increasing order."""
    return [m for m in range(1, limit + 1) if m % a == 0 and m % b == 0]
assert common_multiples_up_to(2, 3, 20) == [6, 12, 18]
assert common_multiples_up_to(4, 6, 12) == [12]
assert common_multiples_up_to(2, 3, 11) == [6]
assert common_multiples_up_to(3, 5, 10) == []
assert common_multiples_up_to(1, 1, 3) == [1, 2, 3]
assert common_multiples_up_to(2, 4, 9) == [4, 8]

def next_prime(n):
    """Return the smallest prime number that is greater than the integer n, where n is at least 1."""
    n += 1
    while any(n % d == 0 for d in range(2, n)):
        n += 1
    return n
assert next_prime(1) == 2
assert next_prime(2) == 3
assert next_prime(3) == 5
assert next_prime(7) == 11
assert next_prime(13) == 17
assert next_prime(23) == 29

def is_perfect_number(n):
    """Return True if the positive integer n equals the sum of the positive integers smaller than n that divide n (as 6 = 1 + 2 + 3), otherwise False."""
    return sum(d for d in range(1, n) if n % d == 0) == n
assert is_perfect_number(6) == True
assert is_perfect_number(28) == True
assert is_perfect_number(12) == False
assert is_perfect_number(1) == False
assert is_perfect_number(8) == False
assert is_perfect_number(496) == True

def largest_odd_divisor(n):
    """Return the largest odd integer that divides the positive integer n."""
    while n % 2 == 0:
        n //= 2
    return n
assert largest_odd_divisor(12) == 3
assert largest_odd_divisor(7) == 7
assert largest_odd_divisor(8) == 1
assert largest_odd_divisor(1) == 1
assert largest_odd_divisor(40) == 5
assert largest_odd_divisor(18) == 9

family = "classification"

def temperature_word(degrees):
    """Return 'cold' if degrees is less than 10, 'warm' if degrees is at least 10 but less than 25, otherwise 'hot'."""
    if degrees < 10:
        return 'cold'
    if degrees < 25:
        return 'warm'
    return 'hot'
assert temperature_word(5) == 'cold'
assert temperature_word(10) == 'warm'
assert temperature_word(24) == 'warm'
assert temperature_word(25) == 'hot'
assert temperature_word(-3) == 'cold'
assert temperature_word(9.5) == 'cold'

def weekday_or_weekend(day):
    """Return 'weekend' if the day number day, which is from 1 to 7, is 6 or 7, otherwise 'weekday'."""
    if day >= 6:
        return 'weekend'
    return 'weekday'
assert weekday_or_weekend(1) == 'weekday'
assert weekday_or_weekend(5) == 'weekday'
assert weekday_or_weekend(6) == 'weekend'
assert weekday_or_weekend(7) == 'weekend'
assert weekday_or_weekend(3) == 'weekday'

def five_or_ten_word(n):
    """Return 'ten' if n is divisible by 10, otherwise 'five' if n is divisible by 5, otherwise 'other'."""
    if n % 10 == 0:
        return 'ten'
    if n % 5 == 0:
        return 'five'
    return 'other'
assert five_or_ten_word(20) == 'ten'
assert five_or_ten_word(15) == 'five'
assert five_or_ten_word(7) == 'other'
assert five_or_ten_word(0) == 'ten'
assert five_or_ten_word(5) == 'five'
assert five_or_ten_word(12) == 'other'

def how_many_positive(a, b):
    """Return 'both' if a and b are both greater than zero, 'one' if exactly one of them is greater than zero, otherwise 'none'."""
    if a > 0 and b > 0:
        return 'both'
    if a > 0 or b > 0:
        return 'one'
    return 'none'
assert how_many_positive(1, 1) == 'both'
assert how_many_positive(1, -2) == 'one'
assert how_many_positive(0, 1) == 'one'
assert how_many_positive(5, 0) == 'one'
assert how_many_positive(-1, -2) == 'none'
assert how_many_positive(0, 0) == 'none'

def count_word(count):
    """Return 'none' if count is 0, 'one' if count is 1, otherwise 'many'."""
    if count == 0:
        return 'none'
    if count == 1:
        return 'one'
    return 'many'
assert count_word(0) == 'none'
assert count_word(1) == 'one'
assert count_word(2) == 'many'
assert count_word(100) == 'many'
assert count_word(3) == 'many'

family = "digits"

def digit_product(n):
    """Return the product of the digits of the integer n, which is not negative."""
    product = 1
    for ch in str(n):
        product *= int(ch)
    return product
assert digit_product(123) == 6
assert digit_product(0) == 0
assert digit_product(7) == 7
assert digit_product(105) == 0
assert digit_product(99) == 81
assert digit_product(222) == 8

def largest_digit(n):
    """Return the largest digit of the integer n, which is not negative."""
    return max(int(ch) for ch in str(n))
assert largest_digit(123) == 3
assert largest_digit(0) == 0
assert largest_digit(907) == 9
assert largest_digit(55) == 5
assert largest_digit(1000) == 1
assert largest_digit(382) == 8

def without_last_digit(n):
    """Return the integer n without its last digit, where n is not negative (0 if n is less than 10)."""
    return n // 10
assert without_last_digit(123) == 12
assert without_last_digit(5) == 0
assert without_last_digit(10) == 1
assert without_last_digit(1000) == 100
assert without_last_digit(99) == 9

def even_digits(n):
    """Return the list of the even digits of the integer n from left to right, where n is not negative."""
    return [int(ch) for ch in str(n) if int(ch) % 2 == 0]
assert even_digits(1234) == [2, 4]
assert even_digits(135) == []
assert even_digits(0) == [0]
assert even_digits(2082) == [2, 0, 8, 2]
assert even_digits(7) == []

def count_digit_in_number(n, digit):
    """Return how many of the digits of the integer n, which is not negative, are equal to digit, a number from 0 to 9."""
    return sum(1 for ch in str(n) if int(ch) == digit)
assert count_digit_in_number(12321, 2) == 2
assert count_digit_in_number(12321, 3) == 1
assert count_digit_in_number(555, 5) == 3
assert count_digit_in_number(123, 7) == 0
assert count_digit_in_number(1000, 0) == 3
assert count_digit_in_number(0, 0) == 1

family = "sequences"

def first_fibonacci_above(limit):
    """Return the smallest Fibonacci number greater than limit, where the Fibonacci numbers are 0, 1, 1, 2, 3, 5, 8 and so on (each is the sum of the two before it)."""
    a, b = 0, 1
    while a <= limit:
        a, b = b, a + b
    return a
assert first_fibonacci_above(0) == 1
assert first_fibonacci_above(1) == 2
assert first_fibonacci_above(4) == 5
assert first_fibonacci_above(5) == 8
assert first_fibonacci_above(50) == 55
assert first_fibonacci_above(-1) == 0

def triangular_numbers(n):
    """Return the list of the first n numbers of the sequence 1, 3, 6, 10, 15 and so on, where number k is the sum of the integers from 1 to k."""
    result = []
    total = 0
    for g in range(1, n + 1):
        total += g
        result.append(total)
    return result
assert triangular_numbers(0) == []
assert triangular_numbers(1) == [1]
assert triangular_numbers(4) == [1, 3, 6, 10]
assert triangular_numbers(6) == [1, 3, 6, 10, 15, 21]

def collatz_sequence(n):
    """Return the list that starts with the positive integer n and goes on until it reaches 1, where an even number is followed by its half and an odd number k by 3 * k + 1."""
    result = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        result.append(n)
    return result
assert collatz_sequence(1) == [1]
assert collatz_sequence(2) == [2, 1]
assert collatz_sequence(3) == [3, 10, 5, 16, 8, 4, 2, 1]
assert collatz_sequence(6) == [6, 3, 10, 5, 16, 8, 4, 2, 1]
assert collatz_sequence(8) == [8, 4, 2, 1]

def count_doublings(start, limit):
    """Return how many times the positive number start must be doubled until it is at least limit (0 if it already is)."""
    count = 0
    while start < limit:
        start *= 2
        count += 1
    return count
assert count_doublings(1, 8) == 3
assert count_doublings(1, 9) == 4
assert count_doublings(3, 10) == 2
assert count_doublings(5, 5) == 0
assert count_doublings(7, 2) == 0
assert count_doublings(1, 2) == 1

def powers_of_three_up_to(limit):
    """Return the list of the numbers 1, 3, 9, 27 and so on, each three times the one before, that are not larger than limit."""
    result = []
    power = 1
    while power <= limit:
        result.append(power)
        power *= 3
    return result
assert powers_of_three_up_to(0) == []
assert powers_of_three_up_to(1) == [1]
assert powers_of_three_up_to(9) == [1, 3, 9]
assert powers_of_three_up_to(26) == [1, 3, 9]
assert powers_of_three_up_to(100) == [1, 3, 9, 27, 81]

family = "binary"

def to_padded_binary(n, width):
    """Return the binary digits of the integer n as a string, with '0' added in front until it has width characters, where n is not negative (5 and 8 give '00000101')."""
    return bin(n)[2:].zfill(width)
assert to_padded_binary(5, 8) == '00000101'
assert to_padded_binary(0, 4) == '0000'
assert to_padded_binary(255, 8) == '11111111'
assert to_padded_binary(1, 1) == '1'
assert to_padded_binary(6, 4) == '0110'

def sum_of_binary_strings(a, b):
    """Return the sum, as an integer, of the two numbers whose binary digits are the strings a and b, so '101' and '11' give 8."""
    return int(a, 2) + int(b, 2)
assert sum_of_binary_strings('101', '11') == 8
assert sum_of_binary_strings('0', '0') == 0
assert sum_of_binary_strings('1', '1') == 2
assert sum_of_binary_strings('1000', '1') == 9
assert sum_of_binary_strings('1111', '1111') == 30

def binary_digits_list(n):
    """Return the list of the binary digits of the integer n from left to right, where n is not negative (5 gives [1, 0, 1] and 0 gives [0])."""
    return [int(ch) for ch in bin(n)[2:]]
assert binary_digits_list(5) == [1, 0, 1]
assert binary_digits_list(0) == [0]
assert binary_digits_list(1) == [1]
assert binary_digits_list(8) == [1, 0, 0, 0]
assert binary_digits_list(6) == [1, 1, 0]

def strip_leading_zeros(bits):
    """Return the string bits, which holds binary digits, without the '0' characters at its front, but '0' if all its characters are '0'."""
    return bits.lstrip('0') or '0'
assert strip_leading_zeros('00101') == '101'
assert strip_leading_zeros('101') == '101'
assert strip_leading_zeros('000') == '0'
assert strip_leading_zeros('0') == '0'
assert strip_leading_zeros('0100') == '100'
assert strip_leading_zeros('1') == '1'

def largest_with_bits(count):
    """Return the largest integer that can be written with count binary digits, where count is at least 1 (8 gives 255)."""
    return 2 ** count - 1
assert largest_with_bits(1) == 1
assert largest_with_bits(2) == 3
assert largest_with_bits(4) == 15
assert largest_with_bits(8) == 255
assert largest_with_bits(10) == 1023
