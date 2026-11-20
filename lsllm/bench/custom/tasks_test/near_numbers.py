"""Test tasks of the near families about numbers and booleans (format: see ../build.py)."""

family = "arith_one"

def double_number(n):
    """Return n multiplied by 2."""
    return n * 2
assert double_number(3) == 6
assert double_number(0) == 0
assert double_number(-4) == -8
assert double_number(2.5) == 5.0
assert double_number(100) == 200

def square_number(n):
    """Return n multiplied by itself."""
    return n * n
assert square_number(3) == 9
assert square_number(0) == 0
assert square_number(-5) == 25
assert square_number(1) == 1
assert square_number(12) == 144

def half_of_number(n):
    """Return n divided by 2, so 5 gives 2.5."""
    return n / 2
assert half_of_number(8) == 4
assert half_of_number(0) == 0
assert half_of_number(5) == 2.5
assert half_of_number(-3) == -1.5
assert half_of_number(1) == 0.5

def add_ten(n):
    """Return n plus 10."""
    return n + 10
assert add_ten(5) == 15
assert add_ten(0) == 10
assert add_ten(-10) == 0
assert add_ten(-25) == -15
assert add_ten(1.5) == 11.5

def negate_number(n):
    """Return n with the opposite sign, so 5 gives -5 and -3 gives 3."""
    return -n
assert negate_number(5) == -5
assert negate_number(-3) == 3
assert negate_number(0) == 0
assert negate_number(2.5) == -2.5
assert negate_number(-100) == 100

family = "arith_two"

def subtract_numbers(a, b):
    """Return a minus b."""
    return a - b
assert subtract_numbers(5, 3) == 2
assert subtract_numbers(3, 5) == -2
assert subtract_numbers(0, 0) == 0
assert subtract_numbers(-4, -6) == 2
assert subtract_numbers(2.5, 0.5) == 2

def multiply_numbers(a, b):
    """Return a times b."""
    return a * b
assert multiply_numbers(3, 4) == 12
assert multiply_numbers(0, 7) == 0
assert multiply_numbers(-2, 5) == -10
assert multiply_numbers(-3, -3) == 9
assert multiply_numbers(1.5, 2) == 3

def whole_quotient(a, b):
    """Return how many whole times b fits into a, where a is not negative and b is positive."""
    return a // b
assert whole_quotient(7, 2) == 3
assert whole_quotient(6, 3) == 2
assert whole_quotient(1, 5) == 0
assert whole_quotient(0, 9) == 0
assert whole_quotient(9, 1) == 9

def division_remainder(a, b):
    """Return the remainder left when a is divided by b, where a is not negative and b is positive."""
    return a % b
assert division_remainder(7, 2) == 1
assert division_remainder(6, 3) == 0
assert division_remainder(1, 5) == 1
assert division_remainder(0, 4) == 0
assert division_remainder(17, 5) == 2

def raise_to_power(base, exponent):
    """Return base raised to the power exponent."""
    return base ** exponent
assert raise_to_power(2, 3) == 8
assert raise_to_power(5, 0) == 1
assert raise_to_power(3, 1) == 3
assert raise_to_power(-2, 3) == -8
assert raise_to_power(10, 2) == 100

family = "arith_three"

def sum_of_three(a, b, c):
    """Return the sum of a, b and c."""
    return a + b + c
assert sum_of_three(1, 2, 3) == 6
assert sum_of_three(0, 0, 0) == 0
assert sum_of_three(-1, 1, 0) == 0
assert sum_of_three(10, 20, 30) == 60
assert sum_of_three(1.5, 2.5, 1) == 5

def product_of_three(a, b, c):
    """Return a times b times c."""
    return a * b * c
assert product_of_three(2, 3, 4) == 24
assert product_of_three(1, 1, 1) == 1
assert product_of_three(5, 0, 7) == 0
assert product_of_three(-2, 3, 1) == -6
assert product_of_three(-1, -1, -1) == -1

def average_of_three(a, b, c):
    """Return the average of a, b and c, which is their sum divided by 3."""
    return (a + b + c) / 3
assert average_of_three(1, 2, 3) == 2
assert average_of_three(0, 0, 0) == 0
assert average_of_three(3, 3, 3) == 3
assert average_of_three(-3, 0, 3) == 0
assert average_of_three(10, 20, 60) == 30
assert average_of_three(1, 1, 2.5) == 1.5

def total_cost(price, count, shipping):
    """Return price times count, plus shipping."""
    return price * count + shipping
assert total_cost(2, 3, 1) == 7
assert total_cost(10, 0, 5) == 5
assert total_cost(0, 4, 0) == 0
assert total_cost(5, 2, 0) == 10
assert total_cost(1.5, 2, 1) == 4

def add_then_multiply(a, b, c):
    """Return the sum of a and b, multiplied by c."""
    return (a + b) * c
assert add_then_multiply(1, 2, 3) == 9
assert add_then_multiply(0, 0, 5) == 0
assert add_then_multiply(2, -2, 7) == 0
assert add_then_multiply(1, 1, -1) == -2
assert add_then_multiply(4, 6, 0.5) == 5

family = "compare"

def is_greater(a, b):
    """Return True if a is greater than b, otherwise False."""
    return a > b
assert is_greater(3, 2) == True
assert is_greater(2, 3) == False
assert is_greater(2, 2) == False
assert is_greater(-1, -2) == True
assert is_greater(0, 0.5) == False
assert is_greater(5, 2) == True

def is_equal(a, b):
    """Return True if a and b are equal, otherwise False."""
    return a == b
assert is_equal(1, 1) == True
assert is_equal(1, 2) == False
assert is_equal(0, 0) == True
assert is_equal(-1, 1) == False
assert is_equal(2.5, 2.5) == True
assert is_equal(2, 1) == False

def is_at_most(a, b):
    """Return True if a is less than or equal to b, otherwise False."""
    return a <= b
assert is_at_most(1, 2) == True
assert is_at_most(2, 2) == True
assert is_at_most(3, 2) == False
assert is_at_most(-5, -5) == True
assert is_at_most(0, -1) == False

def is_between(x, low, high):
    """Return True if x is at least low and at most high, otherwise False."""
    return low <= x <= high
assert is_between(5, 1, 10) == True
assert is_between(1, 1, 10) == True
assert is_between(10, 1, 10) == True
assert is_between(0, 1, 10) == False
assert is_between(11, 1, 10) == False
assert is_between(-3, -5, -1) == True

def all_three_equal(a, b, c):
    """Return True if a, b and c are all equal, otherwise False."""
    return a == b == c
assert all_three_equal(1, 1, 1) == True
assert all_three_equal(1, 1, 2) == False
assert all_three_equal(1, 2, 1) == False
assert all_three_equal(2, 1, 1) == False
assert all_three_equal(0, 0, 0) == True

family = "min_max_few"

def larger_of_two(a, b):
    """Return the larger of a and b."""
    return max(a, b)
assert larger_of_two(1, 2) == 2
assert larger_of_two(2, 1) == 2
assert larger_of_two(3, 3) == 3
assert larger_of_two(-1, -5) == -1
assert larger_of_two(0, -0.5) == 0

def smallest_of_three(a, b, c):
    """Return the smallest of a, b and c."""
    return min(a, b, c)
assert smallest_of_three(1, 2, 3) == 1
assert smallest_of_three(3, 2, 1) == 1
assert smallest_of_three(2, 1, 3) == 1
assert smallest_of_three(5, 5, 5) == 5
assert smallest_of_three(-1, 0, 1) == -1
assert smallest_of_three(4, 4, 2) == 2

def spread_of_three(a, b, c):
    """Return the largest of a, b and c minus the smallest of them."""
    return max(a, b, c) - min(a, b, c)
assert spread_of_three(1, 2, 3) == 2
assert spread_of_three(5, 5, 5) == 0
assert spread_of_three(-1, 0, 1) == 2
assert spread_of_three(10, 2, 7) == 8
assert spread_of_three(3, 9, 3) == 6
assert spread_of_three(5, 9, 1) == 8

def larger_absolute_value(a, b):
    """Return the larger of the absolute values of a and b."""
    return max(abs(a), abs(b))
assert larger_absolute_value(3, -5) == 5
assert larger_absolute_value(-7, 2) == 7
assert larger_absolute_value(0, 0) == 0
assert larger_absolute_value(4, 4) == 4
assert larger_absolute_value(-1, -9) == 9

def sum_of_two_largest(a, b, c):
    """Return the sum of the two largest of a, b and c."""
    return a + b + c - min(a, b, c)
assert sum_of_two_largest(1, 2, 3) == 5
assert sum_of_two_largest(3, 3, 3) == 6
assert sum_of_two_largest(5, 1, 1) == 6
assert sum_of_two_largest(-1, -2, -3) == -3
assert sum_of_two_largest(0, 10, 4) == 14
assert sum_of_two_largest(4, 1, 6) == 10

family = "parity_sign"

def is_even_number(n):
    """Return True if the integer n is even, otherwise False."""
    return n % 2 == 0
assert is_even_number(4) == True
assert is_even_number(7) == False
assert is_even_number(0) == True
assert is_even_number(-2) == True
assert is_even_number(-3) == False

def is_divisible_by_three(n):
    """Return True if the integer n is divisible by 3, otherwise False."""
    return n % 3 == 0
assert is_divisible_by_three(9) == True
assert is_divisible_by_three(10) == False
assert is_divisible_by_three(3) == True
assert is_divisible_by_three(-6) == True
assert is_divisible_by_three(1) == False

def is_positive_number(n):
    """Return True if n is greater than zero, otherwise False."""
    return n > 0
assert is_positive_number(5) == True
assert is_positive_number(-5) == False
assert is_positive_number(0) == False
assert is_positive_number(0.1) == True
assert is_positive_number(-0.1) == False

def is_multiple_of(a, b):
    """Return True if the integer a is a multiple of the positive integer b, otherwise False."""
    return a % b == 0
assert is_multiple_of(10, 5) == True
assert is_multiple_of(10, 3) == False
assert is_multiple_of(12, 4) == True
assert is_multiple_of(7, 7) == True
assert is_multiple_of(-9, 3) == True
assert is_multiple_of(1, 2) == False

def is_not_negative(n):
    """Return True if n is zero or greater, otherwise False."""
    return n >= 0
assert is_not_negative(0) == True
assert is_not_negative(3) == True
assert is_not_negative(-1) == False
assert is_not_negative(-0.5) == False
assert is_not_negative(100) == True

family = "unit_conversion"

def minutes_to_seconds(minutes):
    """Return the number of seconds in the given number of minutes."""
    return minutes * 60
assert minutes_to_seconds(1) == 60
assert minutes_to_seconds(0) == 0
assert minutes_to_seconds(5) == 300
assert minutes_to_seconds(0.5) == 30
assert minutes_to_seconds(60) == 3600

def kilometers_to_meters(kilometers):
    """Return the number of meters in the given number of kilometers."""
    return kilometers * 1000
assert kilometers_to_meters(1) == 1000
assert kilometers_to_meters(0) == 0
assert kilometers_to_meters(3) == 3000
assert kilometers_to_meters(2.5) == 2500
assert kilometers_to_meters(0.5) == 500

def seconds_to_whole_minutes(seconds):
    """Return how many full minutes fit into the given number of seconds."""
    return seconds // 60
assert seconds_to_whole_minutes(60) == 1
assert seconds_to_whole_minutes(59) == 0
assert seconds_to_whole_minutes(0) == 0
assert seconds_to_whole_minutes(125) == 2
assert seconds_to_whole_minutes(3600) == 60

def cents_to_dollars(cents):
    """Return the amount in dollars of the given number of cents, where 100 cents are 1 dollar."""
    return cents / 100
assert cents_to_dollars(100) == 1
assert cents_to_dollars(250) == 2.5
assert cents_to_dollars(0) == 0
assert cents_to_dollars(50) == 0.5
assert cents_to_dollars(1000) == 10

def celsius_to_fahrenheit(celsius):
    """Return the temperature in Fahrenheit, which is celsius times 9 / 5 plus 32."""
    return celsius * 9 / 5 + 32
assert celsius_to_fahrenheit(0) == 32
assert celsius_to_fahrenheit(100) == 212
assert celsius_to_fahrenheit(-40) == -40
assert celsius_to_fahrenheit(10) == 50
assert celsius_to_fahrenheit(2.5) == 36.5

family = "conditional_value"

def absolute_value(n):
    """Return n if n is zero or greater, otherwise return -n."""
    if n < 0:
        return -n
    return n
assert absolute_value(5) == 5
assert absolute_value(-5) == 5
assert absolute_value(0) == 0
assert absolute_value(-0.5) == 0.5
assert absolute_value(0.5) == 0.5
assert absolute_value(-1) == 1

def zero_if_negative(n):
    """Return 0 if n is less than zero, otherwise return n."""
    if n < 0:
        return 0
    return n
assert zero_if_negative(5) == 5
assert zero_if_negative(-3) == 0
assert zero_if_negative(0) == 0
assert zero_if_negative(0.5) == 0.5
assert zero_if_negative(-0.5) == 0

def ticket_price(age):
    """Return 5 if age is less than 12, otherwise return 10."""
    if age < 12:
        return 5
    return 10
assert ticket_price(5) == 5
assert ticket_price(11) == 5
assert ticket_price(12) == 10
assert ticket_price(30) == 10
assert ticket_price(0) == 5

def distance_between(a, b):
    """Return how far apart the numbers a and b are, which is never negative."""
    if a > b:
        return a - b
    return b - a
assert distance_between(5, 3) == 2
assert distance_between(3, 5) == 2
assert distance_between(4, 4) == 0
assert distance_between(-2, 2) == 4
assert distance_between(-5, -1) == 4

def halve_if_even(n):
    """Return half of n if the integer n is even, otherwise return n."""
    if n % 2 == 0:
        return n // 2
    return n
assert halve_if_even(8) == 4
assert halve_if_even(7) == 7
assert halve_if_even(0) == 0
assert halve_if_even(-4) == -2
assert halve_if_even(1) == 1

family = "boolean_logic"

def both_true(a, b):
    """Return True if a and b are both True, otherwise False."""
    return a and b
assert both_true(True, True) == True
assert both_true(True, False) == False
assert both_true(False, True) == False
assert both_true(False, False) == False

def either_true(a, b):
    """Return True if at least one of a and b is True, otherwise False."""
    return a or b
assert either_true(True, True) == True
assert either_true(True, False) == True
assert either_true(False, True) == True
assert either_true(False, False) == False

def neither_true(a, b):
    """Return True if a and b are both False, otherwise False."""
    return not a and not b
assert neither_true(True, True) == False
assert neither_true(True, False) == False
assert neither_true(False, True) == False
assert neither_true(False, False) == True

def exactly_one_true(a, b):
    """Return True if exactly one of a and b is True, otherwise False."""
    return a != b
assert exactly_one_true(True, True) == False
assert exactly_one_true(True, False) == True
assert exactly_one_true(False, True) == True
assert exactly_one_true(False, False) == False

def all_three_true(a, b, c):
    """Return True if a, b and c are all True, otherwise False."""
    return a and b and c
assert all_three_true(True, True, True) == True
assert all_three_true(True, True, False) == False
assert all_three_true(True, False, True) == False
assert all_three_true(False, True, True) == False
assert all_three_true(False, False, False) == False
