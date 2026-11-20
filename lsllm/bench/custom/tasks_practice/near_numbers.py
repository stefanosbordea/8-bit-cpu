"""Practice tasks of the near families about numbers and booleans (format: see ../build.py)."""

family = "arith_one"

def times_ten(n):
    """Return n multiplied by 10."""
    return n * 10
assert times_ten(2) == 20
assert times_ten(0) == 0
assert times_ten(-4) == -40
assert times_ten(1.5) == 15
assert times_ten(10) == 100

def fourth_power(n):
    """Return n to the power 4, which is n times n times n times n."""
    return n ** 4
assert fourth_power(2) == 16
assert fourth_power(0) == 0
assert fourth_power(-3) == 81
assert fourth_power(1) == 1
assert fourth_power(10) == 10000

def subtract_five(n):
    """Return n minus 5."""
    return n - 5
assert subtract_five(8) == 3
assert subtract_five(5) == 0
assert subtract_five(0) == -5
assert subtract_five(-5) == -10
assert subtract_five(7.5) == 2.5

def quarter_of_number(n):
    """Return n divided by 4, so 6 gives 1.5."""
    return n / 4
assert quarter_of_number(8) == 2
assert quarter_of_number(0) == 0
assert quarter_of_number(6) == 1.5
assert quarter_of_number(-2) == -0.5
assert quarter_of_number(1) == 0.25

def subtract_from_hundred(n):
    """Return 100 minus n."""
    return 100 - n
assert subtract_from_hundred(30) == 70
assert subtract_from_hundred(100) == 0
assert subtract_from_hundred(0) == 100
assert subtract_from_hundred(150) == -50
assert subtract_from_hundred(-20) == 120

family = "arith_two"

def divide_numbers(a, b):
    """Return a divided by b, where b is not zero, so 7 and 2 give 3.5."""
    return a / b
assert divide_numbers(8, 2) == 4
assert divide_numbers(7, 2) == 3.5
assert divide_numbers(0, 5) == 0
assert divide_numbers(-9, 3) == -3
assert divide_numbers(1, 4) == 0.25

def average_of_two(a, b):
    """Return the average of a and b, which is their sum divided by 2."""
    return (a + b) / 2
assert average_of_two(2, 4) == 3
assert average_of_two(1, 2) == 1.5
assert average_of_two(0, 0) == 0
assert average_of_two(-3, 3) == 0
assert average_of_two(10, 5) == 7.5

def sum_of_two_squares(a, b):
    """Return the square of a plus the square of b."""
    return a * a + b * b
assert sum_of_two_squares(1, 2) == 5
assert sum_of_two_squares(3, 4) == 25
assert sum_of_two_squares(0, 0) == 0
assert sum_of_two_squares(-2, 1) == 5
assert sum_of_two_squares(0, -3) == 9

def square_of_sum(a, b):
    """Return the square of the sum of a and b."""
    return (a + b) ** 2
assert square_of_sum(1, 2) == 9
assert square_of_sum(3, 4) == 49
assert square_of_sum(0, 0) == 0
assert square_of_sum(-2, 2) == 0
assert square_of_sum(-3, 1) == 4

def total_minutes(hours, minutes):
    """Return the total number of minutes in the given number of hours plus the given number of minutes."""
    return hours * 60 + minutes
assert total_minutes(1, 30) == 90
assert total_minutes(0, 45) == 45
assert total_minutes(2, 0) == 120
assert total_minutes(0, 0) == 0
assert total_minutes(10, 5) == 605

family = "arith_three"

def subtract_both(a, b, c):
    """Return a minus b minus c."""
    return a - b - c
assert subtract_both(10, 3, 2) == 5
assert subtract_both(0, 0, 0) == 0
assert subtract_both(5, 5, 5) == -5
assert subtract_both(1, -1, 2) == 0
assert subtract_both(7.5, 2, 0.5) == 5

def multiply_then_divide(a, b, c):
    """Return a times b, divided by c, where c is not zero."""
    return a * b / c
assert multiply_then_divide(2, 3, 4) == 1.5
assert multiply_then_divide(4, 5, 2) == 10
assert multiply_then_divide(0, 7, 3) == 0
assert multiply_then_divide(3, 3, 2) == 4.5
assert multiply_then_divide(-2, 6, 4) == -3

def change_due(paid, price, count):
    """Return paid minus the product of price and count."""
    return paid - price * count
assert change_due(20, 3, 5) == 5
assert change_due(10, 2, 5) == 0
assert change_due(50, 0, 3) == 50
assert change_due(5, 4, 2) == -3
assert change_due(10, 2.5, 2) == 5

def subtract_then_multiply(a, b, c):
    """Return the result of a minus b, multiplied by c."""
    return (a - b) * c
assert subtract_then_multiply(5, 3, 4) == 8
assert subtract_then_multiply(3, 5, 2) == -4
assert subtract_then_multiply(2, 2, 9) == 0
assert subtract_then_multiply(10, 4, 0.5) == 3
assert subtract_then_multiply(1, 2, -3) == 3

def sum_divided_by(a, b, c):
    """Return the sum of a and b, divided by c, where c is not zero."""
    return (a + b) / c
assert sum_divided_by(4, 2, 2) == 3
assert sum_divided_by(1, 2, 2) == 1.5
assert sum_divided_by(0, 0, 5) == 0
assert sum_divided_by(-3, 1, 4) == -0.5
assert sum_divided_by(5, 5, 1) == 10

family = "compare"

def is_less(a, b):
    """Return True if a is less than b, otherwise False."""
    return a < b
assert is_less(1, 2) == True
assert is_less(2, 1) == False
assert is_less(2, 2) == False
assert is_less(-3, -2) == True
assert is_less(0.5, 0) == False
assert is_less(1, 5) == True

def is_at_least(a, b):
    """Return True if a is greater than or equal to b, otherwise False."""
    return a >= b
assert is_at_least(3, 2) == True
assert is_at_least(2, 2) == True
assert is_at_least(1, 2) == False
assert is_at_least(-5, -5) == True
assert is_at_least(-1, 0) == False

def is_increasing_three(a, b, c):
    """Return True if a is less than b and b is less than c, otherwise False."""
    return a < b < c
assert is_increasing_three(1, 2, 3) == True
assert is_increasing_three(1, 1, 2) == False
assert is_increasing_three(1, 2, 2) == False
assert is_increasing_three(3, 2, 1) == False
assert is_increasing_three(1, 3, 2) == False
assert is_increasing_three(-2, 0, 5) == True

def equals_sum(a, b, total):
    """Return True if a plus b is equal to total, otherwise False."""
    return a + b == total
assert equals_sum(1, 2, 3) == True
assert equals_sum(1, 2, 4) == False
assert equals_sum(0, 0, 0) == True
assert equals_sum(-1, 1, 0) == True
assert equals_sum(2, 2, 2) == False
assert equals_sum(5, -2, 7) == False

def is_first_largest(a, b, c):
    """Return True if a is greater than b and greater than c, otherwise False."""
    return a > b and a > c
assert is_first_largest(5, 1, 2) == True
assert is_first_largest(3, 2, 1) == True
assert is_first_largest(2, 3, 1) == False
assert is_first_largest(2, 1, 3) == False
assert is_first_largest(2, 2, 1) == False
assert is_first_largest(2, 1, 2) == False

family = "min_max_few"

def largest_of_three(a, b, c):
    """Return the largest of a, b and c."""
    return max(a, b, c)
assert largest_of_three(1, 2, 3) == 3
assert largest_of_three(3, 2, 1) == 3
assert largest_of_three(2, 3, 1) == 3
assert largest_of_three(5, 5, 5) == 5
assert largest_of_three(-1, -2, -3) == -1
assert largest_of_three(4, 2, 4) == 4

def smallest_of_four(a, b, c, d):
    """Return the smallest of a, b, c and d."""
    return min(a, b, c, d)
assert smallest_of_four(1, 2, 3, 4) == 1
assert smallest_of_four(4, 3, 2, 1) == 1
assert smallest_of_four(2, 1, 4, 3) == 1
assert smallest_of_four(3, 4, 1, 2) == 1
assert smallest_of_four(5, 5, 5, 5) == 5
assert smallest_of_four(0, -1, -7, -2) == -7

def sum_of_largest_and_smallest(a, b, c):
    """Return the largest of a, b and c plus the smallest of them."""
    return max(a, b, c) + min(a, b, c)
assert sum_of_largest_and_smallest(1, 2, 3) == 4
assert sum_of_largest_and_smallest(5, 5, 5) == 10
assert sum_of_largest_and_smallest(-1, 0, 1) == 0
assert sum_of_largest_and_smallest(10, 2, 7) == 12
assert sum_of_largest_and_smallest(4, 9, 3) == 12

def larger_of_sum_and_product(a, b):
    """Return the sum of a and b or the product of a and b, whichever is larger."""
    return max(a + b, a * b)
assert larger_of_sum_and_product(2, 3) == 6
assert larger_of_sum_and_product(1, 5) == 6
assert larger_of_sum_and_product(0, 4) == 4
assert larger_of_sum_and_product(2, 2) == 4
assert larger_of_sum_and_product(-2, -3) == 6
assert larger_of_sum_and_product(-1, 3) == 2

def sum_of_two_smallest(a, b, c):
    """Return the sum of the two smallest of a, b and c."""
    return a + b + c - max(a, b, c)
assert sum_of_two_smallest(1, 2, 3) == 3
assert sum_of_two_smallest(3, 3, 3) == 6
assert sum_of_two_smallest(5, 1, 1) == 2
assert sum_of_two_smallest(-1, -2, -3) == -5
assert sum_of_two_smallest(0, 10, 4) == 4

family = "parity_sign"

def is_odd_number(n):
    """Return True if the integer n is odd, otherwise False."""
    return n % 2 == 1
assert is_odd_number(7) == True
assert is_odd_number(4) == False
assert is_odd_number(0) == False
assert is_odd_number(-3) == True
assert is_odd_number(-2) == False
assert is_odd_number(1) == True

def is_multiple_of_ten(n):
    """Return True if the integer n is divisible by 10, otherwise False."""
    return n % 10 == 0
assert is_multiple_of_ten(20) == True
assert is_multiple_of_ten(25) == False
assert is_multiple_of_ten(0) == True
assert is_multiple_of_ten(-30) == True
assert is_multiple_of_ten(12) == False
assert is_multiple_of_ten(101) == False

def is_negative_number(n):
    """Return True if n is less than zero, otherwise False."""
    return n < 0
assert is_negative_number(-5) == True
assert is_negative_number(5) == False
assert is_negative_number(0) == False
assert is_negative_number(-0.1) == True
assert is_negative_number(0.1) == False

def is_not_positive(n):
    """Return True if n is zero or less, otherwise False."""
    return n <= 0
assert is_not_positive(0) == True
assert is_not_positive(-3) == True
assert is_not_positive(1) == False
assert is_not_positive(0.5) == False
assert is_not_positive(-100) == True

def same_parity(a, b):
    """Return True if the integers a and b are both even or both odd, otherwise False."""
    return a % 2 == b % 2
assert same_parity(2, 4) == True
assert same_parity(1, 3) == True
assert same_parity(1, 2) == False
assert same_parity(0, 7) == False
assert same_parity(-1, 5) == True
assert same_parity(-2, 0) == True

family = "unit_conversion"

def days_to_hours(days):
    """Return the number of hours in the given number of days."""
    return days * 24
assert days_to_hours(1) == 24
assert days_to_hours(0) == 0
assert days_to_hours(7) == 168
assert days_to_hours(0.5) == 12
assert days_to_hours(30) == 720

def meters_to_centimeters(meters):
    """Return the number of centimeters in the given number of meters."""
    return meters * 100
assert meters_to_centimeters(1) == 100
assert meters_to_centimeters(0) == 0
assert meters_to_centimeters(3) == 300
assert meters_to_centimeters(2.5) == 250
assert meters_to_centimeters(0.25) == 25

def days_to_whole_weeks(days):
    """Return how many full weeks fit into the given number of days."""
    return days // 7
assert days_to_whole_weeks(7) == 1
assert days_to_whole_weeks(6) == 0
assert days_to_whole_weeks(0) == 0
assert days_to_whole_weeks(15) == 2
assert days_to_whole_weeks(70) == 10

def grams_to_kilograms(grams):
    """Return the weight in kilograms of the given number of grams, where 1000 grams are 1 kilogram."""
    return grams / 1000
assert grams_to_kilograms(1000) == 1
assert grams_to_kilograms(2500) == 2.5
assert grams_to_kilograms(0) == 0
assert grams_to_kilograms(500) == 0.5
assert grams_to_kilograms(250) == 0.25

def fahrenheit_to_celsius(fahrenheit):
    """Return the temperature in Celsius, which is fahrenheit minus 32, then times 5 / 9."""
    return (fahrenheit - 32) * 5 / 9
assert fahrenheit_to_celsius(32) == 0
assert fahrenheit_to_celsius(212) == 100
assert fahrenheit_to_celsius(-40) == -40
assert fahrenheit_to_celsius(50) == 10
assert fahrenheit_to_celsius(41) == 5
assert fahrenheit_to_celsius(36.5) == 2.5

family = "conditional_value"

def cap_at_hundred(n):
    """Return 100 if n is greater than 100, otherwise return n."""
    if n > 100:
        return 100
    return n
assert cap_at_hundred(150) == 100
assert cap_at_hundred(100) == 100
assert cap_at_hundred(99) == 99
assert cap_at_hundred(99.5) == 99.5
assert cap_at_hundred(-5) == -5
assert cap_at_hundred(100.5) == 100

def shipping_fee(total):
    """Return 0 if total is at least 50, otherwise return 5."""
    if total >= 50:
        return 0
    return 5
assert shipping_fee(60) == 0
assert shipping_fee(50) == 0
assert shipping_fee(49) == 5
assert shipping_fee(0) == 5
assert shipping_fee(49.5) == 5

def difference_or_zero(a, b):
    """Return a minus b if a is greater than b, otherwise return 0."""
    if a > b:
        return a - b
    return 0
assert difference_or_zero(5, 3) == 2
assert difference_or_zero(3, 5) == 0
assert difference_or_zero(4, 4) == 0
assert difference_or_zero(-1, -4) == 3
assert difference_or_zero(0, 2.5) == 0

def double_if_odd(n):
    """Return n times 2 if the integer n is odd, otherwise return n."""
    if n % 2 == 1:
        return n * 2
    return n
assert double_if_odd(3) == 6
assert double_if_odd(4) == 4
assert double_if_odd(0) == 0
assert double_if_odd(-1) == -2
assert double_if_odd(1) == 2
assert double_if_odd(10) == 10

def round_up_to_even(n):
    """Return n if the integer n is even, otherwise return n plus 1."""
    if n % 2 == 0:
        return n
    return n + 1
assert round_up_to_even(4) == 4
assert round_up_to_even(5) == 6
assert round_up_to_even(0) == 0
assert round_up_to_even(-3) == -2
assert round_up_to_even(1) == 2

family = "boolean_logic"

def not_both_true(a, b):
    """Return False if a and b are both True, otherwise True."""
    return not (a and b)
assert not_both_true(True, True) == False
assert not_both_true(True, False) == True
assert not_both_true(False, True) == True
assert not_both_true(False, False) == True

def any_of_three_true(a, b, c):
    """Return True if at least one of a, b and c is True, otherwise False."""
    return a or b or c
assert any_of_three_true(False, False, False) == False
assert any_of_three_true(True, False, False) == True
assert any_of_three_true(False, True, False) == True
assert any_of_three_true(False, False, True) == True
assert any_of_three_true(True, True, True) == True

def none_of_three_true(a, b, c):
    """Return True if a, b and c are all False, otherwise False."""
    return not (a or b or c)
assert none_of_three_true(False, False, False) == True
assert none_of_three_true(True, False, False) == False
assert none_of_three_true(False, True, False) == False
assert none_of_three_true(False, False, True) == False
assert none_of_three_true(True, True, True) == False
assert none_of_three_true(True, False, True) == False

def all_four_true(a, b, c, d):
    """Return True if a, b, c and d are all True, otherwise False."""
    return a and b and c and d
assert all_four_true(True, True, True, True) == True
assert all_four_true(False, True, True, True) == False
assert all_four_true(True, False, True, True) == False
assert all_four_true(True, True, False, True) == False
assert all_four_true(True, True, True, False) == False
assert all_four_true(False, False, False, False) == False

def first_and_either_other(a, b, c):
    """Return True if a is True and at least one of b and c is True, otherwise False."""
    return a and (b or c)
assert first_and_either_other(True, True, False) == True
assert first_and_either_other(True, False, True) == True
assert first_and_either_other(True, False, False) == False
assert first_and_either_other(False, True, True) == False
assert first_and_either_other(False, False, True) == False
assert first_and_either_other(True, True, True) == True
