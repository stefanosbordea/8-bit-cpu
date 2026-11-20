"""Practice tasks of the near families about range loops and lists (format: see ../build.py)."""

family = "range_loops"

def sum_of_odd_numbers_up_to(n):
    """Return the sum of the odd integers from 1 to n (0 if n is less than 1)."""
    total = 0
    for g in range(1, n + 1, 2):
        total += g
    return total
assert sum_of_odd_numbers_up_to(0) == 0
assert sum_of_odd_numbers_up_to(1) == 1
assert sum_of_odd_numbers_up_to(4) == 4
assert sum_of_odd_numbers_up_to(5) == 9
assert sum_of_odd_numbers_up_to(10) == 25

def product_of_even_numbers_up_to(n):
    """Return the product of the even integers from 2 to n (1 if n is less than 2)."""
    result = 1
    for g in range(2, n + 1, 2):
        result *= g
    return result
assert product_of_even_numbers_up_to(2) == 2
assert product_of_even_numbers_up_to(4) == 8
assert product_of_even_numbers_up_to(7) == 48
assert product_of_even_numbers_up_to(1) == 1
assert product_of_even_numbers_up_to(10) == 3840

def count_odd_numbers_from_to(start, end):
    """Return how many integers from start to end, including both, are odd, where start is not larger than end."""
    count = 0
    for g in range(start, end + 1):
        if g % 2 == 1:
            count += 1
    return count
assert count_odd_numbers_from_to(1, 5) == 3
assert count_odd_numbers_from_to(2, 5) == 2
assert count_odd_numbers_from_to(4, 4) == 0
assert count_odd_numbers_from_to(7, 7) == 1
assert count_odd_numbers_from_to(-3, 3) == 4
assert count_odd_numbers_from_to(0, 10) == 5

def sum_of_multiples_of_five_up_to(n):
    """Return the sum of the integers from 1 to n that are divisible by 5 (0 if there is none)."""
    total = 0
    for g in range(1, n + 1):
        if g % 5 == 0:
            total += g
    return total
assert sum_of_multiples_of_five_up_to(4) == 0
assert sum_of_multiples_of_five_up_to(5) == 5
assert sum_of_multiples_of_five_up_to(12) == 15
assert sum_of_multiples_of_five_up_to(20) == 50
assert sum_of_multiples_of_five_up_to(0) == 0

def sum_of_squares_from_to(start, end):
    """Return the sum of the squares of the integers from start to end, including both, where start is not larger than end."""
    total = 0
    for g in range(start, end + 1):
        total += g * g
    return total
assert sum_of_squares_from_to(1, 3) == 14
assert sum_of_squares_from_to(2, 2) == 4
assert sum_of_squares_from_to(-1, 1) == 2
assert sum_of_squares_from_to(3, 4) == 25
assert sum_of_squares_from_to(0, 5) == 55

family = "list_build"

def numbers_from_to(start, end):
    """Return the list of the integers from start to end, including both, in increasing order, where start is not larger than end."""
    return list(range(start, end + 1))
assert numbers_from_to(1, 3) == [1, 2, 3]
assert numbers_from_to(5, 5) == [5]
assert numbers_from_to(-1, 2) == [-1, 0, 1, 2]
assert numbers_from_to(0, 4) == [0, 1, 2, 3, 4]

def countdown_to_zero(n):
    """Return the list of the integers from n down to 0, where n is not negative."""
    return list(range(n, -1, -1))
assert countdown_to_zero(3) == [3, 2, 1, 0]
assert countdown_to_zero(0) == [0]
assert countdown_to_zero(1) == [1, 0]
assert countdown_to_zero(5) == [5, 4, 3, 2, 1, 0]

def odd_numbers_up_to(n):
    """Return the list of the odd integers from 1 to n in increasing order (an empty list if n is less than 1)."""
    return list(range(1, n + 1, 2))
assert odd_numbers_up_to(5) == [1, 3, 5]
assert odd_numbers_up_to(6) == [1, 3, 5]
assert odd_numbers_up_to(1) == [1]
assert odd_numbers_up_to(0) == []
assert odd_numbers_up_to(2) == [1]

def repeat_pair(a, b, count):
    """Return a list in which a followed by b is repeated count times."""
    return [a, b] * count
assert repeat_pair(1, 2, 2) == [1, 2, 1, 2]
assert repeat_pair('a', 'b', 1) == ['a', 'b']
assert repeat_pair(0, 0, 3) == [0, 0, 0, 0, 0, 0]
assert repeat_pair(5, 6, 0) == []

def cubes_up_to(n):
    """Return the list of the cubes of the integers from 1 to n (an empty list if n is 0)."""
    return [g ** 3 for g in range(1, n + 1)]
assert cubes_up_to(3) == [1, 8, 27]
assert cubes_up_to(1) == [1]
assert cubes_up_to(0) == []
assert cubes_up_to(5) == [1, 8, 27, 64, 125]

family = "list_aggregate"

def sum_minus_smallest(nums):
    """Return the sum of the numbers in the list nums minus its smallest number, where nums is not empty."""
    return sum(nums) - min(nums)
assert sum_minus_smallest([1, 2, 3]) == 5
assert sum_minus_smallest([5]) == 0
assert sum_minus_smallest([2, 2]) == 2
assert sum_minus_smallest([-1, -3]) == -1
assert sum_minus_smallest([10, 0, 5]) == 15

def count_zeros(nums):
    """Return how many numbers in the list nums are equal to zero."""
    return nums.count(0)
assert count_zeros([0, 1, 0]) == 2
assert count_zeros([]) == 0
assert count_zeros([1, 2]) == 0
assert count_zeros([0]) == 1
assert count_zeros([0, -3, 0, 5]) == 2

def count_even_numbers(nums):
    """Return how many of the integers in the list nums are even."""
    return sum(1 for x in nums if x % 2 == 0)
assert count_even_numbers([1, 2, 3, 4]) == 2
assert count_even_numbers([]) == 0
assert count_even_numbers([1, 3]) == 0
assert count_even_numbers([0, -2]) == 2
assert count_even_numbers([2, 2, 5]) == 2

def sum_of_absolute_values(nums):
    """Return the sum of the absolute values of the numbers in the list nums (0 if it is empty)."""
    return sum(abs(x) for x in nums)
assert sum_of_absolute_values([1, -2, 3]) == 6
assert sum_of_absolute_values([]) == 0
assert sum_of_absolute_values([-5]) == 5
assert sum_of_absolute_values([0, 0]) == 0
assert sum_of_absolute_values([10, -20]) == 30

def sum_of_positive_numbers(nums):
    """Return the sum of the numbers in the list nums that are greater than zero (0 if there is none)."""
    return sum(x for x in nums if x > 0)
assert sum_of_positive_numbers([1, -2, 3]) == 4
assert sum_of_positive_numbers([]) == 0
assert sum_of_positive_numbers([-1, -5]) == 0
assert sum_of_positive_numbers([0, 2.5, -0.5]) == 2.5
assert sum_of_positive_numbers([1, 2, 3, 4]) == 10

family = "list_filter"

def keep_multiples_of_three(nums):
    """Return a list of the numbers in the list nums that are divisible by 3, in the same order."""
    return [x for x in nums if x % 3 == 0]
assert keep_multiples_of_three([1, 3, 6, 7]) == [3, 6]
assert keep_multiples_of_three([]) == []
assert keep_multiples_of_three([1, 2]) == []
assert keep_multiples_of_three([9, 9]) == [9, 9]
assert keep_multiples_of_three([0, -3, 4]) == [0, -3]

def keep_large_numbers(nums):
    """Return a list of the numbers in the list nums that are greater than 100, in the same order."""
    return [x for x in nums if x > 100]
assert keep_large_numbers([150, 200, 50, 300]) == [150, 200, 300]
assert keep_large_numbers([]) == []
assert keep_large_numbers([100]) == []
assert keep_large_numbers([101, -200]) == [101]
assert keep_large_numbers([100.5, 99]) == [100.5]

def keep_between(nums, low, high):
    """Return a list of the numbers in the list nums that are at least low and at most high, in the same order."""
    return [x for x in nums if low <= x <= high]
assert keep_between([1, 5, 10], 2, 9) == [5]
assert keep_between([], 0, 1) == []
assert keep_between([1, 2, 3], 1, 3) == [1, 2, 3]
assert keep_between([0, 4], 1, 3) == []
assert keep_between([-5, -1, 0], -2, 0) == [-1, 0]

def remove_value(items, target):
    """Return the list items without the items that are equal to target, keeping the order of the others."""
    return [x for x in items if x != target]
assert remove_value([1, 2, 1, 3], 1) == [2, 3]
assert remove_value([], 1) == []
assert remove_value([1, 2], 5) == [1, 2]
assert remove_value([2, 1, 2, 2], 2) == [1]
assert remove_value(['a', 'b'], 'b') == ['a']

def keep_long_words(words):
    """Return a list of the strings in the list words that have at least 5 characters, in the same order."""
    return [word for word in words if len(word) >= 5]
assert keep_long_words(['apple', 'fig', 'banana']) == ['apple', 'banana']
assert keep_long_words([]) == []
assert keep_long_words(['four']) == []
assert keep_long_words(['seven', 'six']) == ['seven']
assert keep_long_words(['a', 'abcdef', 'abcde']) == ['abcdef', 'abcde']

family = "list_map"

def halve_each(nums):
    """Return a list with every number of the list nums divided by 2, so 3 becomes 1.5."""
    return [x / 2 for x in nums]
assert halve_each([2, 4, 6]) == [1, 2, 3]
assert halve_each([]) == []
assert halve_each([3]) == [1.5]
assert halve_each([0, -1]) == [0, -0.5]
assert halve_each([10, 5]) == [5, 2.5]

def multiply_each(nums, factor):
    """Return a list with every number of the list nums multiplied by factor."""
    return [x * factor for x in nums]
assert multiply_each([1, 2], 3) == [3, 6]
assert multiply_each([], 5) == []
assert multiply_each([4], 0) == [0]
assert multiply_each([-1, 1], -2) == [2, -2]
assert multiply_each([2, 3], 0.5) == [1, 1.5]

def absolute_value_of_each(nums):
    """Return a list with the absolute value of every number of the list nums."""
    return [abs(x) for x in nums]
assert absolute_value_of_each([1, -2, 0]) == [1, 2, 0]
assert absolute_value_of_each([]) == []
assert absolute_value_of_each([-5]) == [5]
assert absolute_value_of_each([-1.5, 3]) == [1.5, 3]

def uppercase_each(words):
    """Return a list with every string of the list words in upper case."""
    return [word.upper() for word in words]
assert uppercase_each(['a', 'bc']) == ['A', 'BC']
assert uppercase_each([]) == []
assert uppercase_each(['Hi', 'YOU']) == ['HI', 'YOU']
assert uppercase_each(['x y']) == ['X Y']

def numbers_to_strings(nums):
    """Return a list with every integer of the list nums written as a string."""
    return [str(x) for x in nums]
assert numbers_to_strings([1, 2]) == ['1', '2']
assert numbers_to_strings([]) == []
assert numbers_to_strings([-5]) == ['-5']
assert numbers_to_strings([10, 0, 7]) == ['10', '0', '7']

family = "list_index"

def third_item(items):
    """Return the third item of the list items, which has at least three items."""
    return items[2]
assert third_item([1, 2, 3]) == 3
assert third_item(['a', 'b', 'c', 'd']) == 'c'
assert third_item([5, 4, 3, 2, 1]) == 3
assert third_item([0, 0, 7, 0]) == 7

def item_from_end(items, n):
    """Return the item that is n places from the end of the list items, where n is 1 for the last item, 2 for the item before it and so on."""
    return items[-n]
assert item_from_end([1, 2, 3], 1) == 3
assert item_from_end([1, 2, 3], 2) == 2
assert item_from_end([1, 2, 3], 3) == 1
assert item_from_end(['a', 'b'], 1) == 'b'
assert item_from_end([5, 6, 7, 8], 2) == 7

def last_minus_first(nums):
    """Return the last number of the list nums minus its first number, where nums is not empty."""
    return nums[-1] - nums[0]
assert last_minus_first([1, 2, 5]) == 4
assert last_minus_first([7]) == 0
assert last_minus_first([5, 1]) == -4
assert last_minus_first([-2, 9, 3]) == 5

def first_and_last(items):
    """Return a list with the first item and the last item of the list items, which is not empty."""
    return [items[0], items[-1]]
assert first_and_last([1, 2, 3]) == [1, 3]
assert first_and_last([5]) == [5, 5]
assert first_and_last(['a', 'b']) == ['a', 'b']
assert first_and_last([4, 0, 0, 9]) == [4, 9]

def product_of_first_two(nums):
    """Return the first number of the list nums times its second number, where nums has at least two numbers."""
    return nums[0] * nums[1]
assert product_of_first_two([2, 3, 4]) == 6
assert product_of_first_two([5, 0]) == 0
assert product_of_first_two([-1, 7, 2]) == -7
assert product_of_first_two([3, 3, 3, 3]) == 9

family = "list_slice"

def skip_first_n(items, n):
    """Return the list items without its first n items, where n is not larger than its length."""
    return items[n:]
assert skip_first_n([1, 2, 3], 1) == [2, 3]
assert skip_first_n([1, 2, 3], 0) == [1, 2, 3]
assert skip_first_n([1, 2, 3], 3) == []
assert skip_first_n([], 0) == []
assert skip_first_n(['a', 'b', 'c', 'd'], 2) == ['c', 'd']

def without_last_n(items, n):
    """Return the list items without its last n items, where n is at least 1 and not larger than its length."""
    return items[:-n]
assert without_last_n([1, 2, 3], 1) == [1, 2]
assert without_last_n([1, 2, 3], 2) == [1]
assert without_last_n([1, 2, 3], 3) == []
assert without_last_n(['a', 'b'], 1) == ['a']
assert without_last_n([5, 6, 7, 8], 3) == [5]

def last_two_items(items):
    """Return a list with the last two items of the list items, which has at least two items."""
    return items[-2:]
assert last_two_items([1, 2, 3]) == [2, 3]
assert last_two_items([1, 2]) == [1, 2]
assert last_two_items(['a', 'b', 'c', 'd']) == ['c', 'd']
assert last_two_items([0, 0, 1]) == [0, 1]

def first_half(items):
    """Return a list with the first half of the list items, which has an even number of items."""
    return items[:len(items) // 2]
assert first_half([1, 2, 3, 4]) == [1, 2]
assert first_half([1, 2]) == [1]
assert first_half([]) == []
assert first_half(['a', 'b', 'c', 'd', 'e', 'f']) == ['a', 'b', 'c']

def every_second_from_second(items):
    """Return a list with every second item of the list items, starting with the second item."""
    return items[1::2]
assert every_second_from_second([1, 2, 3, 4]) == [2, 4]
assert every_second_from_second([1]) == []
assert every_second_from_second([]) == []
assert every_second_from_second([1, 2, 3]) == [2]
assert every_second_from_second(['a', 'b']) == ['b']

family = "list_predicates"

def contains_both(items, first, second):
    """Return True if first and second are both items of the list items, otherwise False."""
    return first in items and second in items
assert contains_both([1, 2, 3], 1, 3) == True
assert contains_both([1, 2, 3], 1, 5) == False
assert contains_both([1, 2, 3], 5, 1) == False
assert contains_both([], 1, 2) == False
assert contains_both(['a', 'b'], 'b', 'b') == True

def has_one_item(items):
    """Return True if the list items has exactly one item, otherwise False."""
    return len(items) == 1
assert has_one_item([5]) == True
assert has_one_item([]) == False
assert has_one_item([1, 2]) == False
assert has_one_item([[]]) == True
assert has_one_item([0, 0, 0]) == False

def all_even_numbers(nums):
    """Return True if every number in the list nums is even (True for an empty list), otherwise False."""
    return all(x % 2 == 0 for x in nums)
assert all_even_numbers([2, 4]) == True
assert all_even_numbers([2, 3]) == False
assert all_even_numbers([]) == True
assert all_even_numbers([1]) == False
assert all_even_numbers([0, -2]) == True
assert all_even_numbers([1, 2]) == False

def any_greater_than(nums, limit):
    """Return True if at least one number in the list nums is greater than limit, otherwise False."""
    return any(x > limit for x in nums)
assert any_greater_than([1, 5, 2], 4) == True
assert any_greater_than([1, 2], 2) == False
assert any_greater_than([], 0) == False
assert any_greater_than([-1, -3], -2) == True
assert any_greater_than([7], 7) == False

def has_at_least(items, n):
    """Return True if the list items has n or more items, otherwise False."""
    return len(items) >= n
assert has_at_least([4, 5, 6], 2) == True
assert has_at_least([0, 0], 2) == True
assert has_at_least([9], 2) == False
assert has_at_least([], 0) == True
assert has_at_least([], 1) == False

family = "list_edit"

def append_two_items(items, a, b):
    """Return a new list with the items of the list items followed by a and then b."""
    return items + [a, b]
assert append_two_items([1], 2, 3) == [1, 2, 3]
assert append_two_items([], 1, 2) == [1, 2]
assert append_two_items(['a', 'b'], 'c', 'd') == ['a', 'b', 'c', 'd']
assert append_two_items([0], 0, 1) == [0, 0, 1]

def add_first_and_last(items, first, last):
    """Return a new list with first, then the items of the list items, then last."""
    return [first] + items + [last]
assert add_first_and_last([2, 3], 1, 4) == [1, 2, 3, 4]
assert add_first_and_last([], 5, 6) == [5, 6]
assert add_first_and_last(['b'], 'a', 'c') == ['a', 'b', 'c']
assert add_first_and_last([0], 1, 0) == [1, 0, 0]

def replace_last(items, item):
    """Return a new list like the list items but with item in place of its last item, where items is not empty."""
    return items[:-1] + [item]
assert replace_last([1, 2, 3], 9) == [1, 2, 9]
assert replace_last([1], 5) == [5]
assert replace_last(['a', 'b'], 'c') == ['a', 'c']
assert replace_last([0, 0], 1) == [0, 1]

def insert_item_at(items, index, item):
    """Return a new list like the list items but with item put in at the given index (counting from 0), so that the items from that index on move one place back."""
    return items[:index] + [item] + items[index:]
assert insert_item_at([1, 2, 3], 0, 9) == [9, 1, 2, 3]
assert insert_item_at([1, 2, 3], 1, 9) == [1, 9, 2, 3]
assert insert_item_at([1, 2, 3], 3, 9) == [1, 2, 3, 9]
assert insert_item_at([], 0, 5) == [5]
assert insert_item_at(['a', 'c'], 1, 'b') == ['a', 'b', 'c']

def replace_item_at(items, index, item):
    """Return a new list like the list items but with item in place of the item at the given index (counting from 0)."""
    return items[:index] + [item] + items[index + 1:]
assert replace_item_at([1, 2, 3], 0, 9) == [9, 2, 3]
assert replace_item_at([1, 2, 3], 1, 9) == [1, 9, 3]
assert replace_item_at([1, 2, 3], 2, 9) == [1, 2, 9]
assert replace_item_at([5], 0, 6) == [6]
assert replace_item_at(['a', 'b'], 1, 'c') == ['a', 'c']
