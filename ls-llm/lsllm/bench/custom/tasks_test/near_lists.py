"""Test tasks of the near families about range loops and lists (format: see ../build.py)."""

family = "range_loops"

def sum_of_squares_up_to(n):
    """Return the sum of the squares of the integers from 1 to n (0 if n is 0)."""
    total = 0
    for g in range(1, n + 1):
        total += g * g
    return total
assert sum_of_squares_up_to(1) == 1
assert sum_of_squares_up_to(2) == 5
assert sum_of_squares_up_to(3) == 14
assert sum_of_squares_up_to(0) == 0
assert sum_of_squares_up_to(10) == 385

def product_up_to(n):
    """Return the product of the integers from 1 to n (1 if n is 0)."""
    result = 1
    for g in range(1, n + 1):
        result *= g
    return result
assert product_up_to(1) == 1
assert product_up_to(2) == 2
assert product_up_to(3) == 6
assert product_up_to(5) == 120
assert product_up_to(0) == 1

def count_multiples_of_three(n):
    """Return how many integers from 1 to n are divisible by 3."""
    count = 0
    for g in range(1, n + 1):
        if g % 3 == 0:
            count += 1
    return count
assert count_multiples_of_three(3) == 1
assert count_multiples_of_three(2) == 0
assert count_multiples_of_three(10) == 3
assert count_multiples_of_three(9) == 3
assert count_multiples_of_three(1) == 0
assert count_multiples_of_three(30) == 10

def sum_of_even_numbers_up_to(n):
    """Return the sum of the even integers from 2 to n (0 if n is less than 2)."""
    total = 0
    for g in range(2, n + 1, 2):
        total += g
    return total
assert sum_of_even_numbers_up_to(1) == 0
assert sum_of_even_numbers_up_to(2) == 2
assert sum_of_even_numbers_up_to(5) == 6
assert sum_of_even_numbers_up_to(10) == 30
assert sum_of_even_numbers_up_to(6) == 12

def sum_from_to(start, end):
    """Return the sum of the integers from start to end, including both, where start is not larger than end."""
    total = 0
    for g in range(start, end + 1):
        total += g
    return total
assert sum_from_to(1, 3) == 6
assert sum_from_to(5, 5) == 5
assert sum_from_to(0, 10) == 55
assert sum_from_to(-2, 2) == 0
assert sum_from_to(3, 4) == 7

family = "list_build"

def numbers_up_to(n):
    """Return the list of the integers from 1 to n in increasing order (an empty list if n is 0)."""
    return list(range(1, n + 1))
assert numbers_up_to(3) == [1, 2, 3]
assert numbers_up_to(1) == [1]
assert numbers_up_to(0) == []
assert numbers_up_to(5) == [1, 2, 3, 4, 5]

def countdown_from(n):
    """Return the list of the integers from n down to 1 (an empty list if n is 0)."""
    return list(range(n, 0, -1))
assert countdown_from(3) == [3, 2, 1]
assert countdown_from(1) == [1]
assert countdown_from(0) == []
assert countdown_from(5) == [5, 4, 3, 2, 1]

def even_numbers_up_to(n):
    """Return the list of the even integers from 2 to n in increasing order (an empty list if n is less than 2)."""
    return list(range(2, n + 1, 2))
assert even_numbers_up_to(6) == [2, 4, 6]
assert even_numbers_up_to(7) == [2, 4, 6]
assert even_numbers_up_to(2) == [2]
assert even_numbers_up_to(1) == []
assert even_numbers_up_to(0) == []

def repeat_value(value, count):
    """Return a list made of count copies of value."""
    return [value] * count
assert repeat_value(0, 3) == [0, 0, 0]
assert repeat_value('a', 2) == ['a', 'a']
assert repeat_value(5, 1) == [5]
assert repeat_value(7, 0) == []

def squares_up_to(n):
    """Return the list of the squares of the integers from 1 to n (an empty list if n is 0)."""
    return [g * g for g in range(1, n + 1)]
assert squares_up_to(3) == [1, 4, 9]
assert squares_up_to(1) == [1]
assert squares_up_to(0) == []
assert squares_up_to(5) == [1, 4, 9, 16, 25]

family = "list_aggregate"

def sum_minus_largest(nums):
    """Return the sum of the numbers in the list nums minus its largest number, where nums is not empty."""
    return sum(nums) - max(nums)
assert sum_minus_largest([1, 2, 3]) == 3
assert sum_minus_largest([5]) == 0
assert sum_minus_largest([2, 2]) == 2
assert sum_minus_largest([-1, -3]) == -3
assert sum_minus_largest([10, 0, 5]) == 5

def largest_in_list(nums):
    """Return the largest number in the list nums, which is not empty."""
    return max(nums)
assert largest_in_list([1, 3, 2]) == 3
assert largest_in_list([5]) == 5
assert largest_in_list([-1, -3]) == -1
assert largest_in_list([2, 2]) == 2
assert largest_in_list([0, -1, 7, 7]) == 7

def count_positive_numbers(nums):
    """Return how many numbers in the list nums are greater than zero."""
    return sum(1 for x in nums if x > 0)
assert count_positive_numbers([1, -2, 3]) == 2
assert count_positive_numbers([]) == 0
assert count_positive_numbers([0]) == 0
assert count_positive_numbers([-1, -2]) == 0
assert count_positive_numbers([5, 5, 0.5]) == 3

def sum_of_squares_in_list(nums):
    """Return the sum of the squares of the numbers in the list nums (0 if it is empty)."""
    return sum(x * x for x in nums)
assert sum_of_squares_in_list([1, 2, 3]) == 14
assert sum_of_squares_in_list([]) == 0
assert sum_of_squares_in_list([-2]) == 4
assert sum_of_squares_in_list([0, 5]) == 25
assert sum_of_squares_in_list([1, 1, 1, 1]) == 4

def total_length_of_words(words):
    """Return the total number of characters in all the strings of the list words."""
    return sum(len(word) for word in words)
assert total_length_of_words(['a', 'bc']) == 3
assert total_length_of_words([]) == 0
assert total_length_of_words(['hello']) == 5
assert total_length_of_words(['', 'x']) == 1
assert total_length_of_words(['ab', 'cd', 'ef']) == 6

family = "list_filter"

def keep_even_numbers(nums):
    """Return a list of the even numbers in the list nums, in the same order."""
    return [x for x in nums if x % 2 == 0]
assert keep_even_numbers([1, 2, 3, 4]) == [2, 4]
assert keep_even_numbers([]) == []
assert keep_even_numbers([1, 3]) == []
assert keep_even_numbers([2, 2]) == [2, 2]
assert keep_even_numbers([0, -2, 5]) == [0, -2]

def keep_negative_numbers(nums):
    """Return a list of the numbers in the list nums that are less than zero, in the same order."""
    return [x for x in nums if x < 0]
assert keep_negative_numbers([1, -2, 3, -4]) == [-2, -4]
assert keep_negative_numbers([]) == []
assert keep_negative_numbers([0, 1]) == []
assert keep_negative_numbers([-1]) == [-1]
assert keep_negative_numbers([-0.5, 0.5]) == [-0.5]

def keep_greater_than(nums, limit):
    """Return a list of the numbers in the list nums that are greater than limit, in the same order."""
    return [x for x in nums if x > limit]
assert keep_greater_than([1, 5, 3], 2) == [5, 3]
assert keep_greater_than([], 0) == []
assert keep_greater_than([1, 2], 5) == []
assert keep_greater_than([3, 3], 3) == []
assert keep_greater_than([-1, 0, 1], -1) == [0, 1]

def remove_zeros(nums):
    """Return the list nums without its zeros, keeping the order of the other numbers."""
    return [x for x in nums if x != 0]
assert remove_zeros([0, 1, 0, 2]) == [1, 2]
assert remove_zeros([]) == []
assert remove_zeros([0, 0]) == []
assert remove_zeros([1, 2]) == [1, 2]
assert remove_zeros([-1, 0]) == [-1]

def keep_short_words(words):
    """Return a list of the strings in the list words that have at most 3 characters, in the same order."""
    return [word for word in words if len(word) <= 3]
assert keep_short_words(['a', 'abcd', 'abc']) == ['a', 'abc']
assert keep_short_words([]) == []
assert keep_short_words(['hello']) == []
assert keep_short_words(['', 'no']) == ['', 'no']
assert keep_short_words(['four', 'six', 'ten']) == ['six', 'ten']

family = "list_map"

def double_each(nums):
    """Return a list with every number of the list nums multiplied by 2."""
    return [x * 2 for x in nums]
assert double_each([1, 2, 3]) == [2, 4, 6]
assert double_each([]) == []
assert double_each([0]) == [0]
assert double_each([-1, 5]) == [-2, 10]
assert double_each([0.5]) == [1]

def square_each(nums):
    """Return a list with the square of every number of the list nums."""
    return [x * x for x in nums]
assert square_each([1, 2, 3]) == [1, 4, 9]
assert square_each([]) == []
assert square_each([-2]) == [4]
assert square_each([0, 10]) == [0, 100]

def add_to_each(nums, amount):
    """Return a list with amount added to every number of the list nums."""
    return [x + amount for x in nums]
assert add_to_each([1, 2], 3) == [4, 5]
assert add_to_each([], 5) == []
assert add_to_each([0], 0) == [0]
assert add_to_each([-1, 1], 1) == [0, 2]
assert add_to_each([10], -10) == [0]

def lengths_of_words(words):
    """Return a list with the length of every string of the list words."""
    return [len(word) for word in words]
assert lengths_of_words(['a', 'bc']) == [1, 2]
assert lengths_of_words([]) == []
assert lengths_of_words(['']) == [0]
assert lengths_of_words(['hello', 'hi', 'hey']) == [5, 2, 3]

def negate_each(nums):
    """Return a list with every number of the list nums multiplied by -1."""
    return [-x for x in nums]
assert negate_each([1, -2, 0]) == [-1, 2, 0]
assert negate_each([]) == []
assert negate_each([5]) == [-5]
assert negate_each([-1, -1]) == [1, 1]

family = "list_index"

def first_item(items):
    """Return the first item of the list items, which is not empty."""
    return items[0]
assert first_item([1, 2, 3]) == 1
assert first_item(['a']) == 'a'
assert first_item([0, 1]) == 0
assert first_item([5, 4]) == 5
assert first_item(['x', 'y']) == 'x'

def second_to_last_item(items):
    """Return the item just before the last item of the list items, which has at least two items."""
    return items[-2]
assert second_to_last_item([1, 2, 3]) == 2
assert second_to_last_item(['a', 'b']) == 'a'
assert second_to_last_item([0, 1]) == 0
assert second_to_last_item([5, 4, 3, 2]) == 3

def middle_item(items):
    """Return the middle item of the list items, which has an odd number of items."""
    return items[len(items) // 2]
assert middle_item([1, 2, 3]) == 2
assert middle_item([5]) == 5
assert middle_item([1, 2, 3, 4, 5]) == 3
assert middle_item(['a', 'b', 'c']) == 'b'
assert middle_item([3, 1, 2]) == 1

def sum_of_first_and_last(nums):
    """Return the first number of the list nums plus its last number, where nums is not empty."""
    return nums[0] + nums[-1]
assert sum_of_first_and_last([1, 2, 3]) == 4
assert sum_of_first_and_last([5]) == 10
assert sum_of_first_and_last([-1, 1]) == 0
assert sum_of_first_and_last([2, 9, 9, 4]) == 6

def second_item(items):
    """Return the second item of the list items, which has at least two items."""
    return items[1]
assert second_item([1, 2, 3]) == 2
assert second_item(['a', 'b']) == 'b'
assert second_item([0, 1]) == 1
assert second_item([5, 4, 3, 2]) == 4

family = "list_slice"

def first_n_items(items, n):
    """Return a list with the first n items of the list items, where n is not larger than its length."""
    return items[:n]
assert first_n_items([1, 2, 3], 2) == [1, 2]
assert first_n_items([1, 2, 3], 0) == []
assert first_n_items([1, 2, 3], 3) == [1, 2, 3]
assert first_n_items([], 0) == []
assert first_n_items(['a', 'b'], 1) == ['a']
assert first_n_items([3, 1, 2], 2) == [3, 1]

def last_n_items(items, n):
    """Return a list with the last n items of the list items, where n is at least 1 and not larger than its length."""
    return items[-n:]
assert last_n_items([1, 2, 3], 2) == [2, 3]
assert last_n_items([1, 2, 3], 1) == [3]
assert last_n_items([1, 2, 3], 3) == [1, 2, 3]
assert last_n_items(['a', 'b'], 1) == ['b']
assert last_n_items([3, 1, 2], 2) == [1, 2]

def without_first(items):
    """Return the list items without its first item (an empty list stays empty)."""
    return items[1:]
assert without_first([1, 2, 3]) == [2, 3]
assert without_first([1]) == []
assert without_first([]) == []
assert without_first(['a', 'b']) == ['b']
assert without_first([3, 1, 2]) == [1, 2]

def without_first_and_last(items):
    """Return the list items without its first and last items, where items has at least two items."""
    return items[1:-1]
assert without_first_and_last([1, 2, 3]) == [2]
assert without_first_and_last([1, 2]) == []
assert without_first_and_last([1, 2, 3, 4]) == [2, 3]
assert without_first_and_last(['a', 'b', 'c', 'd', 'e']) == ['b', 'c', 'd']

def every_second_item(items):
    """Return a list with every second item of the list items, starting with the first item."""
    return items[::2]
assert every_second_item([1, 2, 3, 4]) == [1, 3]
assert every_second_item([1]) == [1]
assert every_second_item([]) == []
assert every_second_item([1, 2, 3]) == [1, 3]
assert every_second_item(['a', 'b']) == ['a']

family = "list_predicates"

def contains_item(items, target):
    """Return True if target is one of the items of the list items, otherwise False."""
    return target in items
assert contains_item([1, 2, 3], 2) == True
assert contains_item([1, 2, 3], 5) == False
assert contains_item([], 1) == False
assert contains_item(['a'], 'a') == True
assert contains_item([2, 2], 2) == True

def is_empty_list(items):
    """Return True if the list items has no items, otherwise False."""
    return len(items) == 0
assert is_empty_list([]) == True
assert is_empty_list([1]) == False
assert is_empty_list([0]) == False
assert is_empty_list([[]]) == False
assert is_empty_list(['']) == False
assert is_empty_list([1, 1]) == False

def all_positive(nums):
    """Return True if every number in the list nums is greater than zero (True for an empty list), otherwise False."""
    return all(x > 0 for x in nums)
assert all_positive([1, 2]) == True
assert all_positive([1, -1]) == False
assert all_positive([0]) == False
assert all_positive([]) == True
assert all_positive([5]) == True

def has_negative(nums):
    """Return True if at least one number in the list nums is less than zero, otherwise False."""
    return any(x < 0 for x in nums)
assert has_negative([1, -1]) == True
assert has_negative([1, 2]) == False
assert has_negative([]) == False
assert has_negative([0]) == False
assert has_negative([-5]) == True

def is_longer_than(items, n):
    """Return True if the list items has more than n items, otherwise False."""
    return len(items) > n
assert is_longer_than([1, 2, 3], 2) == True
assert is_longer_than([1, 2], 2) == False
assert is_longer_than([], 0) == False
assert is_longer_than([1], 0) == True
assert is_longer_than([1, 2, 3], 5) == False
assert is_longer_than([0, 0], 1) == True

family = "list_edit"

def append_item(items, item):
    """Return a new list with the items of the list items followed by item."""
    return items + [item]
assert append_item([1, 2], 3) == [1, 2, 3]
assert append_item([], 1) == [1]
assert append_item(['a'], 'b') == ['a', 'b']
assert append_item([0], 0) == [0, 0]

def prepend_item(items, item):
    """Return a new list with item followed by the items of the list items."""
    return [item] + items
assert prepend_item([1, 2], 0) == [0, 1, 2]
assert prepend_item([], 1) == [1]
assert prepend_item(['b'], 'a') == ['a', 'b']
assert prepend_item([0], 0) == [0, 0]

def replace_first(items, item):
    """Return a new list like the list items but with item in place of its first item, where items is not empty."""
    return [item] + items[1:]
assert replace_first([1, 2, 3], 9) == [9, 2, 3]
assert replace_first([1], 5) == [5]
assert replace_first(['a', 'b'], 'c') == ['c', 'b']
assert replace_first([0, 0], 1) == [1, 0]

def add_at_both_ends(items, item):
    """Return a new list with item, then the items of the list items, then item again."""
    return [item] + items + [item]
assert add_at_both_ends([1, 2], 0) == [0, 1, 2, 0]
assert add_at_both_ends([], 5) == [5, 5]
assert add_at_both_ends(['a'], 'b') == ['b', 'a', 'b']
assert add_at_both_ends([3], 3) == [3, 3, 3]

def remove_item_at(items, index):
    """Return a new list like the list items but without the item at the given index (counting from 0)."""
    return items[:index] + items[index + 1:]
assert remove_item_at([1, 2, 3], 0) == [2, 3]
assert remove_item_at([1, 2, 3], 1) == [1, 3]
assert remove_item_at([1, 2, 3], 2) == [1, 2]
assert remove_item_at([5], 0) == []
