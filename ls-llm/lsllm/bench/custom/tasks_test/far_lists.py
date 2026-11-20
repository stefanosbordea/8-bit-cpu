"""Test tasks of the far families about lists and dictionaries (format: see ../build.py)."""

family = "dict_ops"

def count_each_item(items):
    """Return a dictionary that maps every different item of the list items to the number of times it appears."""
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts
assert count_each_item(['a', 'b', 'a']) == {'a': 2, 'b': 1}
assert count_each_item([]) == {}
assert count_each_item([1, 1, 1]) == {1: 3}
assert count_each_item(['x']) == {'x': 1}

def sum_of_values(d):
    """Return the sum of the values of the dictionary d (0 if it is empty)."""
    return sum(d.values())
assert sum_of_values({'a': 1, 'b': 2}) == 3
assert sum_of_values({}) == 0
assert sum_of_values({'x': -1, 'y': 1}) == 0
assert sum_of_values({1: 10}) == 10

def keys_with_value(d, value):
    """Return the list of the keys of the dictionary d whose value equals value, in the order of the dictionary."""
    return [key for key in d if d[key] == value]
assert keys_with_value({'a': 1, 'b': 2, 'c': 1}, 1) == ['a', 'c']
assert keys_with_value({}, 1) == []
assert keys_with_value({'a': 1}, 2) == []
assert keys_with_value({'x': 0, 'y': 0}, 0) == ['x', 'y']

def value_or_zero(d, key):
    """Return the value that the dictionary d stores for key, or 0 if key is not in d."""
    return d.get(key, 0)
assert value_or_zero({'a': 1}, 'a') == 1
assert value_or_zero({'a': 1}, 'b') == 0
assert value_or_zero({}, 'a') == 0
assert value_or_zero({'a': 0, 'b': 5}, 'b') == 5

def invert_dictionary(d):
    """Return a dictionary that maps every value of the dictionary d to its key, where all values of d are different."""
    return {value: key for key, value in d.items()}
assert invert_dictionary({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}
assert invert_dictionary({}) == {}
assert invert_dictionary({'x': 'y'}) == {'y': 'x'}
assert invert_dictionary({1: 1}) == {1: 1}

family = "sorting"

def sort_descending(nums):
    """Return a list with the numbers of the list nums sorted from largest to smallest."""
    return sorted(nums, reverse=True)
assert sort_descending([1, 3, 2]) == [3, 2, 1]
assert sort_descending([]) == []
assert sort_descending([5]) == [5]
assert sort_descending([2, 2, 1]) == [2, 2, 1]
assert sort_descending([-1, 0, -2]) == [0, -1, -2]

def second_largest(nums):
    """Return the second largest number of the list nums, which holds at least two numbers that are all different."""
    return sorted(nums)[-2]
assert second_largest([1, 3, 2]) == 2
assert second_largest([5, 1]) == 1
assert second_largest([-1, -2, -3]) == -2
assert second_largest([10, 30, 20, 40]) == 30

def sort_words_by_length(words):
    """Return the strings of the list words sorted from shortest to longest, where no two strings have the same length."""
    return sorted(words, key=len)
assert sort_words_by_length(['bb', 'a', 'ccc']) == ['a', 'bb', 'ccc']
assert sort_words_by_length([]) == []
assert sort_words_by_length(['hello']) == ['hello']
assert sort_words_by_length(['xy', 'abcd', '']) == ['', 'xy', 'abcd']

def is_sorted_ascending(nums):
    """Return True if no number in the list nums is smaller than the number before it (True for an empty list), otherwise False."""
    return nums == sorted(nums)
assert is_sorted_ascending([1, 2, 3]) == True
assert is_sorted_ascending([3, 1]) == False
assert is_sorted_ascending([]) == True
assert is_sorted_ascending([1, 1, 2]) == True
assert is_sorted_ascending([2, 1, 3]) == False
assert is_sorted_ascending([7]) == True

def sorted_letters(word):
    """Return the letters of the lower-case string word sorted in alphabetical order, as one string."""
    return ''.join(sorted(word))
assert sorted_letters('cab') == 'abc'
assert sorted_letters('') == ''
assert sorted_letters('a') == 'a'
assert sorted_letters('hello') == 'ehllo'
assert sorted_letters('bbaa') == 'aabb'

family = "nested_lists"

def flatten_rows(rows):
    """Return one list with the items of all the lists inside the list rows, in order."""
    return [item for row in rows for item in row]
assert flatten_rows([[1, 2], [3]]) == [1, 2, 3]
assert flatten_rows([]) == []
assert flatten_rows([[], [1]]) == [1]
assert flatten_rows([[1], [2], [3]]) == [1, 2, 3]
assert flatten_rows([['a', 'b']]) == ['a', 'b']

def row_sums(matrix):
    """Return a list with the sum of every row of matrix, which is a list of lists of numbers."""
    return [sum(row) for row in matrix]
assert row_sums([[1, 2], [3, 4]]) == [3, 7]
assert row_sums([]) == []
assert row_sums([[5]]) == [5]
assert row_sums([[1, -1], [0, 0], [2, 2]]) == [0, 0, 4]

def main_diagonal(matrix):
    """Return the main diagonal of the square matrix, a list of rows, as a list: item 0 of row 0, item 1 of row 1 and so on."""
    return [matrix[g][g] for g in range(len(matrix))]
assert main_diagonal([[1, 2], [3, 4]]) == [1, 4]
assert main_diagonal([[5]]) == [5]
assert main_diagonal([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == [1, 2, 3]
assert main_diagonal([['a', 'b'], ['c', 'd']]) == ['a', 'd']

def largest_in_matrix(matrix):
    """Return the largest number in matrix, which is a list of lists of numbers with no empty list."""
    return max(max(row) for row in matrix)
assert largest_in_matrix([[1, 2], [3, 4]]) == 4
assert largest_in_matrix([[5]]) == 5
assert largest_in_matrix([[-1, -2], [-3, -4]]) == -1
assert largest_in_matrix([[1, 9], [9, 1]]) == 9

def column_of_matrix(matrix, index):
    """Return a list with the item at the given index (counting from 0) of every row of matrix, a list of rows."""
    return [row[index] for row in matrix]
assert column_of_matrix([[1, 2], [3, 4]], 0) == [1, 3]
assert column_of_matrix([[1, 2], [3, 4]], 1) == [2, 4]
assert column_of_matrix([], 0) == []
assert column_of_matrix([['a', 'b', 'c']], 2) == ['c']
assert column_of_matrix([[5], [6], [7]], 0) == [5, 6, 7]

family = "running"

def running_total(nums):
    """Return a list where every position holds the sum of the numbers of the list nums up to and including that position."""
    totals = []
    total = 0
    for x in nums:
        total += x
        totals.append(total)
    return totals
assert running_total([1, 2, 3]) == [1, 3, 6]
assert running_total([]) == []
assert running_total([5]) == [5]
assert running_total([1, -1, 1]) == [1, 0, 1]
assert running_total([0, 0]) == [0, 0]

def adjacent_differences(nums):
    """Return a list with every number of the list nums, except the first, minus the number just before it."""
    return [nums[g] - nums[g - 1] for g in range(1, len(nums))]
assert adjacent_differences([1, 3, 6]) == [2, 3]
assert adjacent_differences([5]) == []
assert adjacent_differences([]) == []
assert adjacent_differences([3, 3, 3]) == [0, 0]
assert adjacent_differences([10, 5, 7]) == [-5, 2]

def count_increases(nums):
    """Return how many numbers of the list nums are larger than the number just before them."""
    count = 0
    for g in range(1, len(nums)):
        if nums[g] > nums[g - 1]:
            count += 1
    return count
assert count_increases([1, 2, 3]) == 2
assert count_increases([3, 2, 1]) == 0
assert count_increases([]) == 0
assert count_increases([5]) == 0
assert count_increases([1, 1, 2]) == 1
assert count_increases([1, 3, 2, 4]) == 2

def largest_step(nums):
    """Return the largest value of a number of the list nums minus the number just before it, where nums has at least two numbers."""
    return max(nums[g] - nums[g - 1] for g in range(1, len(nums)))
assert largest_step([1, 3, 6]) == 3
assert largest_step([5, 1]) == -4
assert largest_step([1, 1]) == 0
assert largest_step([0, 10, 11]) == 10

def running_product(nums):
    """Return a list where every position holds the product of the numbers of the list nums up to and including that position."""
    products = []
    product = 1
    for x in nums:
        product *= x
        products.append(product)
    return products
assert running_product([1, 2, 3]) == [1, 2, 6]
assert running_product([]) == []
assert running_product([5]) == [5]
assert running_product([2, 0, 4]) == [2, 0, 0]
assert running_product([-1, -1, -1]) == [-1, 1, -1]

family = "searching"

def index_of_item(items, target):
    """Return the index (counting from 0) of the first item of the list items that equals target, or -1 if there is none."""
    for g in range(len(items)):
        if items[g] == target:
            return g
    return -1
assert index_of_item([1, 2, 3], 2) == 1
assert index_of_item([1, 2, 3], 5) == -1
assert index_of_item([], 1) == -1
assert index_of_item([7, 7], 7) == 0
assert index_of_item(['a', 'b'], 'b') == 1

def first_negative_number(nums):
    """Return the first number of the list nums that is less than zero, or None if there is none."""
    for x in nums:
        if x < 0:
            return x
    return None
assert first_negative_number([1, -2, -3]) == -2
assert first_negative_number([1, 2]) is None
assert first_negative_number([]) is None
assert first_negative_number([-1]) == -1
assert first_negative_number([0, -5, 3]) == -5

def index_of_largest(nums):
    """Return the index (counting from 0) where the largest number of the list nums first appears, where nums is not empty."""
    return nums.index(max(nums))
assert index_of_largest([1, 3, 2]) == 1
assert index_of_largest([5]) == 0
assert index_of_largest([2, 2]) == 0
assert index_of_largest([4, 1, 2, 3]) == 0
assert index_of_largest([-5, -1, -3]) == 1
assert index_of_largest([0, 1, 9, 9]) == 2

def position_of_character(s, ch):
    """Return the index (counting from 0) where the character ch first appears in the string s, or -1 if it does not appear."""
    return s.find(ch)
assert position_of_character('hello', 'l') == 2
assert position_of_character('hello', 'z') == -1
assert position_of_character('', 'a') == -1
assert position_of_character('abc', 'a') == 0
assert position_of_character('a b', ' ') == 1

def last_index_of_item(items, target):
    """Return the index (counting from 0) of the last item of the list items that equals target, or -1 if there is none."""
    for g in range(len(items) - 1, -1, -1):
        if items[g] == target:
            return g
    return -1
assert last_index_of_item([1, 2, 1], 1) == 2
assert last_index_of_item([1, 2, 3], 5) == -1
assert last_index_of_item([], 1) == -1
assert last_index_of_item([7], 7) == 0
assert last_index_of_item([1, 1, 1], 1) == 2
assert last_index_of_item([1, 2, 3], 2) == 1

family = "duplicates"

def has_duplicates(items):
    """Return True if some item appears more than once in the list items, otherwise False."""
    return len(set(items)) < len(items)
assert has_duplicates([1, 2, 1]) == True
assert has_duplicates([1, 2, 3]) == False
assert has_duplicates([]) == False
assert has_duplicates([5]) == False
assert has_duplicates(['a', 'a']) == True
assert has_duplicates([3, 3, 3]) == True

def count_repeats(items):
    """Return how many items of the list items are equal to an item that comes earlier in the list."""
    return len(items) - len(set(items))
assert count_repeats([1, 2, 1, 3, 2]) == 2
assert count_repeats([]) == 0
assert count_repeats([5, 5, 5]) == 2
assert count_repeats([1, 2, 3]) == 0
assert count_repeats(['a', 'b', 'a']) == 1

def count_distinct(items):
    """Return the number of different items in the list items."""
    return len(set(items))
assert count_distinct([1, 2, 1]) == 2
assert count_distinct([]) == 0
assert count_distinct([5, 5, 5]) == 1
assert count_distinct(['a', 'b', 'c']) == 3

def repeated_items(items):
    """Return a list, sorted in increasing order, of the different items that appear more than once in the list items."""
    return sorted(x for x in set(items) if items.count(x) > 1)
assert repeated_items([1, 2, 1, 3, 2]) == [1, 2]
assert repeated_items([]) == []
assert repeated_items([1, 2, 3]) == []
assert repeated_items([5, 5, 5]) == [5]
assert repeated_items([8, 1, 8, 1]) == [1, 8]

def all_the_same(items):
    """Return True if all items of the list items are equal to each other (True for an empty list), otherwise False."""
    return len(set(items)) <= 1
assert all_the_same([1, 1, 1]) == True
assert all_the_same([2, 1]) == False
assert all_the_same([]) == True
assert all_the_same([5]) == True
assert all_the_same(['a', 'a', 'b']) == False
assert all_the_same([1, 2, 1]) == False

family = "two_lists"

def pairwise_sums(a, b):
    """Return a list with a[k] + b[k] for every position k of the lists a and b, which have the same length."""
    return [x + y for x, y in zip(a, b)]
assert pairwise_sums([1, 2], [3, 4]) == [4, 6]
assert pairwise_sums([], []) == []
assert pairwise_sums([0], [5]) == [5]
assert pairwise_sums([1, -1, 2], [1, 1, 1]) == [2, 0, 3]

def dot_product(a, b):
    """Return the sum of a[k] * b[k] over every position k of the lists a and b, which have the same length (0 if they are empty)."""
    return sum(x * y for x, y in zip(a, b))
assert dot_product([1, 2], [3, 4]) == 11
assert dot_product([], []) == 0
assert dot_product([2], [5]) == 10
assert dot_product([1, 0, -1], [1, 5, 1]) == 0

def common_items(a, b):
    """Return a list of the items of the list a that also appear in the list b, in the order they have in a."""
    return [x for x in a if x in b]
assert common_items([1, 2, 3], [2, 3, 4]) == [2, 3]
assert common_items([1, 2], [3]) == []
assert common_items([], [1]) == []
assert common_items(['a', 'b'], ['b', 'a']) == ['a', 'b']
assert common_items([5, 6, 7], [7, 5]) == [5, 7]
assert common_items([3, 1, 2], [1, 3]) == [3, 1]

def count_matching_positions(a, b):
    """Return at how many positions the lists a and b, which have the same length, hold equal items."""
    return sum(1 for x, y in zip(a, b) if x == y)
assert count_matching_positions([1, 2, 3], [1, 5, 3]) == 2
assert count_matching_positions([], []) == 0
assert count_matching_positions([1], [2]) == 0
assert count_matching_positions(['a', 'b'], ['a', 'b']) == 2
assert count_matching_positions([1, 2], [2, 1]) == 0

def alternate_items(a, b):
    """Return a list that takes items in turn from the lists a and b, starting with a, where a and b have the same length."""
    result = []
    for x, y in zip(a, b):
        result.append(x)
        result.append(y)
    return result
assert alternate_items([1, 2], [3, 4]) == [1, 3, 2, 4]
assert alternate_items([], []) == []
assert alternate_items(['a'], ['b']) == ['a', 'b']
assert alternate_items([1, 2, 3], [0, 0, 0]) == [1, 0, 2, 0, 3, 0]

family = "rearrange"

def move_first_to_end(items):
    """Return the list items with its first item moved to the end (an empty list stays empty)."""
    return items[1:] + items[:1]
assert move_first_to_end([1, 2, 3]) == [2, 3, 1]
assert move_first_to_end([]) == []
assert move_first_to_end([5]) == [5]
assert move_first_to_end(['a', 'b']) == ['b', 'a']
assert move_first_to_end([1, 2, 3, 4]) == [2, 3, 4, 1]

def move_last_to_front(items):
    """Return the list items with its last item moved to the front (an empty list stays empty)."""
    return items[-1:] + items[:-1]
assert move_last_to_front([1, 2, 3]) == [3, 1, 2]
assert move_last_to_front([]) == []
assert move_last_to_front([5]) == [5]
assert move_last_to_front(['a', 'b']) == ['b', 'a']
assert move_last_to_front([1, 2, 3, 4]) == [4, 1, 2, 3]

def swap_first_and_last(items):
    """Return the list items with its first and last items exchanged, where items has at least two items."""
    return [items[-1]] + items[1:-1] + [items[0]]
assert swap_first_and_last([1, 2, 3]) == [3, 2, 1]
assert swap_first_and_last([1, 2]) == [2, 1]
assert swap_first_and_last([1, 2, 3, 4]) == [4, 2, 3, 1]
assert swap_first_and_last(['a', 'b', 'c']) == ['c', 'b', 'a']

def rotate_left_by(items, n):
    """Return the list items with its first n items moved to the end in the same order, where n is not larger than its length."""
    return items[n:] + items[:n]
assert rotate_left_by([1, 2, 3, 4], 2) == [3, 4, 1, 2]
assert rotate_left_by([1, 2, 3], 0) == [1, 2, 3]
assert rotate_left_by([1, 2, 3], 3) == [1, 2, 3]
assert rotate_left_by([], 0) == []
assert rotate_left_by(['a', 'b', 'c'], 1) == ['b', 'c', 'a']

def swap_items_at(items, a, b):
    """Return a copy of the list items in which the items at the indexes a and b (counting from 0) are exchanged."""
    result = items[:]
    result[a], result[b] = result[b], result[a]
    return result
assert swap_items_at([1, 2, 3], 0, 2) == [3, 2, 1]
assert swap_items_at([1, 2, 3], 1, 1) == [1, 2, 3]
assert swap_items_at(['a', 'b'], 0, 1) == ['b', 'a']
assert swap_items_at([1, 2, 3, 4], 1, 2) == [1, 3, 2, 4]
