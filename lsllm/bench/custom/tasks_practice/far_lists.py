"""Practice tasks of the far families about lists and dictionaries (format: see ../build.py)."""

family = "dict_ops"

def count_by_length(words):
    """Return a dictionary that maps every length of a string in the list words to the number of strings with that length."""
    counts = {}
    for word in words:
        counts[len(word)] = counts.get(len(word), 0) + 1
    return counts
assert count_by_length(['a', 'bb', 'c']) == {1: 2, 2: 1}
assert count_by_length([]) == {}
assert count_by_length(['abc', 'xyz', 'abc']) == {3: 3}
assert count_by_length(['', 'no']) == {0: 1, 2: 1}

def largest_value(d):
    """Return the largest value of the dictionary d, which is not empty."""
    return max(d.values())
assert largest_value({'a': 1, 'b': 3, 'c': 2}) == 3
assert largest_value({'x': -1}) == -1
assert largest_value({1: 5, 2: 5}) == 5
assert largest_value({'a': -2, 'b': -7}) == -2

def key_of_largest_value(d):
    """Return the key of the dictionary d that has the largest value, where d is not empty and no two values are equal."""
    return max(d, key=d.get)
assert key_of_largest_value({'a': 1, 'b': 3, 'c': 2}) == 'b'
assert key_of_largest_value({'x': -1}) == 'x'
assert key_of_largest_value({1: 5, 2: 9}) == 2
assert key_of_largest_value({'z': 4, 'a': 3}) == 'z'
assert key_of_largest_value({3: 10, 7: 1}) == 3

def has_value(d, value):
    """Return True if value is one of the values of the dictionary d, otherwise False."""
    return value in d.values()
assert has_value({'a': 1, 'b': 2}, 2) == True
assert has_value({'a': 1, 'b': 2}, 3) == False
assert has_value({}, 1) == False
assert has_value({'a': 'b'}, 'a') == False
assert has_value({1: 2}, 2) == True
assert has_value({1: 2}, 1) == False

def double_all_values(d):
    """Return a dictionary with the keys of the dictionary d, in which every value is the value in d multiplied by 2."""
    return {key: value * 2 for key, value in d.items()}
assert double_all_values({'a': 1, 'b': 2}) == {'a': 2, 'b': 4}
assert double_all_values({}) == {}
assert double_all_values({'x': 0}) == {'x': 0}
assert double_all_values({1: -3, 2: 0.5}) == {1: -6, 2: 1}

family = "sorting"

def sort_by_absolute_value(nums):
    """Return a list with the numbers of the list nums sorted from the smallest absolute value to the largest, where no two absolute values are equal."""
    return sorted(nums, key=abs)
assert sort_by_absolute_value([3, -1, 2]) == [-1, 2, 3]
assert sort_by_absolute_value([]) == []
assert sort_by_absolute_value([-5]) == [-5]
assert sort_by_absolute_value([-4, 1, -2, 3]) == [1, -2, 3, -4]
assert sort_by_absolute_value([10, -20, 0]) == [0, 10, -20]

def smallest_two(nums):
    """Return a list with the two smallest numbers of the list nums, the smaller one first, where nums has at least two numbers that are all different."""
    return sorted(nums)[:2]
assert smallest_two([3, 1, 2]) == [1, 2]
assert smallest_two([5, 4]) == [4, 5]
assert smallest_two([-1, -2, -3]) == [-3, -2]
assert smallest_two([10, 30, 20, 0]) == [0, 10]

def is_sorted_descending(nums):
    """Return True if no number in the list nums is larger than the number before it (True for an empty list), otherwise False."""
    return nums == sorted(nums, reverse=True)
assert is_sorted_descending([3, 2, 1]) == True
assert is_sorted_descending([1, 3]) == False
assert is_sorted_descending([]) == True
assert is_sorted_descending([2, 2, 1]) == True
assert is_sorted_descending([3, 1, 2]) == False
assert is_sorted_descending([7]) == True

def sort_longest_first(words):
    """Return the strings of the list words sorted from longest to shortest, where no two strings have the same length."""
    return sorted(words, key=len, reverse=True)
assert sort_longest_first(['bb', 'a', 'ccc']) == ['ccc', 'bb', 'a']
assert sort_longest_first([]) == []
assert sort_longest_first(['hello']) == ['hello']
assert sort_longest_first(['xy', 'abcd', '']) == ['abcd', 'xy', '']
assert sort_longest_first(['z', 'aaa', 'mm']) == ['aaa', 'mm', 'z']

def sorted_letters_descending(word):
    """Return the letters of the lower-case string word sorted in reverse alphabetical order, as one string."""
    return ''.join(sorted(word, reverse=True))
assert sorted_letters_descending('cab') == 'cba'
assert sorted_letters_descending('') == ''
assert sorted_letters_descending('a') == 'a'
assert sorted_letters_descending('hello') == 'ollhe'
assert sorted_letters_descending('aabb') == 'bbaa'

family = "nested_lists"

def positive_numbers_in_matrix(matrix):
    """Return one list with the numbers greater than zero from all rows of matrix, a list of lists of numbers, in order."""
    return [x for row in matrix for x in row if x > 0]
assert positive_numbers_in_matrix([[1, -2], [3, 0]]) == [1, 3]
assert positive_numbers_in_matrix([]) == []
assert positive_numbers_in_matrix([[-1], [-2]]) == []
assert positive_numbers_in_matrix([[], [5, 6]]) == [5, 6]
assert positive_numbers_in_matrix([[2, -1, 4]]) == [2, 4]

def row_maximums(matrix):
    """Return a list with the largest number of every row of matrix, which is a list of lists of numbers with no empty list."""
    return [max(row) for row in matrix]
assert row_maximums([[1, 2], [4, 3]]) == [2, 4]
assert row_maximums([]) == []
assert row_maximums([[5]]) == [5]
assert row_maximums([[-1, -2], [0, 0], [7, 9]]) == [-1, 0, 9]

def diagonal_sum(matrix):
    """Return the sum of the main diagonal of the square matrix, a list of rows: item 0 of row 0 plus item 1 of row 1 and so on."""
    return sum(matrix[g][g] for g in range(len(matrix)))
assert diagonal_sum([[1, 2], [3, 4]]) == 5
assert diagonal_sum([[7]]) == 7
assert diagonal_sum([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == 6
assert diagonal_sum([[0, 5], [5, 0]]) == 0
assert diagonal_sum([[2, 9, 9], [9, 2, 9], [9, 9, 2]]) == 6

def smallest_in_matrix(matrix):
    """Return the smallest number in matrix, which is a list of lists of numbers with no empty list."""
    return min(min(row) for row in matrix)
assert smallest_in_matrix([[1, 2], [3, 4]]) == 1
assert smallest_in_matrix([[5]]) == 5
assert smallest_in_matrix([[-1, -2], [-3, -4]]) == -4
assert smallest_in_matrix([[9, 1], [1, 9]]) == 1
assert smallest_in_matrix([[3, 7], [2, 8]]) == 2

def item_in_matrix(matrix, row, column):
    """Return the item of matrix, a list of rows, in the given row and at the given column (both counting from 0)."""
    return matrix[row][column]
assert item_in_matrix([[1, 2], [3, 4]], 0, 1) == 2
assert item_in_matrix([[1, 2], [3, 4]], 1, 0) == 3
assert item_in_matrix([[5]], 0, 0) == 5
assert item_in_matrix([['a', 'b', 'c'], ['d', 'e', 'f']], 1, 2) == 'f'

family = "running"

def running_balance(start, changes):
    """Return a list with the balance after every change, where the balance begins at start and every number of the list changes is added to it in turn."""
    balances = []
    for x in changes:
        start += x
        balances.append(start)
    return balances
assert running_balance(10, [5, -3]) == [15, 12]
assert running_balance(0, []) == []
assert running_balance(100, [-100]) == [0]
assert running_balance(0, [1, 2, 3]) == [1, 3, 6]
assert running_balance(-5, [5, 5]) == [0, 5]

def adjacent_maximums(nums):
    """Return a list with, for every number of the list nums except the first, the larger of that number and the number just before it."""
    return [max(nums[g - 1], nums[g]) for g in range(1, len(nums))]
assert adjacent_maximums([1, 3, 2]) == [3, 3]
assert adjacent_maximums([5]) == []
assert adjacent_maximums([]) == []
assert adjacent_maximums([4, 4, 1, 0]) == [4, 4, 1]
assert adjacent_maximums([-1, -5, -2]) == [-1, -2]

def count_equal_neighbours(nums):
    """Return how many numbers of the list nums are equal to the number just before them."""
    count = 0
    for g in range(1, len(nums)):
        if nums[g] == nums[g - 1]:
            count += 1
    return count
assert count_equal_neighbours([1, 1, 2, 2, 2]) == 3
assert count_equal_neighbours([1, 2, 1]) == 0
assert count_equal_neighbours([]) == 0
assert count_equal_neighbours([5]) == 0
assert count_equal_neighbours([3, 3]) == 1
assert count_equal_neighbours([1, 2, 2, 1, 1]) == 2

def largest_gap(nums):
    """Return the largest distance between two neighbouring numbers of the list nums, which has at least two numbers (a distance is never negative)."""
    return max(abs(nums[g] - nums[g - 1]) for g in range(1, len(nums)))
assert largest_gap([1, 3, 6]) == 3
assert largest_gap([5, 1]) == 4
assert largest_gap([1, 1]) == 0
assert largest_gap([0, 10, 11]) == 10
assert largest_gap([9, 2, 4]) == 7

def is_strictly_increasing(nums):
    """Return True if every number of the list nums, except the first, is larger than the number just before it (True for fewer than two numbers), otherwise False."""
    return all(nums[g] > nums[g - 1] for g in range(1, len(nums)))
assert is_strictly_increasing([1, 2, 3]) == True
assert is_strictly_increasing([1, 1, 2]) == False
assert is_strictly_increasing([3, 2]) == False
assert is_strictly_increasing([]) == True
assert is_strictly_increasing([5]) == True
assert is_strictly_increasing([1, 3, 2, 4]) == False

family = "searching"

def first_number_above(nums, limit):
    """Return the first number of the list nums that is greater than limit, or None if there is none."""
    for x in nums:
        if x > limit:
            return x
    return None
assert first_number_above([1, 5, 3, 8], 2) == 5
assert first_number_above([1, 2], 5) is None
assert first_number_above([], 0) is None
assert first_number_above([4, 9], 4) == 9
assert first_number_above([-3, -1], -2) == -1

def index_of_smallest(nums):
    """Return the index (counting from 0) where the smallest number of the list nums first appears, where nums is not empty."""
    return nums.index(min(nums))
assert index_of_smallest([3, 1, 2]) == 1
assert index_of_smallest([5]) == 0
assert index_of_smallest([2, 2]) == 0
assert index_of_smallest([4, 3, 2, 1]) == 3
assert index_of_smallest([-1, -5, -3]) == 1
assert index_of_smallest([9, 0, 1, 0]) == 1

def index_of_first_vowel(s):
    """Return the index (counting from 0) of the first vowel (a, e, i, o, u) in the lower-case string s, or -1 if it has no vowel."""
    for g in range(len(s)):
        if s[g] in 'aeiou':
            return g
    return -1
assert index_of_first_vowel('hello') == 1
assert index_of_first_vowel('apple') == 0
assert index_of_first_vowel('xyz') == -1
assert index_of_first_vowel('') == -1
assert index_of_first_vowel('rhythm and') == 7

def index_of_first_negative(nums):
    """Return the index (counting from 0) of the first number of the list nums that is less than zero, or -1 if there is none."""
    for g in range(len(nums)):
        if nums[g] < 0:
            return g
    return -1
assert index_of_first_negative([1, -2, -3]) == 1
assert index_of_first_negative([1, 2]) == -1
assert index_of_first_negative([]) == -1
assert index_of_first_negative([-1]) == 0
assert index_of_first_negative([0, 5, -5, -1]) == 2

def last_even_number(nums):
    """Return the last even number of the list nums of integers, or None if there is none."""
    for x in reversed(nums):
        if x % 2 == 0:
            return x
    return None
assert last_even_number([1, 2, 4, 5]) == 4
assert last_even_number([1, 3]) is None
assert last_even_number([]) is None
assert last_even_number([6]) == 6
assert last_even_number([2, 3, 0, 7]) == 0

family = "duplicates"

def appears_more_than_once(items, target):
    """Return True if target appears more than once in the list items, otherwise False."""
    return items.count(target) > 1
assert appears_more_than_once([1, 2, 1], 1) == True
assert appears_more_than_once([1, 2, 1], 2) == False
assert appears_more_than_once([1, 2, 1], 5) == False
assert appears_more_than_once([], 1) == False
assert appears_more_than_once(['a', 'a', 'a'], 'a') == True

def count_of_most_frequent(items):
    """Return how many times the most frequent item of the list items appears in it, where items is not empty."""
    return max(items.count(x) for x in items)
assert count_of_most_frequent([1, 2, 1, 3, 1]) == 3
assert count_of_most_frequent([5]) == 1
assert count_of_most_frequent([1, 2, 3]) == 1
assert count_of_most_frequent(['a', 'b', 'a', 'b']) == 2
assert count_of_most_frequent([7, 7, 7, 7]) == 4

def has_exactly_two_values(items):
    """Return True if the list items holds exactly two different values, however often each one appears, otherwise False."""
    return len(set(items)) == 2
assert has_exactly_two_values([1, 2, 1]) == True
assert has_exactly_two_values([1, 1, 1]) == False
assert has_exactly_two_values([1, 2, 3]) == False
assert has_exactly_two_values([]) == False
assert has_exactly_two_values(['a', 'b']) == True
assert has_exactly_two_values([5]) == False

def all_appear_twice(items):
    """Return True if every item of the list items appears exactly twice in it (True for an empty list), otherwise False."""
    return all(items.count(x) == 2 for x in items)
assert all_appear_twice([1, 2, 1, 2]) == True
assert all_appear_twice([1, 1, 2]) == False
assert all_appear_twice([]) == True
assert all_appear_twice([3, 3, 3]) == False
assert all_appear_twice(['a', 'a']) == True
assert all_appear_twice([1, 2]) == False

def items_appearing_twice(items):
    """Return a list, sorted in increasing order, of the different items that appear exactly twice in the list items."""
    return sorted(x for x in set(items) if items.count(x) == 2)
assert items_appearing_twice([1, 2, 1, 3, 2]) == [1, 2]
assert items_appearing_twice([]) == []
assert items_appearing_twice([5, 5, 5]) == []
assert items_appearing_twice([4, 1, 4, 1, 1]) == [4]
assert items_appearing_twice([8, 1, 8, 1]) == [1, 8]

family = "two_lists"

def pairwise_maximums(a, b):
    """Return a list with the larger of a[k] and b[k] for every position k of the lists a and b, which have the same length."""
    return [max(x, y) for x, y in zip(a, b)]
assert pairwise_maximums([1, 5], [3, 4]) == [3, 5]
assert pairwise_maximums([], []) == []
assert pairwise_maximums([2], [2]) == [2]
assert pairwise_maximums([-1, 0, 7], [-2, 1, 7]) == [-1, 1, 7]

def total_distance(a, b):
    """Return the sum of the absolute values of a[k] - b[k] over every position k of the lists a and b, which have the same length (0 if they are empty)."""
    return sum(abs(x - y) for x, y in zip(a, b))
assert total_distance([1, 5], [3, 4]) == 3
assert total_distance([], []) == 0
assert total_distance([2], [2]) == 0
assert total_distance([0, 0, 0], [1, -1, 2]) == 4
assert total_distance([10], [-10]) == 20

def all_items_in_second(a, b):
    """Return True if every item of the list a also appears in the list b (True if a is empty), otherwise False."""
    return all(x in b for x in a)
assert all_items_in_second([1, 2], [2, 1, 3]) == True
assert all_items_in_second([1, 4], [2, 1, 3]) == False
assert all_items_in_second([], [1]) == True
assert all_items_in_second([1], []) == False
assert all_items_in_second(['a', 'a'], ['a']) == True
assert all_items_in_second([3, 2], [3]) == False

def count_where_first_is_larger(a, b):
    """Return at how many positions k the item a[k] is greater than b[k], where the lists a and b have the same length."""
    return sum(1 for x, y in zip(a, b) if x > y)
assert count_where_first_is_larger([3, 1, 5], [2, 4, 5]) == 1
assert count_where_first_is_larger([], []) == 0
assert count_where_first_is_larger([2, 2], [1, 1]) == 2
assert count_where_first_is_larger([0], [0]) == 0
assert count_where_first_is_larger([1, 9, 3], [0, 8, 4]) == 2

def pairs_list(a, b):
    """Return a list with the two-item list [a[k], b[k]] for every position k of the lists a and b, which have the same length."""
    return [[x, y] for x, y in zip(a, b)]
assert pairs_list([1, 2], [3, 4]) == [[1, 3], [2, 4]]
assert pairs_list([], []) == []
assert pairs_list(['a'], ['b']) == [['a', 'b']]
assert pairs_list([1, 2, 3], [0, 0, 0]) == [[1, 0], [2, 0], [3, 0]]

family = "rearrange"

def move_first_two_to_end(items):
    """Return the list items with its first two items moved to the end in the same order, where items has at least two items."""
    return items[2:] + items[:2]
assert move_first_two_to_end([1, 2, 3]) == [3, 1, 2]
assert move_first_two_to_end([1, 2]) == [1, 2]
assert move_first_two_to_end([1, 2, 3, 4, 5]) == [3, 4, 5, 1, 2]
assert move_first_two_to_end(['a', 'b', 'c', 'd']) == ['c', 'd', 'a', 'b']

def rotate_right_by(items, n):
    """Return the list items with its last n items moved to the front in the same order, where n is at least 1 and not larger than its length."""
    return items[-n:] + items[:-n]
assert rotate_right_by([1, 2, 3, 4], 1) == [4, 1, 2, 3]
assert rotate_right_by([1, 2, 3, 4], 3) == [2, 3, 4, 1]
assert rotate_right_by([1, 2, 3], 3) == [1, 2, 3]
assert rotate_right_by(['a', 'b', 'c'], 2) == ['b', 'c', 'a']
assert rotate_right_by([7], 1) == [7]

def swap_first_two(items):
    """Return the list items with its first two items exchanged, where items has at least two items."""
    return [items[1], items[0]] + items[2:]
assert swap_first_two([1, 2, 3]) == [2, 1, 3]
assert swap_first_two([1, 2]) == [2, 1]
assert swap_first_two([1, 2, 3, 4]) == [2, 1, 3, 4]
assert swap_first_two(['a', 'b', 'c']) == ['b', 'a', 'c']

def move_item_to_front(items, index):
    """Return the list items with the item at the given index (counting from 0) moved to the front, the other items keeping their order."""
    return [items[index]] + items[:index] + items[index + 1:]
assert move_item_to_front([1, 2, 3], 2) == [3, 1, 2]
assert move_item_to_front([1, 2, 3], 1) == [2, 1, 3]
assert move_item_to_front([1, 2, 3], 0) == [1, 2, 3]
assert move_item_to_front(['a', 'b', 'c', 'd'], 2) == ['c', 'a', 'b', 'd']
assert move_item_to_front([5], 0) == [5]

def swap_halves(items):
    """Return the list items with its first half and its second half exchanged, where items has an even number of items."""
    return items[len(items) // 2:] + items[:len(items) // 2]
assert swap_halves([1, 2, 3, 4]) == [3, 4, 1, 2]
assert swap_halves([1, 2]) == [2, 1]
assert swap_halves([]) == []
assert swap_halves(['a', 'b', 'c', 'd', 'e', 'f']) == ['d', 'e', 'f', 'a', 'b', 'c']
