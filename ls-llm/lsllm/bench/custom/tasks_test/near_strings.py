"""Test tasks of the near families about strings and reversing (format: see ../build.py)."""

family = "string_case_length"

def capitalize_first_letter(s):
    """Return the string s with its first letter in upper case and all its other letters in lower case."""
    return s.capitalize()
assert capitalize_first_letter('hello') == 'Hello'
assert capitalize_first_letter('HELLO') == 'Hello'
assert capitalize_first_letter('h') == 'H'
assert capitalize_first_letter('') == ''
assert capitalize_first_letter('hello world') == 'Hello world'

def length_difference(a, b):
    """Return the length of the string a minus the length of the string b."""
    return len(a) - len(b)
assert length_difference('abc', 'a') == 2
assert length_difference('a', 'abc') == -2
assert length_difference('', '') == 0
assert length_difference('hi', 'yo') == 0
assert length_difference('hello', '') == 5

def same_ignoring_case(a, b):
    """Return True if the strings a and b are equal when upper and lower case are ignored, otherwise False."""
    return a.lower() == b.lower()
assert same_ignoring_case('Hello', 'hello') == True
assert same_ignoring_case('abc', 'abd') == False
assert same_ignoring_case('abc', 'ABC') == True
assert same_ignoring_case('ab', 'a') == False
assert same_ignoring_case('a', 'ab') == False
assert same_ignoring_case('abc', 'cba') == False

def shout_word(word):
    """Return the string word in upper case, followed by an exclamation mark."""
    return word.upper() + '!'
assert shout_word('hi') == 'HI!'
assert shout_word('') == '!'
assert shout_word('Hey') == 'HEY!'
assert shout_word('OK') == 'OK!'
assert shout_word('a b') == 'A B!'

def has_length(s, n):
    """Return True if the string s has exactly n characters, otherwise False."""
    return len(s) == n
assert has_length('abc', 3) == True
assert has_length('abc', 2) == False
assert has_length('', 0) == True
assert has_length('a b', 3) == True
assert has_length('hello', 4) == False

family = "string_predicates"

def starts_with(s, prefix):
    """Return True if the string s begins with the string prefix, otherwise False."""
    return s.startswith(prefix)
assert starts_with('hello', 'he') == True
assert starts_with('hello', 'lo') == False
assert starts_with('', 'a') == False
assert starts_with('a', 'a') == True
assert starts_with('Hello', 'h') == False
assert starts_with('hello', 'el') == False

def contains_space(s):
    """Return True if the string s has at least one space character, otherwise False."""
    return ' ' in s
assert contains_space('a b') == True
assert contains_space('ab') == False
assert contains_space('') == False
assert contains_space(' ') == True
assert contains_space('hello world ') == True

def is_all_digits(s):
    """Return True if the string s is not empty and all its characters are digits from 0 to 9, otherwise False."""
    return s.isdigit()
assert is_all_digits('123') == True
assert is_all_digits('12a') == False
assert is_all_digits('') == False
assert is_all_digits('0') == True
assert is_all_digits('1 2') == False
assert is_all_digits('-1') == False

def is_empty_string(s):
    """Return True if the string s has no characters, otherwise False."""
    return s == ''
assert is_empty_string('') == True
assert is_empty_string('a') == False
assert is_empty_string(' ') == False
assert is_empty_string('abc') == False
assert is_empty_string('0') == False
assert is_empty_string('aa') == False

def starts_with_uppercase(s):
    """Return True if the string s, which is not empty, begins with an upper-case letter, otherwise False."""
    return s[0].isupper()
assert starts_with_uppercase('Hello') == True
assert starts_with_uppercase('hello') == False
assert starts_with_uppercase('A') == True
assert starts_with_uppercase('1abc') == False
assert starts_with_uppercase('aB') == False

family = "string_count"

def count_letter(s, letter):
    """Return how many times the character letter appears in the string s."""
    return s.count(letter)
assert count_letter('banana', 'a') == 3
assert count_letter('banana', 'z') == 0
assert count_letter('', 'a') == 0
assert count_letter('aaa', 'a') == 3
assert count_letter('Aa', 'a') == 1

def number_of_vowels(s):
    """Return how many characters of the lower-case string s are vowels (a, e, i, o, u)."""
    return sum(1 for ch in s if ch in 'aeiou')
assert number_of_vowels('hello') == 2
assert number_of_vowels('') == 0
assert number_of_vowels('xyz') == 0
assert number_of_vowels('aeiou') == 5
assert number_of_vowels('banana') == 3

def count_spaces(s):
    """Return the number of space characters in the string s."""
    return s.count(' ')
assert count_spaces('a b c') == 2
assert count_spaces('') == 0
assert count_spaces('abc') == 0
assert count_spaces('   ') == 3
assert count_spaces(' a') == 1

def count_digit_characters(s):
    """Return how many characters of the string s are digits from 0 to 9."""
    return sum(1 for ch in s if ch.isdigit())
assert count_digit_characters('a1b2') == 2
assert count_digit_characters('') == 0
assert count_digit_characters('abc') == 0
assert count_digit_characters('2024') == 4
assert count_digit_characters('1 2') == 2

def count_uppercase_letters(s):
    """Return how many characters of the string s are upper-case letters."""
    return sum(1 for ch in s if ch.isupper())
assert count_uppercase_letters('Hello World') == 2
assert count_uppercase_letters('') == 0
assert count_uppercase_letters('abc') == 0
assert count_uppercase_letters('ABC') == 3
assert count_uppercase_letters('aB1') == 1

family = "string_index_slice"

def first_two_characters(s):
    """Return the first two characters of the string s, which has at least two characters."""
    return s[:2]
assert first_two_characters('hello') == 'he'
assert first_two_characters('ab') == 'ab'
assert first_two_characters('abc') == 'ab'
assert first_two_characters('12345') == '12'

def last_three_characters(s):
    """Return the last three characters of the string s, which has at least three characters."""
    return s[-3:]
assert last_three_characters('hello') == 'llo'
assert last_three_characters('abc') == 'abc'
assert last_three_characters('abcd') == 'bcd'
assert last_three_characters('x y z') == 'y z'

def character_at(s, index):
    """Return the character of the string s at the given index (counting from 0)."""
    return s[index]
assert character_at('hello', 0) == 'h'
assert character_at('hello', 4) == 'o'
assert character_at('ab', 1) == 'b'
assert character_at('a', 0) == 'a'
assert character_at('a b', 1) == ' '

def without_last_character(s):
    """Return the string s without its last character (an empty string stays empty)."""
    return s[:-1]
assert without_last_character('hello') == 'hell'
assert without_last_character('a') == ''
assert without_last_character('') == ''
assert without_last_character('ab') == 'a'

def substring_between(s, start, end):
    """Return the part of the string s from index start up to, but not including, index end (counting from 0)."""
    return s[start:end]
assert substring_between('hello', 1, 3) == 'el'
assert substring_between('hello', 0, 5) == 'hello'
assert substring_between('hello', 2, 2) == ''
assert substring_between('abc', 0, 1) == 'a'

family = "string_build"

def join_with_space(a, b):
    """Return the string a, then one space, then the string b."""
    return a + ' ' + b
assert join_with_space('hello', 'world') == 'hello world'
assert join_with_space('a', 'b') == 'a b'
assert join_with_space('', '') == ' '
assert join_with_space('x', '') == 'x '

def make_greeting(name):
    """Return the text 'Hello, ' followed by the string name and an exclamation mark."""
    return 'Hello, ' + name + '!'
assert make_greeting('Ann') == 'Hello, Ann!'
assert make_greeting('') == 'Hello, !'
assert make_greeting('Bob') == 'Hello, Bob!'
assert make_greeting('a b') == 'Hello, a b!'

def wrap_in_brackets(s):
    """Return the string s with '[' in front of it and ']' after it."""
    return '[' + s + ']'
assert wrap_in_brackets('a') == '[a]'
assert wrap_in_brackets('') == '[]'
assert wrap_in_brackets('hello') == '[hello]'
assert wrap_in_brackets('[x]') == '[[x]]'

def repeat_three_times(s):
    """Return the string s written three times in a row."""
    return s * 3
assert repeat_three_times('ab') == 'ababab'
assert repeat_three_times('') == ''
assert repeat_three_times('x') == 'xxx'
assert repeat_three_times('a ') == 'a a a '

def surround_with(s, mark):
    """Return the string mark, then the string s, then the string mark again."""
    return mark + s + mark
assert surround_with('a', '*') == '*a*'
assert surround_with('', '-') == '--'
assert surround_with('hi', '') == 'hi'
assert surround_with('x', 'ab') == 'abxab'

family = "reverse"

def reverse_list(items):
    """Return a list with the items of the list items in reverse order."""
    return list(reversed(items))
assert reverse_list([1, 3, 2]) == [2, 3, 1]
assert reverse_list([]) == []
assert reverse_list([5]) == [5]
assert reverse_list(['a', 'b']) == ['b', 'a']
assert reverse_list([1, 1, 2]) == [2, 1, 1]

def reverse_string(s):
    """Return the characters of the string s in reverse order."""
    return s[::-1]
assert reverse_string('abc') == 'cba'
assert reverse_string('') == ''
assert reverse_string('a') == 'a'
assert reverse_string('ab cd') == 'dc ba'
assert reverse_string('aab') == 'baa'
assert reverse_string('Ab') == 'bA'

def mirror_string(s):
    """Return the string s followed by its characters in reverse order."""
    return s + s[::-1]
assert mirror_string('ab') == 'abba'
assert mirror_string('') == ''
assert mirror_string('a') == 'aa'
assert mirror_string('abc') == 'abccba'

def reverse_uppercase(s):
    """Return the characters of the string s in reverse order and in upper case."""
    return s[::-1].upper()
assert reverse_uppercase('abc') == 'CBA'
assert reverse_uppercase('') == ''
assert reverse_uppercase('a') == 'A'
assert reverse_uppercase('Hi you') == 'UOY IH'

def reverse_three(a, b, c):
    """Return a list with c, b and a in that order."""
    return [c, b, a]
assert reverse_three(1, 2, 3) == [3, 2, 1]
assert reverse_three('a', 'b', 'c') == ['c', 'b', 'a']
assert reverse_three(0, 0, 0) == [0, 0, 0]
assert reverse_three(1, 1, 2) == [2, 1, 1]
