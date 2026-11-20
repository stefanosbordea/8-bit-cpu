"""Practice tasks of the near families about strings and reversing (format: see ../build.py)."""

family = "string_case_length"

def combined_length(a, b):
    """Return the length of the string a plus the length of the string b."""
    return len(a) + len(b)
assert combined_length('abc', 'de') == 5
assert combined_length('', '') == 0
assert combined_length('a', '') == 1
assert combined_length('hi', 'you') == 5
assert combined_length('a b', 'c') == 4

def is_longer_string(a, b):
    """Return True if the string a has more characters than the string b, otherwise False."""
    return len(a) > len(b)
assert is_longer_string('abc', 'ab') == True
assert is_longer_string('aaa', 'zz') == True
assert is_longer_string('zz', 'aaa') == False
assert is_longer_string('ab', 'cd') == False
assert is_longer_string('a', '') == True
assert is_longer_string('', '') == False

def whisper_word(word):
    """Return the string word in lower case, followed by three dots."""
    return word.lower() + '...'
assert whisper_word('HI') == 'hi...'
assert whisper_word('') == '...'
assert whisper_word('Hey') == 'hey...'
assert whisper_word('ok') == 'ok...'
assert whisper_word('A B') == 'a b...'

def has_no_uppercase(s):
    """Return True if the string s has no upper-case letters (True for an empty string), otherwise False."""
    return s == s.lower()
assert has_no_uppercase('hello') == True
assert has_no_uppercase('Hello') == False
assert has_no_uppercase('') == True
assert has_no_uppercase('abc 12') == True
assert has_no_uppercase('abC') == False
assert has_no_uppercase('ABC') == False

def upper_then_lower(a, b):
    """Return the string a in upper case followed by the string b in lower case."""
    return a.upper() + b.lower()
assert upper_then_lower('ab', 'CD') == 'ABcd'
assert upper_then_lower('', '') == ''
assert upper_then_lower('Hi', 'You') == 'HIyou'
assert upper_then_lower('x', '') == 'X'
assert upper_then_lower('', 'Y') == 'y'

family = "string_predicates"

def ends_with(s, suffix):
    """Return True if the string s ends with the string suffix, otherwise False."""
    return s.endswith(suffix)
assert ends_with('hello', 'lo') == True
assert ends_with('hello', 'he') == False
assert ends_with('', 'a') == False
assert ends_with('a', 'a') == True
assert ends_with('Hello', 'O') == False

def contains_comma(s):
    """Return True if the string s has at least one comma, otherwise False."""
    return ',' in s
assert contains_comma('a,b') == True
assert contains_comma('ab') == False
assert contains_comma('') == False
assert contains_comma(',') == True
assert contains_comma('a. b') == False
assert contains_comma('one, two, ') == True

def is_all_letters(s):
    """Return True if the string s is not empty and all its characters are letters, otherwise False."""
    return s.isalpha()
assert is_all_letters('abc') == True
assert is_all_letters('ab1') == False
assert is_all_letters('') == False
assert is_all_letters('Z') == True
assert is_all_letters('a b') == False
assert is_all_letters('abC') == True

def is_blank(s):
    """Return True if the string s is empty or made only of space characters, otherwise False."""
    return s.strip() == ''
assert is_blank('') == True
assert is_blank(' ') == True
assert is_blank('   ') == True
assert is_blank('a') == False
assert is_blank(' a ') == False
assert is_blank('0') == False

def starts_with_digit(s):
    """Return True if the string s, which is not empty, begins with a digit from 0 to 9, otherwise False."""
    return s[0].isdigit()
assert starts_with_digit('1abc') == True
assert starts_with_digit('abc1') == False
assert starts_with_digit('7') == True
assert starts_with_digit('x') == False
assert starts_with_digit(' 5') == False
assert starts_with_digit('09') == True

family = "string_count"

def count_letter_ignoring_case(s, letter):
    """Return how many times the character letter appears in the string s when upper and lower case are ignored."""
    return s.lower().count(letter.lower())
assert count_letter_ignoring_case('Banana', 'a') == 3
assert count_letter_ignoring_case('Abba', 'A') == 2
assert count_letter_ignoring_case('abc', 'z') == 0
assert count_letter_ignoring_case('', 'a') == 0
assert count_letter_ignoring_case('BBb', 'b') == 3

def count_plus_and_minus_signs(s):
    """Return how many characters of the string s are plus signs or minus signs."""
    return sum(1 for ch in s if ch in '+-')
assert count_plus_and_minus_signs('1+2-3') == 2
assert count_plus_and_minus_signs('') == 0
assert count_plus_and_minus_signs('abc') == 0
assert count_plus_and_minus_signs('--') == 2
assert count_plus_and_minus_signs('+a+b-') == 3

def count_commas(s):
    """Return the number of commas in the string s."""
    return s.count(',')
assert count_commas('a,b,c') == 2
assert count_commas('') == 0
assert count_commas('abc') == 0
assert count_commas(',,,') == 3
assert count_commas('a, b') == 1

def count_lowercase_letters(s):
    """Return how many characters of the string s are lower-case letters."""
    return sum(1 for ch in s if ch.islower())
assert count_lowercase_letters('Hello World') == 8
assert count_lowercase_letters('') == 0
assert count_lowercase_letters('ABC') == 0
assert count_lowercase_letters('abc') == 3
assert count_lowercase_letters('aB1 c') == 2

def count_letter_characters(s):
    """Return how many characters of the string s are letters."""
    return sum(1 for ch in s if ch.isalpha())
assert count_letter_characters('a1b2') == 2
assert count_letter_characters('') == 0
assert count_letter_characters('123') == 0
assert count_letter_characters('Hello World') == 10
assert count_letter_characters('aB c!') == 3

family = "string_index_slice"

def first_three_characters(s):
    """Return the first three characters of the string s, which has at least three characters."""
    return s[:3]
assert first_three_characters('hello') == 'hel'
assert first_three_characters('abc') == 'abc'
assert first_three_characters('abcd') == 'abc'
assert first_three_characters('1 2 3') == '1 2'

def last_four_characters(s):
    """Return the last four characters of the string s, which has at least four characters."""
    return s[-4:]
assert last_four_characters('hello') == 'ello'
assert last_four_characters('abcd') == 'abcd'
assert last_four_characters('abcdef') == 'cdef'
assert last_four_characters('a b c') == ' b c'

def without_first_two_characters(s):
    """Return the string s without its first two characters, where s has at least two characters."""
    return s[2:]
assert without_first_two_characters('hello') == 'llo'
assert without_first_two_characters('ab') == ''
assert without_first_two_characters('abc') == 'c'
assert without_first_two_characters('a bcd') == 'bcd'

def two_characters_at(s, index):
    """Return the character of the string s at the given index (counting from 0) followed by the character after it."""
    return s[index:index + 2]
assert two_characters_at('hello', 0) == 'he'
assert two_characters_at('hello', 3) == 'lo'
assert two_characters_at('hello', 1) == 'el'
assert two_characters_at('ab', 0) == 'ab'
assert two_characters_at('a b c', 1) == ' b'

def second_half_of_string(s):
    """Return the second half of the string s, which has an even number of characters."""
    return s[len(s) // 2:]
assert second_half_of_string('abcd') == 'cd'
assert second_half_of_string('ab') == 'b'
assert second_half_of_string('') == ''
assert second_half_of_string('hello!') == 'lo!'

family = "string_build"

def make_label(name, value):
    """Return the string name, then a colon and a space, then the string value."""
    return name + ': ' + value
assert make_label('age', '30') == 'age: 30'
assert make_label('a', 'b') == 'a: b'
assert make_label('', '') == ': '
assert make_label('x', '') == 'x: '

def letter_opening(name):
    """Return the text 'Dear ' followed by the string name and a comma."""
    return 'Dear ' + name + ','
assert letter_opening('Ann') == 'Dear Ann,'
assert letter_opening('') == 'Dear ,'
assert letter_opening('Bob') == 'Dear Bob,'
assert letter_opening('a b') == 'Dear a b,'

def wrap_in_parentheses(s):
    """Return the string s with '(' in front of it and ')' after it."""
    return '(' + s + ')'
assert wrap_in_parentheses('a') == '(a)'
assert wrap_in_parentheses('') == '()'
assert wrap_in_parentheses('hello') == '(hello)'
assert wrap_in_parentheses('(x)') == '((x))'

def dash_line(length):
    """Return a string made of length '-' characters."""
    return '-' * length
assert dash_line(3) == '---'
assert dash_line(0) == ''
assert dash_line(1) == '-'
assert dash_line(10) == '----------'

def echo_word(word):
    """Return the string word, then one space, then the string word again."""
    return word + ' ' + word
assert echo_word('hi') == 'hi hi'
assert echo_word('') == ' '
assert echo_word('x') == 'x x'
assert echo_word('a b') == 'a b a b'

family = "reverse"

def reverse_four(a, b, c, d):
    """Return a list with d, c, b and a in that order."""
    return [d, c, b, a]
assert reverse_four(1, 2, 3, 4) == [4, 3, 2, 1]
assert reverse_four('a', 'b', 'c', 'd') == ['d', 'c', 'b', 'a']
assert reverse_four(0, 0, 0, 0) == [0, 0, 0, 0]
assert reverse_four(1, 1, 2, 3) == [3, 2, 1, 1]

def reverse_lowercase(s):
    """Return the characters of the string s in reverse order and in lower case."""
    return s[::-1].lower()
assert reverse_lowercase('ABC') == 'cba'
assert reverse_lowercase('') == ''
assert reverse_lowercase('a') == 'a'
assert reverse_lowercase('Hi You') == 'uoy ih'

def reverse_then_original(s):
    """Return the characters of the string s in reverse order, followed by the string s itself."""
    return s[::-1] + s
assert reverse_then_original('ab') == 'baab'
assert reverse_then_original('') == ''
assert reverse_then_original('a') == 'aa'
assert reverse_then_original('abc') == 'cbaabc'

def reverse_joined(a, b):
    """Return the string a followed by the string b, with all the characters of the result in reverse order."""
    return (a + b)[::-1]
assert reverse_joined('ab', 'cd') == 'dcba'
assert reverse_joined('', '') == ''
assert reverse_joined('a', '') == 'a'
assert reverse_joined('abc', 'x') == 'xcba'
assert reverse_joined('x', 'yz') == 'zyx'

def reverse_and_append(items, item):
    """Return a list with the items of the list items in reverse order, followed by item."""
    return items[::-1] + [item]
assert reverse_and_append([1, 2, 3], 4) == [3, 2, 1, 4]
assert reverse_and_append([], 1) == [1]
assert reverse_and_append(['a', 'b'], 'c') == ['b', 'a', 'c']
assert reverse_and_append([5], 5) == [5, 5]
assert reverse_and_append([1, 1, 2], 0) == [2, 1, 1, 0]
