"""Test tasks of the far families about strings (format: see ../build.py)."""

family = "palindromes_anagrams"

def is_palindrome_ignoring_case(text):
    """Return True if the string text reads the same backwards as forwards when upper and lower case are ignored, otherwise False."""
    text = text.lower()
    return text == text[::-1]
assert is_palindrome_ignoring_case('Level') == True
assert is_palindrome_ignoring_case('abc') == False
assert is_palindrome_ignoring_case('a') == True
assert is_palindrome_ignoring_case('Abba') == True
assert is_palindrome_ignoring_case('ab') == False
assert is_palindrome_ignoring_case('Noon') == True

def are_anagrams_ignoring_case(a, b):
    """Return True if the strings a and b use the same letters the same number of times when upper and lower case are ignored, otherwise False."""
    return sorted(a.lower()) == sorted(b.lower())
assert are_anagrams_ignoring_case('Listen', 'Silent') == True
assert are_anagrams_ignoring_case('abc', 'abd') == False
assert are_anagrams_ignoring_case('Aab', 'abb') == False
assert are_anagrams_ignoring_case('Tea', 'EAT') == True
assert are_anagrams_ignoring_case('a', 'aa') == False
assert are_anagrams_ignoring_case('ab', 'Abc') == False

def are_anagrams(a, b):
    """Return True if the strings a and b use the same characters the same number of times, otherwise False."""
    return sorted(a) == sorted(b)
assert are_anagrams('listen', 'silent') == True
assert are_anagrams('abc', 'abd') == False
assert are_anagrams('aab', 'abb') == False
assert are_anagrams('abc', 'cab') == True
assert are_anagrams('ab', 'a b') == False
assert are_anagrams('Ab', 'ba') == False

def is_palindrome_list(items):
    """Return True if the list items is equal to itself in reverse order (True for an empty list), otherwise False."""
    return items == list(reversed(items))
assert is_palindrome_list([1, 2, 1]) == True
assert is_palindrome_list([1, 2]) == False
assert is_palindrome_list([]) == True
assert is_palindrome_list([5]) == True
assert is_palindrome_list([1, 2, 2, 1]) == True
assert is_palindrome_list([1, 2, 3, 1]) == False

def is_palindrome_ignoring_spaces(text):
    """Return True if the string text without its spaces reads the same backwards as forwards, otherwise False."""
    letters = text.replace(' ', '')
    return letters == letters[::-1]
assert is_palindrome_ignoring_spaces('taco cat') == True
assert is_palindrome_ignoring_spaces('nurses run') == True
assert is_palindrome_ignoring_spaces('ab c') == False
assert is_palindrome_ignoring_spaces('a') == True
assert is_palindrome_ignoring_spaces('ab, a') == False
assert is_palindrome_ignoring_spaces('a bb') == False

family = "words"

def count_words(text):
    """Return the number of words in the string text, where words are separated by single spaces (0 for an empty string)."""
    return len(text.split())
assert count_words('a b c') == 3
assert count_words('') == 0
assert count_words('hello') == 1
assert count_words('one two') == 2
assert count_words('I am here now') == 4

def first_word(text):
    """Return the first word of the string text, which holds one or more words separated by single spaces."""
    return text.split()[0]
assert first_word('hello world') == 'hello'
assert first_word('one') == 'one'
assert first_word('a b c') == 'a'
assert first_word('Hi there you') == 'Hi'
assert first_word('zebra and ox') == 'zebra'

def join_with_dashes(words):
    """Return one string made of the strings of the list words with '-' between them (an empty string for an empty list)."""
    return '-'.join(words)
assert join_with_dashes(['a', 'b', 'c']) == 'a-b-c'
assert join_with_dashes([]) == ''
assert join_with_dashes(['one']) == 'one'
assert join_with_dashes(['x', '']) == 'x-'

def longest_word(text):
    """Return the longest word of the string text, where words are separated by single spaces and exactly one word is the longest."""
    return max(text.split(), key=len)
assert longest_word('a bb ccc') == 'ccc'
assert longest_word('hello') == 'hello'
assert longest_word('the quick fox') == 'quick'
assert longest_word('I am here') == 'here'

def first_letters_of_words(text):
    """Return a string with the first letter of every word of text, where words are separated by single spaces (an empty string if text is empty)."""
    return ''.join(word[0] for word in text.split())
assert first_letters_of_words('hello world') == 'hw'
assert first_letters_of_words('a') == 'a'
assert first_letters_of_words('') == ''
assert first_letters_of_words('Portable Network Graphics') == 'PNG'

family = "char_transform"

def remove_character(s, ch):
    """Return the string s without the characters that are equal to the character ch."""
    return s.replace(ch, '')
assert remove_character('banana', 'a') == 'bnn'
assert remove_character('abc', 'z') == 'abc'
assert remove_character('', 'a') == ''
assert remove_character('aaa', 'a') == ''
assert remove_character('a b', ' ') == 'ab'

def replace_spaces_with_dashes(s):
    """Return the string s with every space character replaced by '-'."""
    return s.replace(' ', '-')
assert replace_spaces_with_dashes('a b c') == 'a-b-c'
assert replace_spaces_with_dashes('') == ''
assert replace_spaces_with_dashes('abc') == 'abc'
assert replace_spaces_with_dashes('  ') == '--'
assert replace_spaces_with_dashes(' x') == '-x'

def double_every_character(s):
    """Return a string in which every character of the string s is written twice in a row."""
    return ''.join(ch * 2 for ch in s)
assert double_every_character('abc') == 'aabbcc'
assert double_every_character('') == ''
assert double_every_character('a') == 'aa'
assert double_every_character('a b') == 'aa  bb'
assert double_every_character('xx') == 'xxxx'

def keep_only_digits(s):
    """Return a string made of the characters of the string s that are digits from 0 to 9, in the same order."""
    return ''.join(ch for ch in s if ch.isdigit())
assert keep_only_digits('a1b2') == '12'
assert keep_only_digits('') == ''
assert keep_only_digits('abc') == ''
assert keep_only_digits('2024') == '2024'
assert keep_only_digits('1 2') == '12'

def replace_vowels_with_star(s):
    """Return the lower-case string s with every vowel (a, e, i, o, u) replaced by '*'."""
    return ''.join('*' if ch in 'aeiou' else ch for ch in s)
assert replace_vowels_with_star('hello') == 'h*ll*'
assert replace_vowels_with_star('') == ''
assert replace_vowels_with_star('xyz') == 'xyz'
assert replace_vowels_with_star('aeiou') == '*****'
assert replace_vowels_with_star('banana') == 'b*n*n*'
