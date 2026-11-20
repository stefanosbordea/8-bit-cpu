"""Practice tasks of the far families about strings (format: see ../build.py)."""

family = "palindromes_anagrams"

def is_word_palindrome(text):
    """Return True if reversing the order of the words of the string text, which are separated by single spaces, gives the same words in the same order, otherwise False."""
    words = text.split()
    return words == words[::-1]
assert is_word_palindrome('one two one') == True
assert is_word_palindrome('one two') == False
assert is_word_palindrome('hi') == True
assert is_word_palindrome('ab ba') == False
assert is_word_palindrome('a b b a') == True
assert is_word_palindrome('x y z x') == False

def joined_is_palindrome(a, b):
    """Return True if the string a followed by the string b reads the same backwards as forwards, otherwise False."""
    text = a + b
    return text == text[::-1]
assert joined_is_palindrome('ab', 'ba') == True
assert joined_is_palindrome('ab', 'ab') == False
assert joined_is_palindrome('abc', 'ba') == True
assert joined_is_palindrome('', 'x') == True
assert joined_is_palindrome('a', 'b') == False
assert joined_is_palindrome('no', 'on') == True

def count_palindromes(words):
    """Return how many strings of the list words read the same backwards as forwards."""
    return sum(1 for word in words if word == word[::-1])
assert count_palindromes(['aba', 'abc', 'aa']) == 2
assert count_palindromes([]) == 0
assert count_palindromes(['ab', 'xy']) == 0
assert count_palindromes(['a', 'bb', 'level']) == 3
assert count_palindromes(['noon', 'moon']) == 1

def anagrams_in_list(word, words):
    """Return a list of the strings of the list words that use the same characters the same number of times as the string word, in the same order."""
    return [other for other in words if sorted(other) == sorted(word)]
assert anagrams_in_list('tea', ['eat', 'tee', 'ate']) == ['eat', 'ate']
assert anagrams_in_list('abc', []) == []
assert anagrams_in_list('ab', ['ab', 'ba', 'abb']) == ['ab', 'ba']
assert anagrams_in_list('a', ['b', 'aa']) == []
assert anagrams_in_list('listen', ['silent', 'enlist', 'listens']) == ['silent', 'enlist']

def are_anagrams_ignoring_spaces(a, b):
    """Return True if the strings a and b use the same characters the same number of times when their spaces are left out, otherwise False."""
    return sorted(a.replace(' ', '')) == sorted(b.replace(' ', ''))
assert are_anagrams_ignoring_spaces('the eyes', 'they see') == True
assert are_anagrams_ignoring_spaces('a b', 'ab') == True
assert are_anagrams_ignoring_spaces('abc', 'ab d') == False
assert are_anagrams_ignoring_spaces('aab', 'a b b') == False
assert are_anagrams_ignoring_spaces('dormitory', 'dirty room') == True

family = "words"

def last_word(text):
    """Return the last word of the string text, which holds one or more words separated by single spaces."""
    return text.split()[-1]
assert last_word('hello world') == 'world'
assert last_word('one') == 'one'
assert last_word('a b c') == 'c'
assert last_word('Hi there you') == 'you'

def join_with_commas(words):
    """Return one string made of the strings of the list words with a comma and a space between them (an empty string for an empty list)."""
    return ', '.join(words)
assert join_with_commas(['a', 'b', 'c']) == 'a, b, c'
assert join_with_commas([]) == ''
assert join_with_commas(['one']) == 'one'
assert join_with_commas(['x', '']) == 'x, '

def shortest_word(text):
    """Return the shortest word of the string text, where words are separated by single spaces and exactly one word is the shortest."""
    return min(text.split(), key=len)
assert shortest_word('aaa bb c') == 'c'
assert shortest_word('hello') == 'hello'
assert shortest_word('what is that') == 'is'
assert shortest_word('I am here') == 'I'

def contains_word(text, word):
    """Return True if word is one of the words of the string text, where words are separated by single spaces, otherwise False."""
    return word in text.split()
assert contains_word('the cat sat', 'cat') == True
assert contains_word('the cat sat', 'dog') == False
assert contains_word('concatenate', 'cat') == False
assert contains_word('', 'a') == False
assert contains_word('a b', 'b') == True
assert contains_word('the cat sat', 'at') == False

def capitalize_each_word(text):
    """Return the string text, whose words are in lower case and separated by single spaces, with the first letter of every word in upper case."""
    return ' '.join(word.capitalize() for word in text.split())
assert capitalize_each_word('hello world') == 'Hello World'
assert capitalize_each_word('a') == 'A'
assert capitalize_each_word('') == ''
assert capitalize_each_word('the quick fox') == 'The Quick Fox'

family = "char_transform"

def remove_digit_characters(s):
    """Return the string s without its characters that are digits from 0 to 9."""
    return ''.join(ch for ch in s if not ch.isdigit())
assert remove_digit_characters('a1b2') == 'ab'
assert remove_digit_characters('') == ''
assert remove_digit_characters('2024') == ''
assert remove_digit_characters('abc') == 'abc'
assert remove_digit_characters('1 2') == ' '

def replace_dots_with_commas(s):
    """Return the string s with every '.' replaced by ','."""
    return s.replace('.', ',')
assert replace_dots_with_commas('1.5') == '1,5'
assert replace_dots_with_commas('') == ''
assert replace_dots_with_commas('abc') == 'abc'
assert replace_dots_with_commas('...') == ',,,'
assert replace_dots_with_commas('a.b,c') == 'a,b,c'

def repeat_every_character(s, count):
    """Return a string in which every character of the string s is written count times in a row."""
    return ''.join(ch * count for ch in s)
assert repeat_every_character('abc', 2) == 'aabbcc'
assert repeat_every_character('ab', 3) == 'aaabbb'
assert repeat_every_character('', 4) == ''
assert repeat_every_character('xy', 1) == 'xy'
assert repeat_every_character('a b', 2) == 'aa  bb'

def keep_only_letters(s):
    """Return a string made of the characters of the string s that are letters, in the same order."""
    return ''.join(ch for ch in s if ch.isalpha())
assert keep_only_letters('a1b2') == 'ab'
assert keep_only_letters('') == ''
assert keep_only_letters('123') == ''
assert keep_only_letters('Hi, you!') == 'Hiyou'
assert keep_only_letters('a b') == 'ab'

def uppercase_vowels(s):
    """Return the lower-case string s with every vowel (a, e, i, o, u) changed to upper case."""
    return ''.join(ch.upper() if ch in 'aeiou' else ch for ch in s)
assert uppercase_vowels('hello') == 'hEllO'
assert uppercase_vowels('') == ''
assert uppercase_vowels('xyz') == 'xyz'
assert uppercase_vowels('aeiou') == 'AEIOU'
assert uppercase_vowels('banana') == 'bAnAnA'
