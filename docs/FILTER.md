# FILTER: the syntax filter as tables (binding for the integer reference, the benchmarks "with filter" and the CPU)

Code: `lsllm/filter_table.py`. Tests: `tests/test_filter_table_*.py`. All numbers are for the tokenizer
`data/tok/f512a` (512 tokens, variant B+) and are printed by the commands at the end.

The filter decides, before every generated token, which token ids may come next so that brackets, strings and
indentation stay well formed. It never looks at characters while it runs: every token id has two bytes that
were worked out once from its text, and the state is 7 bytes.

## 1. Two bytes per token
**Class** (25 values): what the token is for brackets, strings and line breaks. Variant B+ guarantees that a token
is either one of the 16 structural tokens or contains no structural character at all.

| class | tokens | class | tokens |
|---|---|---|---|
| 0 NEVER | unknown, begin of text | 16 CLOSE_COLON | `):` |
| 1 END | end of text | 17 SQ, 18 DQ, 19 TQ | `'`, `"`, `"""` |
| 2 NL, 3 IND | `<NL>`, `<IND>` | 20 BACKSLASH, 21 HASH | `\`, `#` |
| 4..8 DED1..DED5 | `<DED1>`..`<DED5>` | 22 COLON | `:` |
| 9 DOCLINE | `<LF>`, `<BLANK>` | 23 SPACE | tokens made of spaces only (3) |
| 10, 11, 12 OPEN | `(`, `[`, `{` | 24 PLAIN | every other token (484), also `,`, `, ` and `()` |
| 13, 14, 15 CLOSE | `)`, `]`, `}` | | |

**Word column** (62 values): what the token does to the first word of a line. It is needed because most block
keywords are not one token: only `def `, `if `, `for `, `with ` and `else` are; `while ` is `wh` `il` `e `, `elif `
is `e` `li` `f `, `try` is `tr` `y`. Tokens that act alike share a column (382 plain tokens share one).

## 2. The state (7 bytes)
| byte | values | meaning |
|---|---|---|
| lex | 0..16 | 0 code; 1, 2 just after `''` or `""`; 3..8 inside a one-line string; 9..16 inside a triple-quoted string |
| word | 0..48 | first word of the line: 0 a block keyword, complete; 1..12 exactly a keyword so far (`else`); 13 no keyword; 14 line start; 15..48 the beginning of a keyword so far (`whi`) |
| colon | 0, 1 | the last token of the line that is not spaces is `:` or `):` |
| level | 0..255 | indentation level (256 tokens cannot nest deeper than 85: every level needs a keyword, a colon and `<IND>`) |
| depth | 0..8 | number of open brackets |
| stack | 2 bytes | type of every open bracket, 2 bits per level (in Python: a list of at most 8 numbers 0, 1, 2) |

## 3. The rules
1. A closing bracket must match the innermost open bracket. At most 8 brackets are open.
2. No line break inside brackets or inside a one-line string. Inside a triple-quoted string only `<LF>`, `<BLANK>`.
3. After a line that ends with `:` the next line break is `<IND>`; after any other line `<IND>` is forbidden.
4. A line that begins with a block keyword (`def if elif else for while try except finally with class async`)
   must end with `:`, and no other line may end with `:`.
5. `<DEDk>` needs level k or more. No empty line. The text ends only where `<NL>` could come.
6. In code: no `#`, no `\`, no space at the start of a line (rules that `lsllm/filter.py` does not have, section 7).

## 4. Abstract state and the ALLOW table
Once per generated token the state is reduced to one number `a` (17 values):

    lex >= 9: a = 1          lex >= 3: a = 0                      (inside a string)
    depth > 0: a = 2 + 2 * (type of the innermost bracket) + (1 if depth = 8)
    word = 14: a = 8                                              (line start)
    (word < 13) != colon: a = 9                                   (the line may not end here)
    colon = 1: a = 10                                             (keyword line with its colon: <IND>)
    otherwise: a = 11 + min(level, 5)                             (the line may end; <DEDk> up to the level)

A token of class c may come if `ALLOW[a][c]` = 1. Columns: `-` never, `E` end, `N I 1..5` line breaks, `L` docstring
line, brackets, `;` = `):`, quotes (`T` = `"""`), `\ # :`, `_` spaces, `x` plain text.

     a  meaning                     - E N I 1 2 3 4 5 L ( [ { ) ] } ; ' " T \ # : _ x
     0  one-line string             0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1  triple-quoted string        0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     2  in ( ... ; 3 if depth = 8   0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 1 1 1 0 0 1 1 1     row 3: ( [ { are 0
     4  in [ ... ; 5 if depth = 8   0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0 1 1 1 0 0 1 1 1     row 5: ( [ { are 0
     6  in { ... ; 7 if depth = 8   0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 0 1 1 1 0 0 1 1 1     row 7: ( [ { are 0
     8  line start                  0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 1 1 1 0 0 1 0 1
     9  line may not end            0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 1 1 1 0 0 1 1 1
    10  keyword line ends with :    0 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 1 1 1 0 0 1 1 1
    11  line may end, level 0       0 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 1 1 1 0 0 1 1 1
    12..16  the same, level 1..5+   as row 11, and <DED1>..<DEDk> are 1 for k = a - 11

Every row allows at least 491 of the 512 tokens, so the model always has a choice.

## 5. The step after a token (class c, word column k)
    if c is NEVER or END: nothing
    if lex < 3 (code):
        line break (c = 2..8): level changes by 0, +1, -1..-5; word = 14; colon = 0
        otherwise: an opening bracket is pushed, a closing bracket (also `):`) pops one;
                   word = WORD[k][word]; if c is not SPACE: colon = 1 for `:` and `):`, else 0
    lex = LEX[lex][c]

`LEX` (17 x 25) is the string automaton: quotes open and close strings, a backslash escapes the next character,
three quotes in a row open and close a triple-quoted string. `WORD` (62 columns x 49 states) spells the first
word of the line along the keywords. Both tables are made by running the characters of each token through a
small automaton when the tables are built (`lex_char`, `word_token`).

## 6. Worked example
    def last_even(items):
        k = len(items) - 1
        while items[k] % 2:
            k = k - 1
        return items[k]

46 tokens. Line 1 ends with `):` (colon = 1, word = keyword), so a = 10 and the only line break allowed is `<IND>`.
Line 2 is no keyword line: a = 12 after every token outside the brackets, `<NL>` and `<DED1>` may come. Line 3:

| token | class | state after the token | a | what follows from it |
|---|---|---|---|---|
| `<NL>` | NL | word = line start, colon = 0, level = 1 | 8 | no line break, no closing bracket, no space |
| `wh` | PLAIN | word = `wh` (could still be a name) | 12 | `<NL>` allowed: `wh` alone is a valid line |
| `il` | PLAIN | word = `whil` | 12 | |
| `e ` | PLAIN | word = keyword (the space ended `while`) | 9 | from here the line must reach a `:` |
| `ite`, `ms` | PLAIN | unchanged | 9 | |
| `[` | OPEN+1 | depth = 1, stack = `[` | 4 | of the closing brackets only `]`; no line break |
| `k` | PLAIN | unchanged | 4 | |
| `]` | CLOSE+1 | depth = 0 | 9 | |
| ` `, `% `, `2` | SPACE, PLAIN | unchanged | 9 | `<NL>` still forbidden |
| `:` | COLON | colon = 1 | 10 | `<IND>` is the only line break; the text may not end |
| `<IND>` | IND | level = 2, word = line start, colon = 0 | 8 | |

Line 4 ends in a = 13 (level 2): `<NL>`, `<DED1>`, `<DED2>` and the end of the text are allowed, `<DED3>` is not.
The function takes `<DED1>`. After `return items[k]` a = 12 and the end of the text is allowed.

## 7. Sizes and measurements
| table | bytes | layout |
|---|---|---|
| token table | 1,024 | byte t = class of token t, byte 512 + t = word column of token t |
| ALLOW | 544 | 17 rows of 32 bytes: byte 32 a + c (425 are used; as packed bits it would be 54 bytes) |
| LEX | 544 | 17 rows of 32 bytes: byte 32 lex + c |
| WORD | 3,968 | 62 columns of 64 bytes: byte 64 k + word |
| total | 6,080 | 1.2 percent of the 2^19 bytes of data memory; state: 7 bytes |

Measured on 6,000 canonical functions (all 450 of `data/final/f512a/human_val.jsonl`, the first 5,550 of
`human_train.jsonl`), 932,094 positions (every token and the end of the text):
- **Accepted from start to end: 5,997 of 6,000 (99.95 percent).** The 3 rejected functions use the `match`
  statement (`match x:` ends with a colon and is not in the keyword list). `lsllm/filter.py` accepts 5,984
  (99.73 percent): the same 3, and 13 functions whose docstring is written with `'''`.
- **Same allowed set as `lsllm/filter.py`: 375,061 positions (40.24 percent)**, all of them inside strings. In
  code the table filter is stricter on purpose: `#` and `\` are forbidden (556,748 positions), a space at the
  start of a line (34,821), the end of the text right after a `:` that needs a block (11,559). Reasons: the
  character filter does not know comments, so a `#` followed by a quote or bracket would derail it; a `\` in
  code is an error unless a line break follows; a leading space is an indentation error; `if x:` with nothing
  after it does not parse. Canonical code contains none of these.
- **Apart from these added rules the sets are identical at 931,803 positions (99.969 percent).** The other 291:
  285 positions inside `'''` strings (the table filter is more permissive and right: the character filter reads
  `'''` as an empty string and a new string); 6 positions with 8 open brackets (the table filter is more
  restrictive: the depth limit; one function of 6,000 reaches depth 8, none goes beyond). No difference is unexplained.
- Not covered by this data: `""` followed by the end of the text (allowed here, refused by the character filter),
  and a quote count that continues across a docstring line break (not continued here). Neither occurs in the 6,000.
- Random text: 240 texts of up to 200 random allowed tokens after a function header, checked by a separate
  character-level checker (brackets, strings, indentation): 0 errors. Without the mask: 40 of 40 texts have errors.
- What the filter does not promise: that the text parses. It knows nothing of expressions (`x = = 1` passes).

## 8. Cost on the CPU (estimate; the measured program is in `lsllm/ls/gen.py`, numbers in `docs/COST_TABLE.md`)
Once per generated token: compute `a` (about 25 instructions), copy the 25 bytes of row `a` of ALLOW to the page
of one `LUT` table (25 x `LD`, `ST`, `DJNZ` = 75), and after the choice the step of section 5 (about 80, with two
table reads): about 180 instructions = 720 cycles. Inside the argmax loop, per token id:

    LD   R1, [P2]+     class byte of this token id
    LUT  R1, #t        R1 <- row[class]; Z = 1 if the token is not allowed
    BZ   next          skip the comparison

3 instructions = 12 cycles per token id, 512 x 12 = 6,144 cycles per generated token. Together about 6,900 cycles
per generated token, which is 0.7 to 1.2 percent of the 0.6 to 1.0 million cycles of one token (D-tok-1).
Measured on the generated program: 6,748 cycles per generated token, 200 per prompt token; it keeps the stack as one byte per level (D-ls-cpu-15). No
signed comparison is needed: the test is the Z flag of `LUT`, and the range tests of the step are the usual
`SUB`, sign table, `BZ`.

## 9. Reproduce
```
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
.venv/bin/python -m pytest -q tests/test_filter_table_rules.py tests/test_filter_table_corpus.py
.venv/bin/python tests/test_filter_table_corpus.py
```
