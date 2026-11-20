"""Two decoding aids that need only small tables and counters on the CPU (measured on the practice sets only).

prompt_tokens: a flag per token id, set for the name-like tokens of the prompt (signature and docstring). Adding a
constant to the logits of flagged tokens makes the model prefer the names and numbers it was given.
On the CPU: one flag byte per token id, set while the prompt is fed; one addition per logit in the argmax loop.

repeated: the tokens that would write the same run of n tokens once more than allowed. Forbidding them stops loops
such as `* 2 * 2 * 2 ...`. On the CPU: a comparison of the last n - 1 tokens against every earlier place of the answer.
"""
import keyword

KEYWORDS = set(keyword.kwlist)


def name_like(piece):
    """True for a token that is (part of) a name or a number: it has a letter or a digit, and is no marker or keyword."""
    text = piece.strip()
    return bool(text) and not (text.startswith("<") and text.endswith(">")) and text not in KEYWORDS \
        and any(ch.isalnum() for ch in text)


def prompt_tokens(tok, ids):
    """For every token id: 1 if it is name-like and occurs in the prompt `ids`, else 0."""
    present = set(ids)
    return [int(t in present and name_like(tok.pieces[t])) for t in range(tok.vocab_size)]


def repeated(answer, n, most):
    """Token ids that may not come next: each would complete a run of n tokens that the answer already has `most` times."""
    if n < 1 or len(answer) < n - 1:
        return set()
    tail = tuple(answer[len(answer) - n + 1:]) if n > 1 else ()
    count = {}
    for g in range(len(answer) - n + 1):
        if tuple(answer[g:g + n - 1]) == tail:
            count[answer[g + n - 1]] = count.get(answer[g + n - 1], 0) + 1
    return {t for t, c in count.items() if c >= most}
