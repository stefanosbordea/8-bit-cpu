"""The one interface every model exposes to the benchmark harness."""
from typing import Protocol


class Completer(Protocol):
    """Anything that can continue a piece of Python source text."""

    name: str

    def complete(self, prompt: str, n: int, max_new_tokens: int, temperature: float, seed: int) -> list[str]:
        """Return `n` continuations of `prompt` as text (without the prompt).

        temperature 0 means greedy decoding (then all n continuations are equal).
        A continuation ends at the end-of-text token, when the code returns to indentation level 0,
        or after max_new_tokens tokens.
        """
        ...
