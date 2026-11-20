Copied from `RESEARCH.md`, section 10, where every row has its link and verification tag. None of these models was
run by us, and none is in our size class: the smallest published HumanEval score found belongs to a model about 35
times larger than Tier A. HumanEval is pass@1 / pass@10 / pass@100 in percent.

| model | parameters | HumanEval | MBPP | source |
|---|---|---|---|---|
| Codex-12M | 12 M (non-embedding count) | 2.00 / 3.62 / 8.58 | none | arXiv 2107.03374 |
| AlphaCode decoder-only 13M | 13 M | 1.5 / 3.6 / 8.6 | none | arXiv 2203.07814 |
| Codex-25M / 42M / 85M | 25 / 42 / 85 M | 3.21 / 7.1 / 12.89; 5.06 / 8.8 / 15.55; 8.22 / 12.81 / 22.4 | none | arXiv 2107.03374 |
| TabNine free model | unknown | 2.58 / 4.35 / 7.59 | none | arXiv 2107.03374 |
| PyBlissa-Coder-40M | 39.9 M | 14.6% pass@1, reported by its authors, not reproduced | 4.4%, reported by its authors | huggingface.co/Rohanify/PyBlissa-Coder-40M-GGUF |
| Pico-1.3-Coding-Large | 6.9 M | not tested (its card says "expect about 0 %") | not tested | huggingface.co/Convergent-ch/Pico-1.3-Coding-Large |
| TinyPy models (Naïr et al.) | about 1 M | not evaluated (synthetic programs only) | not evaluated | arXiv 2407.10194 |
