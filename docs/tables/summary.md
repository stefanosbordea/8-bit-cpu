- **Model (Tier A).** 350,880 int8 parameters: 2 layers, width 160, 4 query heads sharing 1 key/value head, ReLU feed-forward of 219, 512 tokens, context 256. Its complete data memory image is 450,432 bytes (110 of the 135 block RAMs), the program 976 words.
- **Speed (level L1, 100 MHz, greedy with the syntax filter).** 495,456 / 549,600 / 766,176 cycles per token = **202 / 182 / 131 tokens per second** at context 16 / 64 / 256. The stream instruction makes the same program 3.20 / 3.15 / 3.02 times faster than one classic 4-cycle `MAC` per product.
- **Integer arithmetic.** Validation loss 2.1770 nats per token in float and 2.2506 with the integers that the CPU computes (+0.0736); training forward pass, integer reference and generated LS program give identical tokens.
- **Scores on the test sets** (pass@1 greedy, best of filter on and off; pass@10 from 10 samples at temperature 0.8). Every model went
  through the same harness; the comparators are public models run as published. Details and the ladder below pass@1: section 3.

  | model | parameters | custom 200 | MBPP | HumanEval | custom: near families, tasks passed of 120 | custom: far families, of 80 | custom 200 pass@10 | MBPP pass@10 | HumanEval pass@10 |
  |---|---|---|---|---|---|---|---|---|---|
  | **Tier A, float** | 350,880 | 4.0% | 0.8% | 0.0% | 7 | 1 | 12.0% | 0.8% | 0.0% |
  | **Tier A, int8 (what the CPU computes)** | 350,880 | 6.5% | 0.4% | 0.0% | 12 | 1 | 11.0% | 1.4% | 0.0% |
  | **Tier A, earlier model (10,000 steps; the editing sessions used it), float** | 350,880 | 4.5% | 0.4% | 0.0% | 9 | 0 | 6.0% | 0.8% | 0.0% |
  | **Tier A, earlier model (10,000 steps; the editing sessions used it), int8 (what the CPU computes)** | 350,880 | 5.0% | 0.2% | 0.0% | 10 | 0 | 9.5% | 0.6% | 0.0% |
  | **Tier B, float** | 1,252,032 | 6.5% | 0.2% | 0.0% | 13 | 0 | 12.5% | 1.6% | 0.0% |
  | **Tier B, int8 (what the CPU computes)** | 1,252,032 | 6.0% | 1.0% | 0.0% | 12 | 0 | 13.0% | 2.0% | 0.6% |
  | stories260K | 260,032 | 0.0% | 0.0% | 0.0% | 0 | 0 | - | - | - |
  | ngram-432KB | 77,988 | 0.0% | 0.0% | 0.0% | 0 | 0 | - | - | - |
  | Pico-1 | 254,400 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0.0% | 0.0% | 0.0% |
  | Pico-1.2 | 460,608 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0.0% | 0.0% | 0.0% |
  | CinnabarLM-4M-Python-Base | 4,229,568 | 1.0% | 0.6% | 0.0% | 2 | 0 | - | - | - |
  | Pico-1.3-Coding-Large | 6,917,376 | 8.5% | 1.6% | 0.6% | 11 | 6 | - | - | - |
  | chispa-20m-python | 20,119,360 | 5.0% | - | - | 7 | 3 | - | - | - |

  Near families are kinds of task that the generated part of the training text also covers (other tasks, same kind); far families are not.
  Our models pass near-family tasks and at most 1 far-family task. The two larger public models pass several far-family tasks.

- **Programs on request (board model, int8, on the simulated CPU).** Of 401 requests such as "make me a bubble sort program", 400 come back as a program that passes the tests of its request; in all 401 the CPU program sends exactly the tokens of the integer reference. Median 0.33 s per program at 100 MHz, longest 1.54 s. This is recall of programs the board model was trained on, not a benchmark (section 3).
- **Weight editing without gradients (4 sessions, on the earlier Tier A model).** Practice scores rose, the held-out custom score did not: 5.0% unedited, 4.0%, 4.0%, 4.5%, 3.5% after the sessions. No edited model is used (section 10).
- **What was not done.** Nothing ran on the board or in Vivado; cycle counts come from a simulator of the proposed CPU (section 13).
