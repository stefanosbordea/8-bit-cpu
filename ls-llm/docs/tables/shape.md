Every shape was trained for 600 steps on the same text with the same seed, and its speed was counted on the LS program
that the generator writes for it. Two things were learned before any shape was chosen:

- **The budget has to be counted on the real memory image.** Parameters plus KV cache undercounts by 50 to 70 KB: the
  RoPE tables in the position records, the filter tables and alignment are block RAM too. Tier A is "the data memory
  image fits 110 block RAMs = 450,560 bytes" (D-shape-1). The first round of the search (`runs/p`) used the loose count;
  its budget-filling shapes are over the budget and are marked so. The second round (`runs/q`) uses the largest
  feed-forward width that fits, found by `tools/fit_budget.py`.
- **Width beats depth at this size, and shared key/value heads pay for it.** Going from 5 layers of width 64 to 2 layers
  of width 160 at equal memory lowers the validation loss by 0.36 nats per token. Four query heads that share one
  key/value head halve the KV cache and the RoPE tables, and the freed memory is worth more as feed-forward units.
  ReLU with a larger hidden layer beats SwiGLU at equal parameters and removes the gating pass from the program.

**Chosen shape (D-shape-2):** 2 layers, width 160, 4 query heads of size 40 on 1 key/value head, ReLU feed-forward of
219, 350,880 parameters, 450,432 bytes. On the front of quality against speed it has the lowest loss of all shapes
inside the budget and is within 15% of the fastest. The stories260K shape (5 x 64, 8 heads, 4 key/value heads,
SwiGLU) is the slowest shape of the table at long context and has the highest loss but one.

The chart shows validation bits per byte against bytes of Python per second at context 64; the table has every run.
Limits of this search: 600 steps and one seed per shape, so only clear differences count (section 13).
