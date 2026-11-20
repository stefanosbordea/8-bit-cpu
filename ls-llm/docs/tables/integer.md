The arithmetic is fixed by `docs/SPEC.md`: int8 weights in -127..127 and int8 activations, one power-of-two scale per
tensor, products accumulated in a signed 24-bit accumulator, requantisation by an arithmetic right shift with clamp
(rounding by a preload of half the divisor), and every non-linear function as a 256-entry table indexed by an int8
value: the norm (1 / sqrt of the mean square), exp for the attention weights, a reciprocal for their sum, RoPE sine and
cosine. There is no division and no floating point at inference. Attention: score codes of 7 bits relative to the
exact row maximum, weights from the exp table, and a division by the sum done with a bit count (`NRM`), a
128-entry reciprocal table and two shifts (`RQV`).

Three implementations of this arithmetic exist and are tested against each other bit for bit:
the QAT forward pass (`lsllm/qat/model.py`, float32 tensors that hold whole numbers), the integer reference in numpy
(`lsllm/intref/model.py`) and the program for the LS CPU run in the cycle-exact simulator (`lsllm/ls/`). The tests
compare logits, intermediate vectors, KV cache, chosen tokens and UART bytes on small random models, on saturating
models and on real exported models, including the syntax filter, sampling and the attention window (840 tests in
`tests/`). Quantization-aware training starts from the calibrated float model and trains the integer model itself,
with straight-through gradients and a distillation term to the float model.

The integer model was calibrated on training windows only. Its loss and the loss of the float model on the same
validation text are in the model table (section 2); "after calibration only" is the integer loss before any
quantization-aware step. The benchmark rows marked int8 were produced by the integer reference, with the same syntax
filter tables as the CPU.
