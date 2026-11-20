What was kept and why:

- **Fewer, wider query heads; shared key/value heads: kept.** The score loop costs head size + 60 cycles per key and
  head, so 8 heads of size 8 pay the 60 cycles eight times. Fewer heads were never worse in loss. Sharing one key/value
  head among the query heads halves the KV cache and the RoPE tables; the final shape uses 4 query heads on 1 shared head.
- **ReLU feed-forward instead of SwiGLU: kept.** Better at equal parameters in both pairs above, one matrix and the
  gating pass (table lookup and product per hidden unit) fewer on the CPU.
- **Vocabulary 512 instead of 256: kept.** With 256 tokens the output layer is cheaper per token, but a token carries
  1.76 bytes instead of 2.09 and the loss per byte is slightly worse: fewer bytes of Python per second.
- **Attention window: implemented and verified end to end, used only where the model table lists a window.** With a
  window of W keys the cost of a token stops growing at context W and the KV cache becomes a ring of W positions. The
  float model, the QAT forward pass, the integer reference and the LS program support it and agree bit for bit
  (`tests/test_ls_gen_window.py`, `tests/test_qat_exact.py`). In the 600-step pair above the window of 64 lost nothing.
- **Prefill skip: kept (exact).** A prompt token needs no output layer, no argmax and no UART; the program jumps back
  after the layers. The saving is the difference between the "generated token" and "prompt token" rows of section 7.1.
- **Syntax filter: kept.** 6,748 cycles per generated token (section 9).
- **Greedy decoding: kept as the default; sampling is available.** Gumbel-max sampling costs 28 cycles per token id
  (14,348 cycles per token at 512 tokens) for the random byte, the table and the addition. The sampled rows of
  section 3 show what it buys.
- **Early exit: not kept.** Reading the logits after an earlier layer needs a second output layer for every token
  (the output layer costs about half a layer of this shape), and the earlier layer is rarely sure enough; the measured
  numbers follow.
