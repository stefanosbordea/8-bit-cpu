- **Nothing ran on the board or in Vivado.** Cycles are counted on the generated program in a simulator of the level-L1
  CPU of `docs/SPEC.md`, which is a proposal: the course CPU (L0) plus the additions of `docs/HARDWARE_CHANGES.md`,
  section 1. The simulator, the integer reference and the training forward pass agree bit for bit; the VHDL of the
  additions does not exist yet. Timing closure at 100 MHz is assumed, not shown (the stream pipeline has about 4 ns of
  slack by the data sheet, `research/09_fpga_costs.md`).
- **Resource numbers of hardware changes are estimates** from data sheets and reference designs, not synthesis results.
- **The model is small and the scores show it.** Zeros are reported as zeros. The custom tasks are split into families
  that the generated training data covers ("near") and families it does not ("far"); a pass on a near task means the
  model has learned that kind of function, not that it can program.
- **Programs on request are recall, not programming.** The board model writes the programs of the request suite
  because it was trained on them until every token was right; a request outside the list gets text that looks like
  Python and is almost never the wanted program (the numbers are in section 3). The PC turns a sentence into the
  name of a request with fixed rules and word lists; a sentence that fits none of them is passed on as typed.
- **A request test checks behaviour, not the method.** A program that sorts passes the tests of any sorting
  request. The texts were chosen by name and by tests; the three texts of the last round that name a method
  (heap sort, binary insertion sort, Dijkstra) were read and do use it, the others were not all read. Before heap
  sort had a text, the model answered that request with its shell sort, which passed.
- **The request tests, request lines and phrases were written by AI agents** (Claude), not by the authors; the program
  texts were not (section 5). Two tests were corrected after they shut out correct programs (two_sum had two right
  answers; heap_sort did not accept sorting in place).
- **Programs written by the local model** (Qwen2.5-Coder-1.5B) are kept because they pass the tests of their request,
  nothing more; they are a small model's code and can be clumsy.
- **The exact repair of the board model costs general ability**: about 0.01 to 0.02 nats of loss on general text
  (section 3). The board model as a whole is worse on general text than the benchmark model (its capacity holds
  the request programs); it is never scored on the benchmarks.
- **Tier B is less trained than Tier A** (3,000 steps) and its speed is extrapolated; see below.
- **The long runs for a rented GPU were prepared, not run**: `notebooks/train_on_colab.ipynb` with its bundle
  (`tools/make_colab_bundle.py`). Everything reported was trained on the laptop (Apple M1, 8 GB).
- **The generated training functions are not guaranteed to be right.** The first batch was checked only for running;
  the later batch must pass tests written by the same small generator model.
- **The practice sets were used many times** (shape, data mix, decoding settings, weight edits). The test sets were
  run only for the rows of section 3. The validation text was used to pick checkpoints.
- **Shape search runs are short** (600 steps, one seed): they rank shapes that differ clearly; differences below about
  0.03 nats are inside what a second seed could change (not measured).
- **Tier B** speed comes from a formula fitted to the generated program (error below 1% inside the fitted range) and is
  an extrapolation for the larger shapes; the DDR2 controller was not built, its size and latency are not known.
- **Sampling** uses the 32-bit LFSR of the CPU through a 256-entry Gumbel table, so sampled scores are those of the
  hardware sampler, not of an ideal one.
- **Novelty.** We do not claim that anything here is first or best. A transformer language model on a custom 8-bit CPU
  with a streamed multiply-accumulate was not found in our search; the nearest published system is a small vector
  processor on a MAX 10 FPGA (section 7.6).
