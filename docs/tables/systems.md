Published speeds of small language models on small machines, copied from `RESEARCH.md` (section 10, table B, where
every row has its link). The timing scope differs from row to row and the models differ, so this table places our
number among the others; it is not a ranking. Our row is a count of the simulator, the others were measured by their
authors on hardware.

| system | machine and clock | model | arithmetic | tokens per second |
|---|---|---|---|---|
| llama2.c64 | Commodore 64 with RAM expansion, about 1 MHz | stories260K | float32 in software | about 0.002 (one token in about 8 minutes) |
| gbc-transformer | Game Boy Color | stories260K | int8 weights, 16-bit and 32-bit activations, integer division | 0.0059 |
| elya-nes | NES, 6502 at 1.79 MHz | own model, 102,400 ternary weights, context 20 | ternary weights, 4-bit activations | 1.6 |
| esp32-llm | ESP32-S3, 240 MHz, two cores | stories260K | float32 | 19.1 (a fork: about 32.8) |
| EmbedLlama | STM32H7A3, 280 MHz | stories260K | float32 | up to about 87 |
| ecp5-inference | Lattice ECP5-45F FPGA, 31.25 MHz | stories260K, context at most 32 | int8 and power-of-two shifts, 16-bit tables | 14.0 |
| fpga-tinystories | Intel MAX 10 FPGA, 60 MHz, small vector processor | own model, 4.72 M parameters (4.20 M ternary), 16 layers, width 128 | ternary weights, int8 activations, 24-bit sums, tables | 55.6 at context 64, 50.3 at 256 |
| Deeploy on Siracusa | 8 RISC-V cores at 360 MHz with an accelerator | TinyStories Llama, width 64, 8 layers | int8 in matrix products, 32-bit divisions in the norm | 340 |
| LLaMA2-on-OpenLA500 | LoongArch soft core on an Artix-7 with a float coprocessor | stories260K | float32 | 4.8 at the first token to 1.9 at token 32 |
| PocketTPU | VexRiscv soft core on a Cyclone V, 50 MHz, dot-product unit | GPT-2 style, 1 M parameters | int8 weights | 4.25 |
