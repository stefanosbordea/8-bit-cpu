# The LS Architecture

A streaming 8-bit CPU for int8 neural inference, written in VHDL for the Digilent Nexys A7-100T (Artix-7 XC7A100T).

Loucas Louka and Stefanos Bordea · CSEN 203L Digital Systems Lab, AUB Mediterraneo · Fall 2026

> **The idea:** pay for control once per stream, not once per operation.
> One instruction, `MAC #n`, runs a whole dot product. After a single fetch and decode, the control FSM stays in one `STREAM` state while the datapath overlaps address, multiply and accumulate, retiring one multiply-accumulate per clock.

## Documents

| Document | What is in it |
|---|---|
| [**Design document** (PDF)](docs/design/ls-architecture.pdf) | the principle, datapath, instruction set, control FSM, programs and predictions, board I/O, test plan, who builds what, schedule |
| [**State of the art** (PDF)](docs/research/sota-review.pdf) | literature review: where the idea comes from, what exists, what is uncommon about this project |
| [Project overview](docs/project-overview.md) | the plan in plain language, no background assumed |

## Predicted results

Every number below is a prediction from the program listings in the design document. It becomes a result when the cycle counter on the board shows it.

| Program (64-input neuron) | Instruction words | Cycles | Cycles per extra input |
|---|---:|---:|---:|
| **S**: one `MAC #64` (LS stream) | 12 | **112** | **1** |
| U: 64 unrolled classic `MAC`s | 75 | 300 | 4 |
| L: classic `MAC` in a loop | 14 | 560 | 8 |
| B: no MAC instruction | 17 | 1,328 | 20 |

The claim is S against U, the best conventional code for the same CPU: **2.68× (predicted)**, measured on one bitstream.

## Repository layout

```
docs/
  design/        design document: LaTeX sources, figures, PDF
  research/      state-of-the-art review: LaTeX sources, PDF
  style/         shared LaTeX style
  build.sh       rebuilds the PDFs
src/             VHDL sources, one entity per file
tb/              self-checking testbenches
constraints/     Nexys A7-100T constraints (.xdc)
```

## Milestones

| Date (2026) | Milestone |
|---|---|
| Fri 20 Nov | proposal |
| Tue 24 Nov | **M1**: the CPU runs programs (simulation) |
| Fri 4 Dec | **M2**: the neuron runs on the board; baselines measured |
| Thu 10 Dec | **M3**: the stream on the board; all measurements and reports |
| Fri 18 Dec | demonstration and defence |

## Who builds what

| | Designs | Verifies |
|---|---|---|
| Loucas | control: FSM and decoder, stream control, PC and program ROM, run modes and clock enable, counters, display | Stefanos' datapath: `tb_alu`, `tb_mul_rq`, `tb_mem`, `tb_io` |
| Stefanos | datapath: registers, ALU, pointers and memories, MAC unit, requantiser, input chain, LEDs, top level | Loucas' control: `tb_fsm`, `tb_stream`, `tb_ce_pause` |

How we work together: [CONTRIBUTING.md](CONTRIBUTING.md).
