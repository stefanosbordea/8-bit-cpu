# How we work

Two people, one design, both able to explain all of it.

## The rules

1. **Nobody verifies their own module.** The designer writes the module; the other person writes its testbench from the spec tables in the design document, not from the designer's code.
2. **Every change goes through a pull request** that the other person reads before it is merged into `main`. Small pull requests, one module or one fix each.
3. **The spec comes first.** If the code and the design document disagree, fix the document first (and bump its version), then the code.
4. **Predictions are frozen before measurements.** Cycle counts are derived on paper and written down before the board runs the program.
5. **Keep a log.** One line per work session in `docs/log.md`: date, who, what. The contribution statement in the report is copied from it.

## Branches and commits

- Branch names: `loucas/fsm`, `stefanos/alu`, `docs/isa-v0.2`.
- Commit messages are plain sentences that say what changed: `Add the ALU flags and their testbench`.
- Never commit Vivado build output; `.gitignore` already excludes it.

## VHDL conventions

- VHDL-2008, `ieee.numeric_std` only (never `std_logic_arith`). Set the same standard in Vivado and in `nvc`.
- One entity per file; the file is named after the entity: `src/ls_alu.vhd` holds `entity ls_alu`.
- Ports: `clk`, synchronous active-high `rst`, clock enable `ce`. One clock in the whole design: never divide a clock or clock a register from a button.
- Registers live in clocked processes. Combinational processes assign every output in every branch, so no latches appear (check the synthesis log for "latch").
- Signed arithmetic uses `signed` from end to end; `resize` before adding values of different widths.

## Testbenches

- `tb/tb_<module>.vhd`, self-checking: count mismatches, print the first few, and end with `PASSED` or `NOT YET`, like the lab templates.
- Expected values come from the spec or from our own reference models, never from the design under test.
- Run them in `nvc` on a laptop and in XSim in the lab:

```sh
nvc --std=2008 -a src/ls_alu.vhd tb/tb_alu.vhd -e tb_alu -r
```
