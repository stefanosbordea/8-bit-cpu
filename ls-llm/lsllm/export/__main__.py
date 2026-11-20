import argparse

from lsllm.export import PROMPT, export

ap = argparse.ArgumentParser(prog="python -m lsllm.export", description="Write the files that put an integer model on the board.")
ap.add_argument("model_dir", help="directory with model.json and model.npz (SPEC Part E)")
ap.add_argument("out_dir")
ap.add_argument("--prompt", default=PROMPT, help=r"the prompt in the memory image, canonical Python; \n stands for a line break")
ap.add_argument("--tokens", type=int, default=8, help="generated tokens with a full trace in the test vectors")
ap.add_argument("--no-filter", action="store_true", help="build the program without the syntax filter")
ap.add_argument("--temperature", type=float, default=0.0, help="0: greedy; above 0: the program samples, and GUMBEL is exported")
ap.add_argument("--no-docstring", action="store_true",
                help="the filter never lets the model open a triple-quoted string: for prompts that are only a name, such as `def bubble_sort(`")
form = ap.add_mutually_exclusive_group()
form.add_argument("--prompts", metavar="FILE", help=r"a table of prompts chosen by the switches: one prompt per line (\n for a line break), at most 64")
form.add_argument("--asks", type=int, metavar="N", help="the same table from the first N requests of data/asks/asks.jsonl")
form.add_argument("--uart-prompt", action="store_true", help="the program reads its prompts from the UART (needs the instruction RX)")
args = ap.parse_args()
if args.no_docstring:
    from lsllm.export import tables
    tables.TRIPLE_QUOTES = False
prompts = None
if args.prompts:
    with open(args.prompts) as f:
        prompts = [line.rstrip("\n").replace("\\n", "\n") for line in f if line.strip()]
if args.asks:
    from lsllm.asks import ASKS, load_jsonl, request_to_prompt
    prompts = [request_to_prompt(ask["request"]) for ask in load_jsonl(ASKS)[:args.asks]]
for line in export(args.model_dir, args.out_dir, args.prompt.replace("\\n", "\n"), args.tokens, not args.no_filter, args.temperature,
                   prompts, args.uart_prompt):
    print(line)
