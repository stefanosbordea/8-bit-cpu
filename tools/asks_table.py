"""docs/tables/asks.md: the request suite, per board model. Rows come from the files that
`python -m lsllm.asks.run --json` and `tools/board_verify.py --json` wrote; the manifest names them:

    "board": [{"label": "Tier A board model, int8", "asks": "results/asks_NAME_int.json", "verify": "results/board_NAME.json",
               "texts": ["data/asks/train_texts.jsonl", "data/asks/qwen_texts.jsonl"]}, ...]      (texts: what it was trained on)
    "unseen": {"label": ..., "asks": "results/asks_OLD_int.json", "texts": [...]}        (an older model: requests it never saw)

    PYTHONPATH=. .venv/bin/python tools/asks_table.py runs/final.json
"""
import json
import os
import sys
from collections import Counter

from lsllm.asks import ASKS, Resolver, load_jsonl


def names_of(paths):
    """Request ids that have a text in the files; the first file that has a request decides."""
    return {json.loads(line)["id"] for path in paths if os.path.exists(path) for line in open(path)}


def main(manifest):
    spec = json.load(open(manifest))
    suite = load_jsonl(ASKS)
    asks = {ask["id"]: ask for ask in suite}
    frozen = [row for path in spec["board"][-1]["texts"] for row in load_jsonl(path)] if spec.get("board") else []
    human = [row for row in frozen if not row["source"].startswith("Qwen")]
    bundled = sum(1 for row in human if "helpers" in row.get("kind", ""))
    long = sum(1 for row in human if row.get("tokens", 0) > 230)
    generated = len(frozen) - len(human)
    categories = sorted({ask["category"] for ask in asks.values()})
    resolve, phrases = Resolver(suite), {}
    if os.path.exists("data/asks/phrases.jsonl"):
        phrases = {row["name"]: row["phrases"] for row in load_jsonl("data/asks/phrases.jsonl")}
    typed = right = 0                                   # every request line, phrase and other name of the suite, typed as a request
    for ask in suite:
        for line in ask["requests"] + phrases.get(ask["name"], []) + ask.get("aliases", []):
            typed += 1
            right += resolve(line.replace("_", " "))[0] == ask["name"]
    lines = ["The request suite (`data/asks/asks.jsonl`) has %d programs with tests (%s). A request such as \"make me a bubble sort" % (
                 len(asks), ", ".join("%s %d" % pair for pair in Counter(ask["category"] for ask in suite).most_common())),
             "program\" becomes the prompt `def bubble_sort(`; the model writes the rest with the syntax filter on and no docstring; the",
             "result is run against the tests of the request in the sandbox.",
             "**This is a recall test of programs that are in the training text of the board model, not a benchmark.** Where the texts",
             "come from: %d requests have a human-written function from the licensed corpus that passes the tests (renamed to the" % len(human),
             "request's name, docstring removed; in %d cases the helper functions of its file were moved inside; %d need more than" % (bundled, long),
             "230 tokens); for %d more the local model Qwen2.5-Coder-1.5B wrote a program that passes the tests; %d %s no text and" % (
                 generated, len(asks) - len(frozen), "has" if len(asks) - len(frozen) == 1 else "have"),
             "%s not trained. Every source and licence is listed in `docs/ASKS_SOURCES.md`. The board model is never used for a" % ("is" if len(asks) - len(frozen) == 1 else "are"),
             "benchmark score (D-board-1).", "",
             "| model | requests passed | of those with a training text | texts written out exactly | " + " | ".join(categories) + " |",
             "|" + "---|" * (4 + len(categories))]
    failing = {}
    for entry in spec.get("board", []):
        if not os.path.exists(entry["asks"]):
            continue
        rows = json.load(open(entry["asks"]))
        passed = [row["id"] for row in rows if row["outcome"] == "pass"]
        have = names_of(entry.get("texts", ["data/asks/train_texts.jsonl"]))
        by = Counter(asks[ask_id]["category"] for ask_id in passed)
        total = Counter(ask["category"] for ask in asks.values())
        exact = "-"
        if entry.get("polish") and os.path.exists(entry["polish"]):       # counted with the CPU's tie rule, from the written model
            r = json.load(open(entry["polish"]))
            exact = "%d of %d" % (r["exact"], r["texts"])
        elif entry.get("run") and os.path.exists(os.path.join(entry["run"], "result.json")):
            r = json.load(open(os.path.join(entry["run"], "result.json")))             # counted by the trainer: every other token at least one code lower
            exact = "%d of %d" % (r["recall_exact"], r["recall_texts"]) if "recall_exact" in r else "-"
        lines.append("| %s | %d of %d | %d of %d | %s | %s |" % (entry["label"], len(passed), len(rows), sum(ask_id in have for ask_id in passed), len(have), exact,
                                                             " | ".join("%d of %d" % (by[c], total[c]) for c in categories)))
        failing[entry["label"]] = (Counter(row["outcome"] for row in rows if row["outcome"] != "pass"),
                                   [row["id"].split("/")[1] for row in rows if row["outcome"] != "pass" and row["id"] in have],
                                   [row["id"].split("/")[1] for row in rows if row["outcome"] == "pass" and row["id"] not in have])
    lines += ["", "\"Texts written out exactly\": the float and QAT rows are the trainer's count (every other token at least one code below the",
              "wanted one at every position); the repaired model is counted from its written files with the rule of the CPU (the smallest",
              "index among the largest logits wins), by `lsllm/intref/polish.py`. A request passes if the program passes its tests; a",
              "program can pass without being the stored text.", ""]
    for label, (outcomes, names, lucky) in failing.items():
        lines.append("%s: not passed: %s. Trained but not passed: %s.%s" % (
            label, ", ".join("%d %s" % (n, kind) for kind, n in outcomes.items()) or "none", ", ".join(names) or "none",
            " Passed without a training text: %s." % ", ".join(lucky) if lucky else ""))
    last = spec["board"][-1] if spec.get("board") else {}
    if last.get("polish") and os.path.exists(last["polish"]) and os.path.exists(last.get("val_log", "")):
        import re
        fixed = json.load(open(last["polish"]))
        body = [float(v) for v in re.findall(r"body only ([0-9.]+)", open(last["val_log"]).read())]       # before and after the repair
        float_body = json.load(open(os.path.join(last["float_run"], "result.json")))["val_body"]
        bench_body = json.load(open(os.path.join(last["bench_float_run"], "result.json")))["val_body"]
        lines += ["", "**How the board model is made** (D-board-2, D-board-3, D-board-4). It starts from the earlier Tier A model and is trained further on",
                  "general text plus request rows: one request program per row, from the first token of the context, as on the board; the run keeps",
                  "the weights that write the most texts exactly. The same is done with the integers (QAT, two stages). What is still wrong",
                  "then is mended exactly: a logit is `rq(y . E[v] + c_l, sh_l)`, and an entry of E can move up to 7 codes without changing the",
                  "input embedding (the input reads `rqr(E[v], 3)`), so the effect of such a change on every logit of every text is known by",
                  "integer arithmetic alone. %d repairs moved %s of the %s entries of E; no input embedding changed; %d of %d texts are" % (
                      fixed["repairs"], "{:,}".format(fixed["entries_of_E_moved"]), "{:,}".format(512 * 160), fixed["exact"], fixed["texts"]),
                  "exact afterwards. The price is paid on general text (loss on the body tokens of the validation text, nats per token):",
                  "the model it started from, float %.4f; board model, float %.4f; int8 after QAT %.4f; after the repair %.4f." % (bench_body, float_body, body[0], body[-1])]
    for entry in spec.get("board", []):
        if entry.get("verify") and os.path.exists(entry["verify"]):
            rows = json.load(open(entry["verify"]))
            cycles, tokens = sum(row["cycles"] for row in rows), sum(row["tokens_sent"] for row in rows)
            ordered = sorted(row["cycles"] for row in rows)
            slowest = max(rows, key=lambda row: row["cycles"])
            lines += ["", "**The same requests on the simulated CPU** (%s; `tools/board_verify.py`: the generated LS program with the UART prompt" % entry["label"],
                      "protocol and the syntax filter, in the cycle-exact simulator, one request after the other like the board):", "",
                      "| | |", "|---|---|",
                      "| requests sent | %d |" % len(rows),
                      "| answers that pass the tests of their request | %d |" % sum(row["outcome"] == "pass" for row in rows),
                      "| answers equal to the integer reference, token by token | %d of %d |" % (sum(row["same_as_reference"] for row in rows), len(rows)),
                      "| tokens sent | {:,} |".format(tokens), "| cycles | {:,} |".format(cycles),
                      "| time per request at 100 MHz: median, longest | %.2f s, %.2f s (`%s`, %d tokens) |" % (
                          ordered[len(ordered) // 2] / 1e8, slowest["cycles"] / 1e8, slowest["name"], slowest["tokens_sent"]),
                      "| all requests at 100 MHz | %.1f s |" % (cycles / 1e8),
                      "| tokens per second over all requests (reading the prompt included) | %.1f |" % (tokens / (cycles / 1e8))]
    if last.get("verify") and os.path.exists(last["verify"]):
        rows = {row["name"]: row for row in json.load(open(last["verify"]))}
        lines += ["", "Three answers exactly as the simulated CPU sent them (requests typed as \"make me a bubblesort program\", \"make me binary search",
                  "trees\", \"shortest path\"; the PC turned them into `def bubble_sort(`, `def bst_insert(`, `def dijkstra(`):", ""]
        for name in ("bubble_sort", "bst_insert", "dijkstra"):
            if name in rows and rows[name]["outcome"] == "pass":
                lines += ["```python", rows[name]["text"], "```", "%d tokens, %s cycles = %.2f s at 100 MHz; passes the %d tests of its request.%s" % (
                    rows[name]["tokens_sent"], "{:,}".format(rows[name]["cycles"]), rows[name]["cycles"] / 1e8, len(asks["asks/" + name]["tests"]),
                    " (`Node` is the node class of the tests: a tree program needs one.)" if name == "bst_insert" else ""), ""]
    unseen = spec.get("unseen")
    if unseen and os.path.exists(unseen["asks"]):
        rows, have = json.load(open(unseen["asks"])), names_of(unseen["texts"])
        new = [row for row in rows if row["id"] not in have]
        lines += ["", "**Requests the model was not trained on.** %s had the texts of %d requests; of the other %d requests of the suite it" % (unseen["label"], len(have), len(new)),
                  "passed %d (%s). A model of this size writes the programs it was trained on; a request outside the list gets text that" % (
                      sum(row["outcome"] == "pass" for row in new), ", ".join(row["id"].split("/")[1] for row in new if row["outcome"] == "pass") or "none"),
                  "looks like Python but is almost never the right program. This is the measured limit, not a setting. (The pass of heap_sort",
                  "was the model's shell sort written under the asked name: the tests check that a list gets sorted, not how.)"]
    lines += ["", "**From a sentence to the name** (`lsllm.asks.Resolver`, on the PC, before anything is sent): the words of the request are",
              "the name of a request (\"make me a bubble sort program\"), a phrase or another name listed for a request (\"bubblesort\",",
              "\"primality test\", \"shortest path\"), a name inside a longer sentence (\"sort my list with bubble sort\"), or a typing",
              "error of one (\"buble sort\"). Of the %s request lines, phrases and other names of the suite, %s lead to their own request" % ("{:,}".format(typed), "{:,}".format(right)),
              "(`tests/test_asks_resolver.py`, `tests/test_asks_phrases.py`); the rest are names that several requests share, which go",
              "to one of them by a fixed rule. A sentence that fits no rule is not sent: `tools/board_ask.py` says that the model does not",
              "know this request and names the closest ones it does know (`--anyway` sends it as typed). The CPU always receives",
              "`def <name>(` as tokens.", "",
              "How a request reaches the CPU: the program takes one of up to 64 stored prompts by the switches, or reads a typed prompt from",
              "the UART (`tools/board_ask.py`; this needs the proposed `RX` instruction)."]
    open("docs/tables/asks.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[9:14]))


if __name__ == "__main__":
    main(sys.argv[1])
