"""For a list of shapes: the largest feed-forward width F whose generated data-memory image (complete program:
prompt buffer, stop rule, syntax filter) stays within the Tier A budget, with the real sizes from lsllm.ls.gen.

    PYTHONPATH=. .venv/bin/python tools/fit_budget.py [BUDGET_BYTES]
"""
import sys

from lsllm.config import ModelConfig
from lsllm.ls import cost

BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 110 * 4096          # 110 of the 135 block RAMs
OPT = {"norm": "rms", "ffn": "relu"}


def image(cfg):
    try:
        return len(cost.token_cycles(cfg, OPT, 0, stop=True, syntax_filter=True)[2].image)
    except ValueError:
        return 1 << 30


def largest_f(L, D, H, G, window, T=256):
    lo, hi = 0, 510
    make = lambda F: ModelConfig(n_layers=L, dim=D, n_heads=H, n_kv_heads=G, hidden_dim=F, ffn="relu", window=window, max_seq_len=T)
    if image(make(2)) > BUDGET:
        return None
    lo = 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if image(make(mid)) <= BUDGET:
            lo = mid
        else:
            hi = mid - 1
    return make(lo)


if __name__ == "__main__":
    print("budget %d bytes" % BUDGET)
    print("L D H G window T | F | parameters | image bytes | cycles @16 @64 @256 | tokens/s @64")
    for L, D, H, G, window, T in [(2, 160, 2, 1, 0, 256), (2, 160, 2, 1, 128, 256), (2, 144, 2, 1, 0, 256), (2, 144, 2, 1, 128, 256), (2, 128, 2, 1, 0, 256),
                                  (2, 128, 2, 1, 128, 256), (3, 128, 2, 1, 0, 256), (3, 128, 2, 1, 128, 256), (3, 112, 2, 1, 0, 256), (3, 112, 2, 1, 128, 256),
                                  (3, 96, 2, 1, 0, 256), (3, 96, 2, 1, 128, 256), (4, 80, 2, 1, 0, 256), (5, 64, 2, 1, 0, 256), (5, 64, 8, 4, 0, 256),
                                  (2, 160, 4, 2, 0, 256), (2, 160, 4, 1, 0, 256), (2, 144, 4, 1, 0, 256), (3, 128, 4, 1, 0, 256), (2, 160, 2, 1, 0, 192), (2, 144, 2, 1, 0, 192)]:
        cfg = largest_f(L, D, H, G, window, T)
        if cfg is None:
            print(L, D, H, G, window, T, "| does not fit")
            continue
        span = lambda n: min(n, window or n, T) - 1
        cyc = [cost.token_cycles(cfg, OPT, span(n), stop=True, syntax_filter=True)[0] for n in (16, 64, 256)]
        print(L, D, H, G, window, T, "|", cfg.hidden_dim, "|", cfg.n_params(), "|", image(cfg), "|", cyc, "| %.1f" % (1e8 / cyc[1]), flush=True)
