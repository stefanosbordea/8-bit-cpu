"""An analytic cycle model fitted to the generated program, for shapes too large to build (Tier B: weights in DDR2).

lsllm.ls.cost counts cycles on the real generated program, but only for models whose weights fit the 2^19-byte
data memory. The program is regular, so its cycle count is a linear function of a few shape terms. This script
generates programs for random shapes that do fit, fits the coefficients by least squares, and reports the largest
error on shapes that were not used for the fit.

    PYTHONPATH=. .venv/bin/python tools/costfit.py relu       prints the fit and writes docs/cost_fit_relu.json
"""
import json
import random
import sys

import numpy as np

from lsllm.config import ModelConfig
from lsllm.ls import cost

FFN = sys.argv[1] if len(sys.argv) > 1 else "swiglu"
OPTIONS = {"norm": "rms", "ffn": FFN}
NAMES = ["multiply-adds", "L*D", "L*kv", "L*F", "V", "L*H*n", "L*H*d", "L*H", "L", "D", "1"]


def terms(cfg, n):
    """Shape terms of one token with n keys in the context (position n - 1)."""
    L, D, F, V, H, d, kv = cfg.n_layers, cfg.dim, cfg.hidden_dim, cfg.vocab_size, cfg.n_heads, cfg.head_dim, cfg.kv_dim
    products = L * (2 * D * D + 2 * kv * D + (3 if cfg.ffn == "swiglu" else 2) * F * D + 2 * H * n * d + H * n + 2 * D) + D + V * D
    return [products, L * D, L * kv, L * F, V, L * H * n, L * H * d, L * H, L, D, 1]


def predict(coef, cfg, n):
    return float(np.dot(coef, terms(cfg, n)))


def sample(rng):
    H = rng.choice([1, 2, 2, 4, 8])
    G = rng.choice([g for g in (1, 2, 4, 8) if g <= H and H % g == 0])
    d = rng.choice([8, 12, 16, 24, 32, 40, 48, 64])
    D = H * d
    if D > 160:
        return None
    return ModelConfig(vocab_size=rng.choice([256, 512]), dim=D, n_layers=rng.randint(1, 6), n_heads=H, n_kv_heads=G,
                       hidden_dim=rng.randrange(32, 500, 2), max_seq_len=256, ffn=FFN)


def main():
    rng, rows = random.Random(0), []
    while len(rows) < 150:
        cfg = sample(rng)
        if cfg is None:
            continue
        n = rng.choice([1, 2, 16, 64, 100, 200, 256])
        try:
            rows.append((cfg, n, cost.cycles_per_token(cfg, OPTIONS, n - 1)[0]))
        except Exception:                     # the shape does not fit the ROM or the data memory
            continue
    fit, held = rows[:100], rows[100:]
    A = np.array([terms(cfg, n) for cfg, n, c in fit], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(A, np.array([c for cfg, n, c in fit], dtype=np.float64), rcond=None)
    error = lambda part: max(abs(predict(coef, cfg, n) - c) / c for cfg, n, c in part)
    out = {"options": OPTIONS, "terms": NAMES, "coefficients": [round(float(v), 4) for v in coef],
           "fit_shapes": len(fit), "largest_error_fit": error(fit), "held_out_shapes": len(held), "largest_error_held_out": error(held)}
    json.dump(out, open("docs/cost_fit_%s.json" % FFN, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
