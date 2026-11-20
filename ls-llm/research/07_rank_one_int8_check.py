"""Does a rank-one direction edit W - a*u(u^T W) survive int8 rounding with power-of-two scales?

Synthetic check for research/07_weight_space_editing.md: random 64 x 172 matrices (the shape of the
baseline SwiGLU down-projection), not LS-LLM weights. Run: .venv/bin/python research/07_rank_one_int8_check.py
"""
import numpy as np

D, H, SEEDS = 64, 172, 200  # residual width, hidden width, repetitions
rng = np.random.default_rng(7)


def one_case(dist, alpha, cols, per_row):
    """Mean statistics of the edit over SEEDS random matrices and directions."""
    rows = []
    for g in range(SEEDS):
        w = rng.standard_normal((D, H)) if dist == "gauss" else rng.laplace(size=(D, H))
        u = rng.standard_normal(D)
        u /= np.linalg.norm(u)
        top = np.abs(w).max(axis=1, keepdims=True) if per_row else np.abs(w).max()
        scale = 2.0 ** np.ceil(np.log2(top / 127.0))            # power-of-two scale, per row or per tensor
        w_int = np.clip(np.round(w / scale), -127, 127)
        w_q = w_int * scale                                       # the deployed matrix
        proj = u @ w_q                                            # what each column writes along u
        mask = np.zeros(H)
        mask[np.argsort(-np.abs(proj))[:cols]] = 1.0              # edit only the strongest writer columns
        target = w_q - alpha * np.outer(u, mask * proj)
        raw = np.round(target / scale)                            # same scale: shifts stay unchanged
        new_int = np.clip(raw, -127, 127)
        d_int, d_flt = (new_int - w_int) * scale, target - w_q
        cos = (d_int * d_flt).sum() / (np.linalg.norm(d_int) * np.linalg.norm(d_flt))
        sel = mask > 0
        left = np.linalg.norm((u @ (new_int * scale))[sel]) / np.linalg.norm(proj[sel])
        changed = new_int != w_int
        rows.append([changed.mean(), (np.abs(raw) > 127).mean(), cos, left,
                     np.linalg.norm(d_flt) / np.linalg.norm(w_q), np.abs(new_int - w_int)[changed].mean()])
    return np.mean(rows, axis=0)


print("dist    scale  alpha cols | changed clipped cos(int,float) left-on-u ideal rel.change mean-step-LSB")
for dist in ("gauss", "laplace"):
    for per_row in (False, True):
        for alpha, cols in ((1, H), (2, H), (1, 16), (6, 16)):
            r = one_case(dist, alpha, cols, per_row)
            print(f"{dist:7s} {'row' if per_row else 'tensor':6s} {alpha:5d} {cols:4d} | {r[0]:7.3f} {r[1]:7.4f} "
                  f"{r[2]:14.3f} {r[3]:9.3f} {abs(1 - alpha):5d} {r[4]:10.3f} {r[5]:13.2f}")
