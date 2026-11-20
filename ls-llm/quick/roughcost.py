"""Rough L1 cycle estimate per token (orchestrator's planning tool; the exact numbers come from lsllm/ls/cost.py)."""
import sys


def cycles(D=64, L=5, H=8, G=4, F=172, V=512, T=256, ffn="swiglu", parts=False):
    d = D // H
    row = lambda n: n + 24 + (4 if n > 255 else 0)
    norm = 120 + 24 * D
    qkv = (D + 2 * G * d) * row(D)
    rope = (H + G) * 2 * ((d // 2) * 22 + 12)
    cache = G * d * 28
    scores = H * T * (d + 28)
    mix = D * (row(T))
    wo = D * (row(D) + 4)
    if ffn == "swiglu":
        ff = 2 * F * row(D) + F * 4 + F * 20 + D * (row(F) + 4)
    else:
        ff = F * row(D) + D * (row(F) + 4)
    layer = 2 * norm + qkv + rope + cache + scores + mix + wo + ff
    out = norm + V * row(D) + V * 24 + 300
    total = L * layer + out
    if parts:
        return {"norms": L * 2 * norm + norm, "qkv": L * qkv, "rope+cache": L * (rope + cache), "scores": L * scores, "mix": L * mix, "wo": L * wo, "ffn": L * ff, "output+argmax": V * row(D) + V * 24}
    return total


def params(D, L, H, G, F, V, ffn="swiglu"):
    d = D // H
    return V * D + L * (2 * D * D + 2 * D * G * d + (3 if ffn == "swiglu" else 2) * D * F + 2 * D) + D


if __name__ == "__main__":
    shapes = [("stories260K 8h/4kv", dict(D=64, L=5, H=8, G=4, F=172)), ("4h/2kv", dict(D=64, L=5, H=4, G=2, F=172)), ("2h/2kv", dict(D=64, L=5, H=2, G=2, F=172)),
              ("2h/1kv", dict(D=64, L=5, H=2, G=1, F=172)), ("relu F=255 8h", dict(D=64, L=5, H=8, G=4, F=255, ffn="relu")), ("relu F=255 2h/2kv", dict(D=64, L=5, H=2, G=2, F=255, ffn="relu")),
              ("L4 D80 2h/2kv F214", dict(D=80, L=4, H=2, G=2, F=214)), ("L3 D96 2h/2kv F256", dict(D=96, L=3, H=2, G=2, F=255)), ("L6 D64 2h/2kv", dict(D=64, L=6, H=2, G=2, F=172)),
              ("L4 D64 2h/2kv", dict(D=64, L=4, H=2, G=2, F=172)), ("L2 D128 2h/2kv F255", dict(D=128, L=2, H=2, G=2, F=255))]
    print("%-24s %8s %6s  %s" % ("shape", "params", "kvKB", "  ".join("T=%-3d cyc/tok (tok/s)" % t for t in (16, 64, 256))))
    for name, kw in shapes:
        ffn = kw.get("ffn", "swiglu")
        p = params(kw["D"], kw["L"], kw["H"], kw["G"], kw["F"], 512, ffn)
        kv = 2 * kw["L"] * kw["G"] * (kw["D"] // kw["H"]) * 256 / 1024
        print("%-24s %8d %6.0f  %s" % (name, p, kv, "  ".join("%9d (%5.0f)    " % (cycles(T=t, V=512, **kw), 1e8 / cycles(T=t, V=512, **kw)) for t in (16, 64, 256))))
    print(cycles(T=256, parts=True))
