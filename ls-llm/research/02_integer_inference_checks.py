"""Numerical checks for research/02_integer_inference.md (numpy only, runs in a few seconds).

Run: .venv/bin/python research/02_integer_inference_checks.py
Every number printed here is quoted in the markdown file with the tag [computed here].
"""
import numpy as np

rng = np.random.default_rng(0)
ACC_MAX = 2**23 - 1


def section(title):
    print("\n== " + title)


# 1. Floor versus round-to-nearest in shift requantisation --------------------------------------
section("1 requantisation error in output LSB (64-term int8 dot products)")
x = rng.integers(-128, 128, size=(200000, 64))
w = rng.integers(-127, 128, size=(200000, 64))
acc = (x * w).sum(axis=1)
for shift in (1, 4, 7, 10):
    exact = acc / 2**shift
    floor = acc >> shift
    near = (acc + (1 << (shift - 1))) >> shift
    print(f"shift {shift:2d}: floor mean {np.mean(floor - exact):+.4f} std {np.std(floor - exact):.4f} | "
          f"round-half-up mean {np.mean(near - exact):+.4f} std {np.std(near - exact):.4f}")

# 2. Overflow limits of a signed 24-bit accumulator --------------------------------------------
section("2 longest dot product that can never overflow 24 bits")
for name, worst in (("x,w in [-128,127]", 128 * 128), ("w in [-127,127]", 128 * 127), ("x,w in [-127,127]", 127 * 127)):
    print(f"{name:20s}: worst product {worst}, n_max = {ACC_MAX // worst}")
print("L1 bound for a row (signed int8 inputs): sum|w| <=", ACC_MAX // 128)
print("sum of squares, n = 64:", 64 * 128 * 128, "= 2^20 ;", "n = 172:", 172 * 128 * 128)
print("un-normalised value mix, T = 256, weights 0..127:", 256 * 127 * 128, "; weights 0..255:", 256 * 255 * 128)


def wrap24(v):
    return (v + 2**23) % 2**24 - 2**23


terms = np.concatenate([np.full(600, 127 * 127), np.full(600, -127 * 127), rng.integers(-16000, 16000, size=300)])
run = 0
for t in terms:
    run = wrap24(run + int(t))
true_sum = int(terms.sum())
print(f"wrap-around check: true partial sums peak at {600 * 127 * 127} (> 2^23), final true sum {true_sum}, "
      f"wrapped accumulator ends at {run}, equal: {run == true_sum}")

# 3. Reciprocal of the softmax denominator without division ------------------------------------
section("3 error of 1/S approximations, S = 1 .. 32512 (256 keys x 127)")
S = np.arange(1, 127 * 256 + 1)
k = np.floor(np.log2(S)).astype(int)            # position of the leading one
frac = S / 2.0**k - 1.0                          # S = 2^k (1 + frac)


def report(name, approx):
    ratio = approx * S                           # 1.0 means exact
    print(f"{name:46s} ratio min {ratio.min():.4f} max {ratio.max():.4f} mean {ratio.mean():.4f} "
          f"rms error {np.sqrt(np.mean((ratio - 1) ** 2)) * 100:5.2f}%")


report("power of two 2^-(k+1)", 2.0 ** -(k + 1))
centre = 1.0 / np.mean(2.0 ** -(k + 1) * S)
report(f"power of two, times constant {centre:.3f}", 2.0 ** -(k + 1) * centre)
report("SOLE ALDivision (1 mantissa bit)", 2.0 ** -(k + 1) * (1.636 - np.where(frac >= 0.5, 0.5, 0.0)))
for bits in (2, 3, 4, 5, 8):
    # mantissa m in [2^(bits-1), 2^bits): the leading one plus (bits-1) more bits
    m = np.floor(S * 2.0 ** (bits - 1 - k)).astype(int)
    for outbits in (7, 8):
        # entry = reciprocal of the bucket centre, scaled so that the largest entry uses the full range
        top = 2**outbits - 1
        entry = np.round(top * 2 ** (bits - 1) / (m + 0.5))
        approx = entry / (top * 2 ** (bits - 1)) * 2.0 ** (bits - 1 - k)
        # buckets hold one integer only when S itself has fewer than `bits` bits: then m is exact
        small = k < bits - 1
        approx[small] = np.round(top * 2 ** (bits - 1) / m[small]) / (top * 2 ** (bits - 1)) * 2.0 ** (bits - 1 - k[small])
        report(f"reciprocal table, {bits}-bit mantissa, {outbits}-bit entries", approx)

# 3b. Whole division-free normaliser in integers: o = y / S with two-octave code paths ------------
section("3b output error of o = y/S done as ((y >> (b+8)) * R[S >> b]) >> 5, S >> b in 64..255")
recip = np.round(2**13 / (np.arange(256) + 0.5)).astype(np.int64)        # R[a], 32..127 for a in 64..255
for label, half in (("round-half-up", 1), ("floor", 0)):
    worst, sq, bias, count = 0.0, 0.0, 0.0, 0
    for b in range(0, 8):                                                 # S from 64 up to 32512
        S = rng.integers(64 << b, min(256 << b, 127 * 256 + 1), size=200000)
        o_true = rng.uniform(-127, 127, size=S.size)                      # the exact weighted average of v
        y = np.round(o_true * S).astype(np.int64)                         # value-mix accumulator
        a = S >> b
        t = (y + half * (1 << (b + 7))) >> (b + 8)                        # requantise the accumulator to int8
        o = np.clip((t * recip[a] + half * 16) >> 5, -128, 127)           # multiply by reciprocal, requantise
        err = o - o_true
        worst = max(worst, np.abs(err).max())
        sq += np.sum(err**2)
        bias += err.sum()
        count += err.size
    print(f"{label:14s}: max |error| {worst:.0f} LSB, rms {np.sqrt(sq / count):.2f} LSB, mean {bias / count:+.2f} LSB "
          f"(true quotient uniform in -127..127; exact rounding alone gives rms 0.29)")

# 4. Softmax with an exp table: end-to-end error on random score rows --------------------------
section("4 total-variation distance to float softmax, 256 keys (mean over 2000 rows)")
variants = ((32, 255), (32, 127), (16, 255), (16, 127), (8, 127), (16, 16383))
print("columns: steps per octave / largest table value (16383 = two byte tables); 'score only' = float exp")
for sigma in (1.0, 2.0, 4.0):
    logits = rng.normal(0, sigma, size=(2000, 256))          # natural-log units
    truth = np.exp(logits - logits.max(1, keepdims=True))
    truth /= truth.sum(1, keepdims=True)
    line = f"logit std {sigma:.0f} nat (largest weight {truth.max(1).mean() * 100:4.1f}% of the mass):"
    for steps_per_octave, outmax in variants:
        step = np.log(2) / steps_per_octave                  # nats per score LSB
        q = np.round(logits / step).astype(int)
        d = np.clip(q.max(1, keepdims=True) - q, 0, 255)     # table index, 0 = the largest score
        table = np.round(outmax * 2.0 ** (-np.arange(256) / steps_per_octave))
        table[255] = 0                                       # index 255 means "255 steps or more below"
        e = table[d]
        p = e / e.sum(1, keepdims=True)                      # exact division: isolates table error
        line += f"  {steps_per_octave}/{outmax}: {np.mean(0.5 * np.abs(p - truth).sum(1)) * 100:5.2f}%"
    for steps_per_octave in (32, 8):
        step = np.log(2) / steps_per_octave
        q = np.round(logits / step)
        e = np.exp((q - q.max(1, keepdims=True)) * step)
        p = e / e.sum(1, keepdims=True)
        line += f"  score only {steps_per_octave}: {np.mean(0.5 * np.abs(p - truth).sum(1)) * 100:4.2f}%"
    print(line)

print("fixed reference instead of a max search (16 steps per octave, table 0..127): row maximum sits")
print("'gap' codes below the top code, so the largest weight is 127 * 2^(-gap/16):")
for sigma in (1.0, 2.0, 4.0):
    logits = rng.normal(0, sigma, size=(2000, 256))
    truth = np.exp(logits - logits.max(1, keepdims=True))
    truth /= truth.sum(1, keepdims=True)
    q = np.round(logits / (np.log(2) / 16)).astype(int)
    table = np.round(127 * 2.0 ** (-np.arange(256) / 16))
    table[255] = 0
    line = f"logit std {sigma:.0f} nat:"
    for gap in (0, 16, 32, 48, 64):
        e = table[np.clip(q.max(1, keepdims=True) - q + gap, 0, 255)]
        p = e / np.maximum(e.sum(1, keepdims=True), 1)
        line += f"  gap {gap} (top {int(table[gap])}): {np.mean(0.5 * np.abs(p - truth).sum(1)) * 100:5.2f}%"
    print(line)

# 5. 1/sqrt lookup for a sum of squares of about 20 bits ---------------------------------------
section("5 relative error of 1/sqrt(S) from a table, S = 1 .. 2^20")
print("one table bucket [a, a+1): geometric-centre entry has max error ((a+1)/a)^(1/4) - 1:")
print("  " + "  ".join(f"a={a}: {(((a + 1) / a) ** 0.25 - 1) * 100:.2f}%" for a in (1, 2, 4, 8, 16, 32, 64, 128)))
S = np.arange(1, 2**20 + 1, dtype=np.int64)
true = 1.0 / np.sqrt(S)


def bucket_entry(idx, shift):
    """Table value for index idx when S was shifted right by `shift`: geometric centre of the bucket."""
    return 1.0 / np.sqrt(np.sqrt((idx * 2.0**shift) * ((idx + 1.0) * 2.0**shift)))


# (a) one table, fixed shift 12: index = S >> 12 in 0..255
idx = S >> 12
for lo in (1, 4, 16, 64):
    sel = idx >= lo
    err = np.abs(bucket_entry(idx[sel], 12) / true[sel] - 1)
    print(f"fixed shift 12, index >= {lo:3d} (S >= 2^{12 + int(np.log2(lo))}): max error {err.max() * 100:5.2f}%, "
          f"usable range of S x{256 // lo}, of RMS x{np.sqrt(256 / lo):.1f}")
# (b) leading-one detection with an even shift: mantissa m = S >> e2 in 64..255
k = np.floor(np.log2(S)).astype(int)
e2 = np.maximum((k - 6) // 2 * 2, 0)
m = S >> e2
sel = S >= 64
for outbits in (7, 8):
    top = 2**outbits - 1
    entry = np.round(bucket_entry(m[sel], 0) * 8 * top) / (8 * top)        # 1/sqrt(64) = 1/8 is the largest
    err = np.abs(entry * 2.0 ** (-e2[sel] / 2) / true[sel] - 1)
    print(f"leading-one + even shift, mantissa 64..255, {outbits}-bit entries: max error {err.max() * 100:.2f}%, "
          f"rms {np.sqrt(np.mean(err**2)) * 100:.2f}% (all S from 64 to 2^20)")
# (c) two tables with two fixed shifts: fine table while S >> lo_shift fits in 8 bits, else coarse table
for lo_shift, hi_shift in ((6, 12), (8, 12), (4, 12)):
    use_hi = (S >> lo_shift) > 255
    sh = np.where(use_hi, hi_shift, lo_shift)
    idx2 = S >> sh
    sel = idx2 >= 1
    err = np.abs(bucket_entry(idx2[sel], sh[sel]) / true[sel] - 1)
    first_hi = (256 << lo_shift) >> hi_shift
    print(f"two tables, shifts {lo_shift} and {hi_shift}: coarse table starts at index {first_hi} "
          f"(max error there {err[use_hi[sel]].max() * 100:.2f}%); "
          f"error <= 1.6% for S >= {S[sel][np.flatnonzero(err > 0.016)[-1] + 1]}, "
          f"<= 6% for S >= {S[sel][np.flatnonzero(err > 0.06)[-1] + 1]}")

# 6. Residual stream held in int8 ---------------------------------------------------------------
section("6 fixed-point residual stream after 10 branch additions (stream std 32 int8-LSB at the end)")
dim, adds, rows = 64, 10, 20000
x0 = rng.normal(0, 1, size=(rows, dim))
branches = rng.normal(0, np.sqrt(3 / adds), size=(adds, rows, dim))          # stream std doubles: 1 -> 2
ideal = x0 + branches.sum(0)
lsb = ideal.std() / 32.0                                                     # int8 LSB: clipping at 4 sigma
fresh = 10 * np.log10(32.0**2 * 12)                                          # one ideal rounding at this LSB
print(f"one rounding at this scale: {fresh:.1f} dB")
for name, quant in (("round", np.round), ("floor", np.floor)):
    for label, frac_bits, top in (("int8", 0, 127), ("int16 (6 extra fraction bits)", 6, 32767)):
        unit = lsb / 2**frac_bits
        stream = np.clip(quant(x0 / unit), -top - 1, top)
        for g in range(adds):
            stream = np.clip(stream + quant(branches[g] / unit), -top - 1, top)
        err = stream * unit - ideal
        snr = 10 * np.log10(ideal.var() / err.var())
        gap = (fresh - snr) / 6.02
        print(f"{name} {label}: SNR {snr:5.1f} dB ({abs(gap):.1f} bits {'worse' if gap > 0 else 'better'} than one rounding), "
              f"mean error {err.mean() / lsb:+.2f} int8-LSB, clipped {np.mean(np.abs(stream) >= top) * 100:.3f}%")
