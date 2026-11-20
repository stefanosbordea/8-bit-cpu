"""The lookup tables of SPEC Part C, each with the slope of the smooth function it samples.

A table has 256 int8 entries, indexed by the unsigned byte of an int8 value (index 0..127 = values 0..127,
index 128..255 = values -128..-1). The slope is the derivative of the smooth function at every entry: the
QAT backward pass uses it, the exported model does not contain it.
"""
import numpy as np

VALUES = np.where(np.arange(256) < 128, np.arange(256), np.arange(256) - 256)     # the int8 value at every index


def dyt(alpha, f_x, amp):
    """NORM table of a dyt site (SPEC C.2): amp * tanh(alpha * x), x = value / 2^f_x; amp = 2^f_n (times a gain)."""
    t = np.tanh(alpha * VALUES * 2.0 ** -f_x)
    return np.clip(np.round(amp * t), -127, 127), amp * alpha * 2.0 ** -f_x * (1 - t * t)


def rms(dim, sa, sb, amp):
    """NORM table of an rms site (SPEC C.2): the multiplier r for m = sum(x^2) >> sa; amp as in dyt."""
    m = np.arange(128)
    smooth = 2.0 ** sb * amp * np.sqrt(dim / ((m + 0.5) * 2.0 ** sa))
    table, slope = np.zeros(256), np.zeros(256)                 # m is never negative: the upper half is unused
    table[:128], slope[:128] = np.clip(np.round(smooth), 1, 127), -0.5 * smooth / (m + 0.5)
    return table, slope


def silu(f_h1, f_g1):
    """SILU table (SPEC C.8): 2^f_g1 * silu(value / 2^f_h1)."""
    z = VALUES * 2.0 ** -f_h1
    s = 1.0 / (1.0 + np.exp(-z))
    return np.clip(np.round(2.0 ** f_g1 * z * s), -127, 127), 2.0 ** (f_g1 - f_h1) * s * (1 + z * (1 - s))


def exp(kappa):
    """EXP table of a layer (SPEC C.6): attention weight 0..127 of (score code - largest code of the row) = 0, -1,
    .., -127 at kappa codes per nat; 0 for the values that cannot occur."""
    smooth = np.where((VALUES <= 0) & (VALUES > -128), 127 * np.exp(VALUES / kappa), 0.0)
    return np.round(smooth), smooth / kappa


def recip():
    """RECIP table (SPEC C.6): 8192 / m for the mantissa m = 64..127 of a sum of attention weights."""
    m = np.maximum(VALUES, 64)                                  # smaller indices occur only for an empty sum
    return np.minimum(127, np.round(8192 / m)), -8192.0 / (m * m)


def rope(seq_len, head_dim):
    """ROPE_C and ROPE_S (SPEC C.4): 127 * cos and sin of position * theta_j, shape (seq_len, head_dim / 2)."""
    theta = 10000.0 ** (-np.arange(0, head_dim, 2) / head_dim)
    angle = np.outer(np.arange(seq_len), theta)
    return np.round(127 * np.cos(angle)), np.round(127 * np.sin(angle))
