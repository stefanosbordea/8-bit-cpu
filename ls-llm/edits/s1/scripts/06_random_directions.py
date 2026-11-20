"""No model run: two random unit directions (seeds 1 and 2) as CONTROLS for script 05. If removing a random
direction changes the scores as much as removing the repetition direction, the repetition direction is not
what helped. Writes directions_06.npz (names r1, r2).
Run from the repository root: PYTHONPATH=.:edits/s1/scripts .venv/bin/python edits/s1/scripts/06_random_directions.py"""
import numpy as np

from common import DIR

out = {}
for seed in (1, 2):
    v = np.random.default_rng(seed).standard_normal(160)
    out["r%d" % seed] = v / np.linalg.norm(v)
np.savez(DIR + "/directions_06.npz", **out)
