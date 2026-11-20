"""Write a QatModel as the integer model of SPEC Part E: a directory with model.json and model.npz."""
import json
import os

import numpy as np

from lsllm.intref import IntModel
from lsllm.intref.model import ACC_MAX
from lsllm.intref.overflow import bounds


def export(model, directory):
    """Returns the integer reference loaded from the written files (loading checks shifts, preloads and ranges).
    Refuses a model in which some stream could leave the 24-bit accumulator in the worst case (SPEC B.4)."""
    arrays, preloads = model.integers()
    spec = {"config": model.cfg.to_dict(), "options": model.options, "frac": model.frac, "shifts": model.shifts,
            "preloads": preloads, "tokenizer": model.tokenizer, "tables": {"alpha": model.alpha, "gain": model.gain},
            "notes": model.notes}
    for stream, bound in bounds(IntModel(spec, arrays)).items():
        if bound > ACC_MAX:
            raise ValueError("stream %s: worst-case accumulator %d is outside 24 bits" % (stream, bound))
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "model.json"), "w") as f:
        json.dump(spec, f, indent=1)
    np.savez(os.path.join(directory, "model.npz"), **arrays)
    return IntModel.load(directory)
