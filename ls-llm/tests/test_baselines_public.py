"""Public checkpoints behind the Completer interface. Skipped when transformers or the weight files are missing."""
import os

import pytest

pytest.importorskip("transformers")
from lsllm.baselines.public import MODELS, ROOT, PublicModel

PROMPT = 'def add_numbers(a, b):\n    """Return the sum of a and b."""\n'


@pytest.mark.parametrize("folder", MODELS)
def test_public_model_completes(folder):
    if not os.path.exists(ROOT + folder + "/model.safetensors"):
        pytest.skip("weights of %s are not on disk" % folder)
    completer = PublicModel(folder)
    assert completer.info["params"] == sum(p.numel() for p in completer.model.parameters()) > 0
    assert completer.info["variant"] == "float" and "licence" in completer.info["notes"]
    greedy = completer.complete(PROMPT, 3, 8, 0.0, 0)
    assert len(greedy) == 3 and greedy[0] == greedy[1] == greedy[2]
    assert greedy == completer.complete(PROMPT, 3, 8, 0.0, 5)           # greedy does not depend on the seed
    sampled = completer.complete(PROMPT, 4, 8, 0.8, 1)
    assert len(sampled) == 4 and all(isinstance(text, str) for text in sampled)
    assert sampled == completer.complete(PROMPT, 4, 8, 0.8, 1)          # same seed, same samples
    assert completer.truncated == 0
    completer.complete(PROMPT * 300, 1, 8, 0.0, 0)                      # longer than any context here
    assert completer.truncated == 1
