"""calibrate(): the integer model starts close to the float model it was made from."""
import numpy as np
import pytest
import torch

from lsllm.config import ModelConfig
from lsllm.model import build
from lsllm.qat.calibrate import calibrate, fbits, linear, residual
from lsllm.qat.export import export

torch.set_num_threads(2)
V, T = 32, 16
SUCCESSORS = np.random.default_rng(5).permuted(np.tile(np.arange(V), (V, 1)), axis=1)[:, :3]
VARIANTS = [("dyt", "swiglu"), ("rmsnorm", "relu"), ("dyt", "relu"), ("rmsnorm", "swiglu")]


def bigram(rng, rows):
    """Token rows of a toy language: every token has three possible successors (70%, 20%, 10%)."""
    out = np.zeros((rows, T + 1), dtype=np.int64)
    out[:, 0] = rng.integers(0, V, rows)
    for t in range(T):
        out[:, t + 1] = SUCCESSORS[out[:, t], rng.choice(3, size=rows, p=[0.7, 0.2, 0.1])]
    out = torch.from_numpy(out)
    return out[:, :-1], out[:, 1:]


def float_model(norm, ffn):
    cfg = ModelConfig(vocab_size=V, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=48, max_seq_len=T,
                      norm=norm, ffn=ffn)
    torch.manual_seed(0)
    return build(cfg)


@pytest.fixture(scope="module", params=VARIANTS, ids=["-".join(v) for v in VARIANTS])
def trained(request):
    """(float model after 50 training steps on the toy language, calibration tokens, validation batch)."""
    model, rng = float_model(*request.param), np.random.default_rng(0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.0)
    for step in range(50):
        x, y = bigram(rng, 16)
        loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval(), bigram(rng, 32)[0], bigram(np.random.default_rng(99), 128)


def test_calibrated_loss_is_close_to_the_float_loss(trained):
    model, calib, (x, y) = trained
    with torch.no_grad():
        logits, float_loss = model(x, y)
        u, int_loss = calibrate(model, [calib])(x, y)
    assert float_loss < 2.6                                     # the float model has learned something (ln 32 = 3.47)
    assert abs(int_loss - float_loss) < 0.05, (float(float_loss), float(int_loss))
    # the same greedy token wherever the float model has a clear favourite (integer logits have steps of 2^-f_l nats)
    first, second = logits.topk(2, -1).values.unbind(-1)
    clear = first - second > 0.5
    assert clear.float().mean() > 0.2 and (u.argmax(-1) == logits.argmax(-1))[clear].float().mean() > 0.97


def test_qat_steps_lower_the_loss(trained):
    model, calib, (x, y) = trained
    qat, rng = calibrate(model, [calib]), np.random.default_rng(1)
    with torch.no_grad():
        start = qat(x, y)[1].item()
    opt = torch.optim.AdamW(qat.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.0)
    for step in range(30):
        loss = qat(*bigram(rng, 16))[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
        qat.clip()
    with torch.no_grad():
        assert qat(x, y)[1].item() < start - 0.02


@pytest.mark.parametrize("norm,ffn", VARIANTS)
def test_integer_tensors_follow_the_float_model(norm, ffn, tmp_path):
    """Random weights (enlarged, with per-channel norm gains): head outputs, hidden units, the stream and the
    logits of the integer model, read at their scales, are those of the float model."""
    model = float_model(norm, ffn)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.mul_(6.0)
        for layer in model.layers:
            for site in (layer.attention_norm, layer.ffn_norm):         # gains that calibrate must fold away
                site.weight.mul_(torch.empty(model.cfg.dim).uniform_(0.6, 1.5))
    tokens, seen = torch.randint(0, V, (8, T)), {}
    for l, layer in enumerate(model.layers):
        layer.attention.wo.register_forward_hook(lambda m, args, out, l=l: seen.update({"l%d.o" % l: args[0]}))
        layer.feed_forward.w2.register_forward_hook(lambda m, args, out, l=l: seen.update({"l%d.g" % l: args[0]}))
        layer.ffn_norm.register_forward_hook(lambda m, args, out, l=l: seen.update({"l%d.xa" % l: args[0]}))
    model.norm.register_forward_hook(lambda m, args, out: seen.update({"l1.xf": args[0]}))
    with torch.no_grad():
        logits = model(tokens)[0].numpy()
        qat = calibrate(model, [tokens])
    reference, traces = export(qat, str(tmp_path)), []
    reference.reset(len(tokens))
    for t in range(T):
        traces.append({})
        reference.step(tokens[:, t].numpy(), trace=traces[-1])
    f = qat.frac
    scale = {"l0.o": 2.0 ** -f["o"][0], "l1.o": 2.0 ** -f["o"][1], "l0.g": 2.0 ** -f["g"][0],
             "l1.g": 2.0 ** -f["g"][1], "l0.xa": 2.0 ** -f["x"], "l1.xa": 2.0 ** -f["x"], "l1.xf": 2.0 ** -f["x"]}
    for name, factor in scale.items():
        got = np.stack([trace[name].reshape(len(tokens), -1) for trace in traces], 1).ravel() * factor
        want = seen[name].numpy().ravel()
        assert np.corrcoef(got, want)[0, 1] > 0.9, name
        if name[-2] == "x":         # the stream itself; o and g may be coarse when their update is a few steps of x
            assert np.linalg.norm(got - want) < 0.15 * np.linalg.norm(want), name
    u = np.stack([trace["u"] for trace in traces], 1)
    diff = u * 2.0 ** -f["u"] - logits
    diff -= np.median(diff, -1, keepdims=True)                          # the logit offset
    free = (u > 0) & (u < 127)
    assert free.mean() > 0.9 and np.sqrt((diff[free] ** 2).mean()) < 0.15


def test_scales_keep_the_model_valid_for_tiny_and_for_huge_weights(tmp_path):
    for factor in (1.0, 0.05, 40.0):
        model = float_model("rmsnorm", "swiglu")
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() == 2:
                    p.mul_(factor)
        tokens = torch.randint(0, V, (4, T))
        qat = calibrate(model, [tokens])
        shifts = np.concatenate([np.ravel(value) for value in qat.shifts.values()])
        assert shifts.min() >= 0 and shifts.max() <= 23
        export(qat, str(tmp_path))                      # the load check: shifts of rqr at most 15, 16-bit preloads
        arrays, preloads = qat.integers()
        assert max(abs(c) for row in preloads["c_s"] for c in row) < 2 ** 15 and abs(preloads["c_l"]) < 2 ** 15
        assert all(np.abs(a).max() <= 127 for name, a in arrays.items() if name[0] in "EW")
        assert torch.isfinite(qat(tokens, tokens)[1])


def test_helpers():
    assert [fbits(size) for size in (127.0, 128.0, 1.0, 0.5, 0.26)] == [0, -1, 6, 7, 8]
    assert linear(9, 7, 6) == (10, 9)                   # shift 9 + 7 - 6, weights keep their scale
    assert linear(20, 7, 2) == (15, 10)                 # the shift stops at 15: the weights get coarser
    assert residual(5, 0.25, 2.0) == 5                  # weights fit from f_h = 5 + 8 - 8, h fits up to 5
    assert residual(5, 0.25, 0.5) == 6                  # room on both sides: halfway between 5 and 7
    assert residual(5, 2.0, 2.0) == 7                   # no f_h fits both (8 and 5): halfway, rounded up


def test_notes_and_refusal_of_what_calibration_cannot_do_well():
    """Reported in the notes: scores too coarse for 4 codes per nat, matrices that use little or nothing of int8."""
    tokens = torch.randint(0, V, (4, T))
    model = float_model("rmsnorm", "swiglu")
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.mul_(8.0)
    assert not [note for note in calibrate(model, [tokens]).notes if note.startswith("layer") or "zeros" in note]
    with torch.no_grad():
        for name in ("wq", "wk"):
            getattr(model.layers[1].attention, name).weight.mul_(60.0)
        model.layers[0].feed_forward.w2.weight.mul_(1e-7)
        model.layers[1].feed_forward.w2.weight.mul_(0.05)
    qat = calibrate(model, [tokens])
    assert qat.notes[0].startswith("layer 1: the scores have ") and not qat.notes[1].startswith("layer")
    assert "W2_0 rounds to all zeros: it does nothing in the integer model" in qat.notes
    used = int(np.abs(qat.integers()[0]["W2_1"]).max())
    assert 0 < used < 32 and "W2_1 uses only -%d..%d of -127..127" % (used, used) in qat.notes
    assert qat.shifts["sh_s"][1] == 0 and qat.frac["s"][1] == qat.frac["q"][1] + qat.frac["k"][1] < 2 == qat.frac["s"][0]
    with torch.no_grad():       # the table of that layer is built for its own number of codes per nat
        assert abs(qat(tokens, tokens)[1] - model(tokens, tokens)[1]) < 0.3


def test_what_has_no_integer_form_is_refused():
    tokens = torch.randint(0, V, (2, T))
    base = dict(vocab_size=V, dim=32, n_layers=1, n_heads=4, n_kv_heads=2, hidden_dim=48, max_seq_len=T)
    for attn in ("sigmoid", "cap", "capnorm"):          # SPEC C.6 is softmax; table attention without the row
        with pytest.raises(ValueError):                 # maximum was rejected (D-arch-2, D-arch-3)
            calibrate(build(ModelConfig(attn=attn, **base)), [tokens])
    with pytest.raises(AssertionError):
        calibrate(build(ModelConfig(tied_embeddings=False, **base)), [tokens])
