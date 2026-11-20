"""Llama-2 style decoder-only Transformer: the float model of LS-LLM.

Adapted from Andrej Karpathy's llama2.c `model.py` (MIT licence; unmodified copy in third_party/llama2c/).
Kept from upstream: the parameter names (upstream weights load directly), RMSNorm, RoPE, grouped-query
attention, SwiGLU, tied embeddings, no biases, the weight initialisation.
Changed: attention is written out operation by operation (scores, causal mask, softmax, value mixing),
forward returns (logits of every position, loss), and the shape options of ModelConfig were added
(untied embeddings, ReLU feed-forward, Dynamic Tanh, sigmoid attention, sliding window).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from lsllm.config import ModelConfig


CODES_PER_NAT = 8.0     # resolution of the int8 attention score code (SPEC C.6)


def _ste_round(x):
    """Round in the forward pass, identity in the backward pass (straight-through)."""
    return x + (torch.round(x) - x).detach()


class CapTable(torch.autograd.Function):
    """The attention weight table of the LS CPU (SPEC C.6) with a straight-through gradient.

    Forward: the score is coded as int8 at CODES_PER_NAT codes per nat (rounded, clamped to -128..127) and looked
    up in the 7-bit exp table round(127 * exp((code - 127) / CODES_PER_NAT)). A hidden position (score -inf) gets
    code -128 and weight 0. Backward: the derivative of the smooth curve 127 * exp(score - 127 / CODES_PER_NAT),
    which is the smooth weight itself. Written as one function so that only one (time, time) tensor is kept.
    """

    @staticmethod
    def forward(ctx, scores):
        code = torch.round(scores * CODES_PER_NAT).clamp_(-128, 127)
        smooth = code.sub_(127.0).div_(CODES_PER_NAT).exp_().mul_(127.0)
        ctx.save_for_backward(smooth)
        return torch.round(smooth)

    @staticmethod
    def backward(ctx, grad):
        (smooth,) = ctx.saved_tensors
        return grad * smooth


class RMSNorm(nn.Module):
    """weight * x / rms(x), over the last dimension."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class DyT(nn.Module):
    """Dynamic Tanh (Zhu et al. 2025, "Transformers without Normalization"): weight * tanh(alpha * x)."""

    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))    # one learnable scalar; 0.5 is the paper's default
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch.tanh(self.alpha * x) * self.weight


def make_norm(cfg):
    return RMSNorm(cfg.dim) if cfg.norm == "rmsnorm" else DyT(cfg.dim)


def rope_tables(head_dim, max_seq_len):
    """cos and sin of (position * frequency), each of shape (max_seq_len, head_dim / 2)."""
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
    return torch.cos(angles), torch.sin(angles)


def rotate(x, cos, sin):
    """RoPE: rotate every (even, odd) pair of x (batch, heads, time, head_dim) by the angle of its position."""
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg, dropout, flash):
        super().__init__()
        self.cfg = cfg
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.kv_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.kv_dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        if cfg.attn == "sigmoid":       # Ramapuram et al. 2024: sigmoid(score + b), b starts at -log(n)
            self.b = nn.Parameter(torch.tensor(-math.log(cfg.max_seq_len)))
        if cfg.attn in ("cap", "capnorm", "cappow2"):
            # float stand-in for the integer attention of the LS CPU (docs/SPEC.md C.6): scores get a per-head
            # offset, are coded as int8 at 8 codes per nat, and go through a 7-bit exp table referenced to the
            # top code 127 (no search for the row maximum). "cap" mixes the values with these weights as they
            # are; "capnorm" divides by their sum.
            # start: "capnorm" one nat below the top code (weights about 47 of 127, then normalised); "cap" three
            # nats below (weights about 6 of 127 each: non-zero at every context length, and the first sums stay small)
            start = 127.0 / CODES_PER_NAT - (3.0 if cfg.attn == "cap" else 1.0)
            self.cap_b = nn.Parameter(torch.full((cfg.n_heads,), start))
        self.dropout = nn.Dropout(dropout)
        self.flash = flash and cfg.attn == "softmax"

    def forward(self, x, cos, sin, hide):
        cfg = self.cfg
        B, T, _ = x.shape
        share = cfg.n_heads // cfg.n_kv_heads       # query heads per key/value head (grouped-query attention)
        # queries, keys and values as (batch, heads, time, head_dim); RoPE turns the queries and the keys
        q = rotate(self.wq(x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2), cos, sin)
        k = rotate(self.wk(x).view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2), cos, sin)
        v = self.wv(x).view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        if self.flash:
            # the same arithmetic done by PyTorch's kernel; on the CPU it never stores the (time, time) matrices
            k, v = k.repeat_interleave(share, dim=1), v.repeat_interleave(share, dim=1)
            drop = self.dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=hide, dropout_p=drop)
        else:
            # the query heads that share a key/value head get their own axis:
            # q (batch, kv heads, share, time, head_dim) against k and v (batch, kv heads, 1, time, head_dim)
            q, k, v = q.view(B, cfg.n_kv_heads, share, T, cfg.head_dim), k[:, :, None], v[:, :, None]
            scores = (q / math.sqrt(cfg.head_dim)) @ k.transpose(-2, -1)    # (batch, kv heads, share, time, time)
            scores += hide                                                  # -inf where a position may not look
            if cfg.attn == "softmax":
                weights = F.softmax(scores, dim=-1)
            elif cfg.attn == "sigmoid":
                weights = torch.sigmoid(scores + self.b)                    # sigmoid(-inf) = 0
            else:
                if cfg.attn.startswith("max"):      # reference = the largest score of the row (found by the CPU)
                    offset = 127.0 / CODES_PER_NAT - scores.amax(-1, keepdim=True).detach()
                else:                               # reference = a trained offset per head (no search on the CPU)
                    offset = self.cap_b.view(1, cfg.n_kv_heads, share, 1, 1)
                weights = CapTable.apply(scores + offset)                   # 0..127
                total = weights.sum(-1, keepdim=True)                       # 0 .. 127 * time
                if cfg.attn in ("capnorm", "maxnorm"):                      # exact division (reciprocal table on the CPU)
                    weights = weights / total.clamp_min(1.0)
                elif cfg.attn in ("cappow2", "maxpow2"):                    # divide by the next power of two above the sum:
                    weights = weights / torch.exp2(torch.floor(torch.log2(total.clamp_min(1.0))) + 1.0)    # a shift on the CPU
                else:
                    v = v / 127.0                                           # weight 127 counts as 1.0
            out = (self.dropout(weights) @ v).view(B, cfg.n_heads, T, cfg.head_dim)
        out = out.transpose(1, 2).reshape(B, T, cfg.dim)
        return self.dropout(self.wo(out))


class FeedForward(nn.Module):
    def __init__(self, cfg, dropout):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)
        if cfg.ffn == "swiglu":
            self.w3 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.swiglu = cfg.ffn == "swiglu"
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = F.silu(self.w1(x)) * self.w3(x) if self.swiglu else F.relu(self.w1(x))
        return self.dropout(self.w2(hidden))


class Block(nn.Module):
    def __init__(self, cfg, dropout, flash):
        super().__init__()
        self.attention = Attention(cfg, dropout, flash)
        self.feed_forward = FeedForward(cfg, dropout)
        self.attention_norm = make_norm(cfg)
        self.ffn_norm = make_norm(cfg)

    def forward(self, x, cos, sin, hide):
        x = x + self.attention(self.attention_norm(x), cos, sin, hide)
        return x + self.feed_forward(self.ffn_norm(x))


class Llama(nn.Module):
    def __init__(self, cfg, dropout=0.0, flash=False):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0 and cfg.head_dim % 2 == 0
        assert cfg.ffn in ("swiglu", "relu") and cfg.norm in ("rmsnorm", "dyt") and cfg.attn in ("softmax", "sigmoid", "cap", "capnorm", "cappow2", "maxnorm", "maxpow2")
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(Block(cfg, dropout, flash) for _ in range(cfg.n_layers))
        self.norm = make_norm(cfg)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tied_embeddings:
            self.tok_embeddings.weight = self.output.weight
        if cfg.norm == "dyt":           # the DyT paper adds this scalar for Llama: without a norm layer
            self.emb_scale = 64.0       # embeddings start at std 0.02: times 64 is about unit size; a power of two, so free on the CPU

        T = cfg.max_seq_len
        cos, sin = rope_tables(cfg.head_dim, T)
        allowed = torch.ones(T, T, dtype=torch.bool).tril()             # allowed[t, s]: may t look at s?
        if cfg.window:
            allowed &= torch.ones(T, T, dtype=torch.bool).triu(1 - cfg.window)    # only the last `window` positions
        hide = torch.zeros(T, T).masked_fill(~allowed, float("-inf"))   # added to the attention scores
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("hide", hide, persistent=False)

        # initialisation as upstream: every matrix N(0, 0.02), smaller for wo and w3
        # (upstream scales w3 although w2 is the output projection of the feed-forward; kept to match it)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                small = name.endswith(("wo.weight", "w3.weight"))
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers) if small else 0.02)

    def forward(self, tokens, targets=None):
        """tokens (batch, time) -> logits (batch, time, vocab), and the mean cross-entropy if targets are given."""
        T = tokens.shape[1]
        x = self.tok_embeddings(tokens)
        if self.cfg.norm == "dyt":
            x = x * self.emb_scale
        x = self.dropout(x)
        cos, sin, hide = self.cos[:T], self.sin[:T], self.hide[:T, :T]
        for layer in self.layers:
            x = layer(x, cos, sin, hide)
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def build(cfg: ModelConfig, dropout=0.0, flash=False):
    """The model for a shape. `dropout` and `flash` are training settings, not part of the shape.

    flash=True computes softmax attention with F.scaled_dot_product_attention (same result; on the CPU
    faster and far smaller in memory). The default is the written-out attention.
    """
    return Llama(cfg, dropout, flash)


def load(path, device="cpu"):
    """Rebuild the model stored in a checkpoint file. Returns (model, checkpoint dictionary)."""
    ckpt = torch.load(path, map_location=device)
    model = build(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt
