"""Model shape shared by the trainer, the speed model, the integer reference and the exports."""
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 512
    dim: int = 64
    n_layers: int = 5
    n_heads: int = 8            # query heads
    n_kv_heads: int = 4         # key/value heads (grouped-query attention); must divide n_heads
    hidden_dim: int = 172       # feed-forward hidden size
    max_seq_len: int = 256      # context length T
    tied_embeddings: bool = True
    ffn: str = "swiglu"         # "swiglu" | "relu"
    norm: str = "rmsnorm"       # "rmsnorm" | "dyt"
    attn: str = "softmax"       # "softmax" | "sigmoid" | "cap" | "capnorm" | "cappow2" (integer-style, SPEC C.6)
    window: int = 0             # 0 = attend to the whole context, else sliding window length

    @property
    def head_dim(self):
        return self.dim // self.n_heads

    @property
    def kv_dim(self):
        return self.n_kv_heads * self.head_dim

    def n_params(self):
        """Number of weights (all stored as int8 on the FPGA, so also bytes)."""
        emb = self.vocab_size * self.dim
        attn = 2 * self.dim * self.dim + 2 * self.dim * self.kv_dim
        ffn = (3 if self.ffn == "swiglu" else 2) * self.dim * self.hidden_dim
        norms = 2 * self.dim
        total = emb + self.n_layers * (attn + ffn + norms) + self.dim
        return total if self.tied_embeddings else total + emb

    def kv_cache_bytes(self):
        """K and V vectors kept for every context position and layer."""
        span = self.window if self.window else self.max_seq_len
        return 2 * self.n_layers * self.kv_dim * span

    def to_dict(self):
        return asdict(self)


STORIES260K = ModelConfig()   # the llama2.c stories260K shape with a 256-token context
