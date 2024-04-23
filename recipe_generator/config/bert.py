from typing import NamedTuple

class BERTConfig(NamedTuple):

    dim: int = 400
    dim_ff: int = dim * 4
    n_layers: int = 2
    n_heads: int = 1
    p_drop_attn: float = 0.1
    p_drop_hidden: float = 0.1
    max_len: int = 15
    n_segments: int = 2
    vocab_size: int = 1110