from typing import NamedTuple

class GPTConfig(NamedTuple):

    vocab_size: int = 1110 
    n_embd: int = 400 
    n_layers: int = 2 
    n_heads: int = 1
    head_size: int = n_embd // n_heads 
    p_drop_hidden: float = 0.1 
    p_drop_attn: float = 0.1 
    block_size: int = 17
    # eisl_weight: float = 0.5
    # eisl_ngram: int = 4
