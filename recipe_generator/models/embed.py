import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import math

class FoodEmbeddings(nn.Module):

    "Embedding module using molecule concentration data"
    def __init__(self, model_cfg, embedding_weights):
        super().__init__()
        self.special_token_embeddings = nn.Embedding.from_pretrained(torch.tensor(embedding_weights['special_tokens'], dtype=torch.float), padding_idx=0, freeze=False)
        self.molecule_embeddings = nn.Embedding.from_pretrained(torch.tensor(embedding_weights['ingredients'], dtype=torch.float), freeze=True)
        self.position_embeddings = PositionalEncoding(d_model=embedding_weights['ingredients'].shape[1], dropout=0, max_len=model_cfg.max_len) if model_cfg.pos_embeds else None

    def get_weights(self):

        special_token_weights = self.special_token_embeddings.weight
        molecule_weights = self.molecule_embeddings.weight

        special_token_weights_padded = torch.zeros(molecule_weights.shape, dtype=torch.float, device=molecule_weights.device)
        special_token_weights_padded[:special_token_weights.shape[0],:] = special_token_weights

        return special_token_weights_padded + molecule_weights

    def forward(self, x):

        special_tokens_mask = x < 4
        special_token_selection = x.clone()
        special_token_selection[~special_tokens_mask] = 0

        e = self.molecule_embeddings(x) + self.special_token_embeddings(special_token_selection)
        if self.position_embeddings: e = self.position_embeddings(e)

        return e
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)