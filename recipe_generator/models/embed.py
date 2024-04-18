import torch
import torch.nn as nn
from torch.nn import functional as F

class FoodEmbeddings(nn.Module):
    "Embedding module using molecule concentration data"
    def __init__(self, cfg, food_embeddings, special_token_embeddings):
        super().__init__()
        self.special_token_embeddings = nn.Embedding.from_pretrained(torch.tensor(special_token_embeddings).float(), padding_idx=0, freeze=False)
        self.molecule_embedding = nn.Embedding.from_pretrained(torch.tensor(food_embeddings).float(), freeze=True)

    def get_weights(self):

        special_token_weights = self.special_token_embeddings.weight
        molecule_weights = self.molecule_embedding.weight

        special_token_weights_padded = torch.zeros(molecule_weights.shape, dtype=torch.float, device=molecule_weights.device)
        special_token_weights_padded[:special_token_weights.shape[0],:]= special_token_weights

        return special_token_weights_padded + molecule_weights

    def forward(self, x):

        special_tokens_mask = x < 4
        special_token_selection = x.clone()
        special_token_selection[~special_tokens_mask] = 0

        e = self.molecule_embedding(x) + self.special_token_embeddings(special_token_selection)

        return e