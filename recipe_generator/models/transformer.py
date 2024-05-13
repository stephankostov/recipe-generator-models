import torch
import torch.nn as nn
import torch.nn.functional as F

from recipe_generator.models.embed import FoodEmbeddings

class IngredientWeightPredictor(nn.Module):

    def __init__(self, embedding_weights, ndim, nhead, num_encoder_layers, num_decoder_layers, **kwargs):
        super(IngredientWeightPredictor, self).__init__()
        self.embedding = FoodEmbeddings(embedding_weights)
        self.transformer = nn.Transformer(ndim, nhead, num_encoder_layers, num_decoder_layers)
        self.output_layer = nn.Linear(ndim, 1)  # Single-unit output for weight prediction

    def forward(self, input_ids):

        # Embedding layer
        embedded_input = self.embedding(input_ids)

        # Transformer layers
        output = self.transformer(embedded_input, embedded_input)  # Self-attention

        # Output layer
        weight_predictions = self.output_layer(output).squeeze(-1)

        return weight_predictions