import torch
import torch.nn as nn
import torch.nn.functional as F

from recipe_generator.models.embed import FoodEmbeddings

class IngredientWeightPredictor(nn.Module):

    def __init__(self, embedding_weights, model_cfg):
        super(IngredientWeightPredictor, self).__init__()
        self.embedding = FoodEmbeddings(model_cfg, embedding_weights)
        self.transformer = nn.Transformer(model_cfg.ndim, model_cfg.nhead, model_cfg.num_encoder_layers, model_cfg.num_decoder_layers, model_cfg.dim_feedforward)
        self.output_layer = nn.Linear(model_cfg.ndim, 1)  # Single-unit output for weight prediction

    def forward(self, input_ids):

        # Embedding layer
        embedded_input = self.embedding(input_ids)

        # Transformer layers
        output = self.transformer(embedded_input, embedded_input)  # Self-attention

        # Output layer
        weight_predictions = self.output_layer(output).squeeze(-1)

        return weight_predictions
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)