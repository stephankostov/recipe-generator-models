import torch
import torch.nn as nn
import torch.nn.functional as F

from recipe_generator.models.embed import FoodEmbeddings, PositionalEncoding

# class IngredientWeightPredictor(nn.Module):

#     def __init__(self, model_cfg, embedding_weights):
#         super(IngredientWeightPredictor, self).__init__()
#         self.embedding = FoodEmbeddings(model_cfg, embedding_weights)
#         self.transformer = nn.Transformer(model_cfg.ndim, model_cfg.nhead, model_cfg.num_encoder_layers, model_cfg.num_decoder_layers, model_cfg.dim_feedforward)
#         self.output_layer = nn.Linear(model_cfg.ndim, 1)  # Single-unit output for weight prediction

#     def forward(self, input_ids):

#         # Embedding layer
#         embedded_input = self.embedding(input_ids)

#         # Transformer layers
#         output = self.transformer(embedded_input, embedded_input)  # Self-attention

#         # Output layer
#         weight_predictions = self.output_layer(output).squeeze(-1)

#         return weight_predictions
    
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)

#     def generate(self, input_ids):
#         samples = self.forward(input_ids)
#         samples = F.sigmoid(samples)
#         return samples
    
class QuantityModel(nn.Module):

    def __init__(self, model_cfg, embedding_weights):

        super(QuantityModel, self).__init__()

        self.model_cfg = model_cfg

        self.embedding = FoodEmbeddings(model_cfg, embedding_weights)
        # self.quantity_embedding = nn.Linear(1, model_cfg.ndim)

        encoder_layer = nn.TransformerEncoderLayer(model_cfg.ndim, model_cfg.nhead, model_cfg.dim_feedforward, model_cfg.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, model_cfg.num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(model_cfg.ndim, model_cfg.nhead, model_cfg.dim_feedforward, model_cfg.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, model_cfg.num_encoder_layers)

        # Linear layer to predict token weights
        self.weight_predictor = nn.Linear(model_cfg.ndim, 1)

        self.apply(self._init_weights)
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, inputs):

        src, tgt = inputs

        src_embed = self.embedding(src)
        # tgt_embed = self.embedding.position_embeddings(self.quantity_embedding(tgt.unsqueeze(-1)))
        tgt_embed = self.embedding.position_embeddings(tgt.unsqueeze(-1).expand(-1,-1,self.model_cfg.ndim))

        memory = self.encoder(src_embed)

        # Apply causal mask during decoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=self.weight_predictor.weight.device)
        output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)

        weights = self.weight_predictor(output).squeeze(-1)

        return weights
    
    def generate(self, src, tgt=None):

        src_embed = self.embedding(src)
        memory = self.encoder(src_embed)

        if tgt == None: tgt = torch.zeros((src.size(0),1), dtype=torch.float, device=self.weight_predictor.weight.device)

        for _ in range(src.size(1)-tgt.size(1)):

            tgt_embed = self.embedding.position_embeddings(tgt.unsqueeze(-1).expand(-1,-1,self.model_cfg.ndim))
            output = self.decoder(tgt_embed, memory)
            weights = self.weight_predictor(output).squeeze(-1)
            weights = weights[:,-1]
            weights = F.sigmoid(weights)

            tgt = torch.cat([tgt, weights.unsqueeze(-1)], dim=1)

        return tgt

class QuantityPredictorEncoder(nn.Module):

    def __init__(self, model_cfg, embedding_weights):
        super(QuantityPredictorEncoder, self).__init__()
        self.model_cfg = model_cfg
        self.embedding = FoodEmbeddings(model_cfg, embedding_weights)
        encoder_layer = nn.TransformerEncoderLayer(model_cfg.ndim, model_cfg.nhead, model_cfg.dim_feedforward, model_cfg.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, model_cfg.num_encoder_layers)
        self.linear = nn.Linear(model_cfg.ndim, 1)
        self.apply(self._init_weights)
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, src):

        src_embed = self.embedding(src)

        encodings = self.encoder(src_embed)
        encodings = self.linear(encodings).squeeze(-1)

        return encodings