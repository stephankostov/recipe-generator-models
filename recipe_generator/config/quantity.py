from typing import NamedTuple

class IngredientWeightsPredictorCFG(NamedTuple):

    ndim: int = 400
    nhead: int = 1
    dim_feedforward: int = 800
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    max_len: int = 14
    pos_embeds = True
    dropout = 0.1
