from typing import NamedTuple

class IngredientWeightsPredictorCFG(NamedTuple):

    ndim: int = 400
    nhead: int = 1
    dim_feedforward: int = 800
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    max_len: int = 15
    pos_embeds = False
