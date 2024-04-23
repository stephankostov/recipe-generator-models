# reference: https://github.com/dhlee347/pytorchic-bert

import numpy as np
import torch
from torch.utils.data import Dataset

from random import randint, shuffle
from random import random as rand

def convert_tokens_to_ids(tokens):
    return tokens