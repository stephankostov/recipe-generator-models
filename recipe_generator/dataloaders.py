# reference: https://github.com/dhlee347/pytorchic-bert

import numpy as np
import torch
from torch.utils.data import Dataset

from random import randint, shuffle
from random import random as rand

def convert_tokens_to_ids(tokens):
    return tokens

class MLMDataMasker():

    def __init__(self, max_pred, mask_prob, max_len, indexer, token_vocab, special_token_ids):
        self.max_pred = max_pred
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.indexer = indexer
        self.token_vocab = token_vocab
        self.special_token_ids = special_token_ids

    def __call__(self, instance):

        tokens = [t for t in instance if not t == self.special_token_ids.index('pad')] # quick hack to match base code.
        input_mask = [self.special_token_ids.index('mask')]*len(tokens)

        n_pred = min(self.max_pred, max(1, int(round(len(tokens)*self.mask_prob))))

        masked_tokens, masked_pos = [], []

        candidate_positions = [i for i, token in enumerate(tokens)
                               if token not in self.special_token_ids]
        shuffle(candidate_positions)
        for pos in candidate_positions[:n_pred]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            if rand() < 0.8: # 80%
                tokens[pos] = self.special_token_ids.index('mask')
            elif rand() < 0.5: # 10%
                random_token = tokens[0]
                while random_token in tokens: random_token = randint(0, len(self.token_vocab)-1)
                tokens[pos] = random_token

        masked_weights = [1]*len(masked_tokens) # when n_pred < max_pred, we only calculate loss within n_pred

        # Token indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([self.special_token_ids.index('pad')]*n_pad)
        input_mask.extend([self.special_token_ids.index('pad')]*n_pad)

        # Padding for masked target
        if n_pred < self.max_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([self.special_token_ids.index('pad')]*n_pad)
            masked_pos.extend([self.special_token_ids.index('pad')]*n_pad)
            masked_weights.extend([self.special_token_ids.index('pad')]*n_pad)

        return (input_ids, input_mask, masked_ids, masked_pos, masked_weights)

#| export
class MaskedRecipeDataset(Dataset):

    def __init__(self, recipes, process_pipeline):
        self.recipes = recipes
        self.process_pipeline = process_pipeline
    
    def __len__(self):
        return len(self.recipes) # should be able to have more than one here
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.to_list()
        instance = list(self.recipes[idx])
        for process_step in self.process_pipeline:
            instance = process_step(instance)
        return [torch.tensor(x, dtype=torch.long) for x in instance]
    
class NextTokenDataset(Dataset):
    
    def __init__(self, recipes, max_len):
        self.recipes = recipes
        self.max_len = max_len

    def __len__(self):
        return len(self.recipes) # should be able to have more than one here
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx): idx = idx.to_list()
        recipe = self.recipes[idx]

        recipe = [food for food in recipe if food != 0]
        recipe_len = len(recipe)
        pad_count = self.max_len - recipe_len - 1
        end_token = [3] if pad_count>=0 else []
        recipe = recipe + end_token + [0]*pad_count
        assert len(recipe) == self.max_len

        x = torch.tensor(recipe[0:len(recipe)-1])
        y = torch.tensor(recipe[1:len(recipe)])
        mask_ids = torch.tensor([1 if food_id != 0 else 0 for food_id in y])

        return x, y, mask_ids
    
class WeightsDataset(Dataset):

    def __init__(self, recipe_foods, recipe_weights, max_len):
        self.recipe_foods = recipe_foods
        self.recipe_weights = recipe_weights
        self.max_len = max_len

    def __len__(self):
        return len(self.recipe_foods)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx): idx = idx.to_list()

        foods = self.recipe_foods[idx]
        weights = self.recipe_weights[idx]

        pad_idxs = np.where(foods == 0)[0]
        if pad_idxs.size: foods[pad_idxs[0]] = 3

        assert len(foods) == len(weights) == self.max_len 
        assert np.isclose(weights[pad_idxs], 0).all(), print(foods, weights)

        x = torch.tensor(foods, dtype=torch.int)
        y = torch.tensor(weights, dtype=torch.float)

        mask_ids = torch.ones(foods.shape, dtype=torch.float)
        mask_ids[pad_idxs] = 0

        return x, y, mask_ids