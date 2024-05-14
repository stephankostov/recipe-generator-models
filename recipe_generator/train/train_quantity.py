from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import os
import json
from typing import NamedTuple
import pickle
from tqdm import tqdm
from collections import namedtuple
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import wandb

import recipe_generator.models.gpt as gpt
import recipe_generator.data.data as data
import recipe_generator.optimiser.optimiser as optimiser
from recipe_generator.loss.cross_entropy_mask import MaskedCrossEntropyLoss
from recipe_generator.utils.utils import set_seeds, nostdout
from recipe_generator.models.transformer import IngredientWeightPredictor
from recipe_generator.loss.loss import MaskedMSELoss
from recipe_generator.train.trainer import Trainer

from recipe_generator.config.quantity import IngredientWeightsPredictorCFG
from recipe_generator.config.train import TrainConfig

def main(food_embeddings_file='../data/local/final/full/food_embeddings/0.npy',
        special_token_embeddings_file='../data/local/final/full/special_token_embeddings/0.npy',
        foods_file='../data/local/final/full/food_names/0.npy',
        recipe_foods_file='../data/local/final/full/recipes/foods.npy',
        recipe_weights_file='../data/local/final/full/recipes/weights.npy'):
    
    model_cfg = IngredientWeightsPredictorCFG()
    train_cfg = TrainConfig()

    set_seeds(train_cfg.seed)

    foods = np.load(foods_file)
    recipe_foods = np.load(recipe_foods_file)
    recipe_weights = np.load(recipe_weights_file)
    embedding_weights = {
        'ingredients': torch.tensor(np.load(food_embeddings_file), dtype=torch.float), 
        'special_tokens': torch.tensor(np.load(special_token_embeddings_file), dtype=torch.float)
    }
    
    cv_ratio = 0.8
    shuffle_idx = np.random.permutation(len(recipe_foods))
    recipe_foods, recipe_weights = recipe_foods[shuffle_idx], recipe_weights[shuffle_idx]
    train_idxs, validation_idxs = range(0, round(cv_ratio*len(recipe_foods))), range(round(cv_ratio*len(recipe_foods)), len(recipe_foods))
    train_ds, validation_ds = data.WeightsDataset(recipe_foods[train_idxs], recipe_weights[train_idxs], model_cfg.max_len), data.WeightsDataset(recipe_foods[validation_idxs], recipe_weights[validation_idxs], model_cfg.max_len), 
    train_dl, validation_dl = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2), DataLoader(validation_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2)

    model = IngredientWeightPredictor(embedding_weights, model_cfg)
    model.to(train_cfg.device)

    if train_cfg.wandb: 
        wandb.init(
            project='recipe-generator-weight-test',
            config={ **model_cfg._asdict(), **train_cfg._asdict(), 'loss_note': 'ce_loss' }
        )
        wandb.watch(model, log_freq=train_cfg.save_steps)

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    loss_func = MaskedMSELoss()
    adam_optimiser = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    trainer = Trainer(train_cfg, 
                      train_dl, 
                      validation_dl, 
                      model, 
                      adam_optimiser, 
                      loss_func,
                      metric_funcs=[baseline_loss],
                      sample_inference=partial(sample_inference, foods=foods),
                      wandb=wandb if train_cfg.wandb else None,
                      save_dir=Path('./outputs/quantity'))
    
    trainer.train()

    

def baseline_loss(trainer, batch):
    xb, yb, mask = batch
    baseline_output = 1/(mask!=0).sum(1).unsqueeze(1).expand([1000,15])
    return trainer.loss_func(baseline_output, yb, mask, False)


def sample_inference(model_input, model_output, foods):

    input, target, mask, output = (*model_input, F.sigmoid(model_output))
    input, target, output = [t.detach().to('cpu') for t in [input, target, output]]

    results = pd.DataFrame([], columns=['input', 'target', 'output']).astype({
        'input': 'string', 'target': 'float', 'output': 'float'
    })

    for i in range(5):

        r = pd.DataFrame({
            'input': foods[input[i]],
            'target': torch.round(target[i], decimals=4),
            'output': torch.round(output[i], decimals=4)
        })

        results = pd.concat([
            results if not results.empty else None, 
            r
        ], axis=0)



    return results


if __name__ == '__main__':
    main()