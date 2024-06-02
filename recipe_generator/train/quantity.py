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

import recipe_generator.models.ingredient as ingredient
import recipe_generator.datasets as datasets
import recipe_generator.optimiser as optimiser
from recipe_generator.loss import MaskedCrossEntropyLoss
from recipe_generator.utils import load_config, set_seeds, nostdout
from recipe_generator.models.quantity import QuantityModel
from recipe_generator.loss import MaskedMSELoss
from recipe_generator.trainer import Trainer

def main(food_embeddings_file='../data/local/final/full/food_embeddings/0.npy',
        special_token_embeddings_file='../data/local/final/full/special_token_embeddings/0.npy',
        foods_file='../data/local/final/full/food_names/0.npy',
        recipe_foods_file='../data/local/final/full/recipes/recipe_food_ids.npy',
        recipe_weights_file='../data/local/final/full/recipes/recipe_food_weights.npy'):
    
    model_cfg = load_config('./recipe_generator/config/quantity.yaml')
    train_cfg = load_config('./recipe_generator/config/train.yaml')

    set_seeds(train_cfg.seed)

    foods = np.load(foods_file)
    recipe_foods = np.load(recipe_foods_file)
    recipe_weights = np.load(recipe_weights_file)
    embedding_weights = {
        'ingredients': np.load(food_embeddings_file), 
        'special_tokens': np.load(special_token_embeddings_file)
    }
    
    cv_ratio = 0.8
    shuffle_idx = np.random.permutation(len(recipe_foods))
    recipe_foods, recipe_weights = recipe_foods[shuffle_idx], recipe_weights[shuffle_idx]
    train_idxs, validation_idxs = range(0, round(cv_ratio*len(recipe_foods))), range(round(cv_ratio*len(recipe_foods)), len(recipe_foods))
    train_ds, validation_ds = datasets.WeightsDataset(recipe_foods[train_idxs], recipe_weights[train_idxs], model_cfg.max_len), datasets.WeightsDataset(recipe_foods[validation_idxs], recipe_weights[validation_idxs], model_cfg.max_len), 
    train_dl, validation_dl = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2), DataLoader(validation_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2)

    model = QuantityModel(model_cfg, embedding_weights)
    model.to(train_cfg.device)

    if train_cfg.wandb: 
        wandb.init(
            project='recipe-generator-quantity-v1',
            config={ **model_cfg.to_dict(), **train_cfg.to_dict(), 'loss_note': 'ce_loss' }
        )
        wandb.watch(model, log_freq=train_cfg.save_steps)
        artifact = wandb.Artifact('model_cfg', type='model_cfg')
        artifact.add_file('./recipe_generator/config/quantity.yaml')
        wandb.log_artifact(artifact)

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
    baseline_output = 1/(mask!=0).sum(1).unsqueeze(1).expand(mask.shape)
    return trainer.loss_func(baseline_output, yb, mask, False)


def sample_inference(trainer, model_input, foods):

    input, target, mask, output = (*model_input, F.sigmoid(trainer.model_output))
    input = [t.detach().to('cpu') for t in input]
    target, output = [t.detach().to('cpu') for t in [target, output]]

    results = pd.DataFrame([], columns=['input', 'target', 'output']).astype({
        'input': 'string', 'target': 'float', 'output': 'float'
    })

    for i in range(5):

        r = pd.DataFrame({
            'input': foods[input[0][i, 1:]],
            'target': torch.round(target[i]*1000, decimals=0),
            'output': torch.round(output[i]*1000, decimals=0)
        })

        results = pd.concat([
            results if not results.empty else None, 
            r
        ], axis=0)

    return results


if __name__ == '__main__':
    main()