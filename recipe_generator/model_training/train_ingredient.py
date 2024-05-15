from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import os
import json
from typing import NamedTuple
import pickle
from tqdm import tqdm
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import wandb

import recipe_generator.models.gpt as gpt
import recipe_generator.dataloaders as dataloaders
import recipe_generator.optimiser as optimiser
from recipe_generator.trainer import Trainer
from recipe_generator.loss import MaskedCrossEntropyLoss
from recipe_generator.utils import set_seeds, nostdout

from recipe_generator.config.gpt import GPTConfig
from recipe_generator.config.train import TrainConfig

def main(recipes_file='../data/local/final/full/recipe_food_ids/0.npy',
        food_embeddings_file='../data/local/final/full/food_embeddings/0.npy',
        special_token_embeddings_file='../data/local/final/full/special_token_embeddings/0.npy',
        foods_file='../data/local/final/full/food_names/0.npy'):
    
    model_cfg = GPTConfig()
    train_cfg = TrainConfig()

    set_seeds(train_cfg.seed)

    save_dir=Path('./output/ingredients/')
    save_dir.mkdir(exist_ok=True, parents=True)

    recipes = np.load(recipes_file)
    embedding_weights = {
        'ingredients': torch.tensor(np.load(food_embeddings_file), dtype=torch.float), 
        'special_tokens': torch.tensor(np.load(special_token_embeddings_file), dtype=torch.float)
    }
    foods = np.load(foods_file)
    
    cv_ratio = 0.8
    np.random.shuffle(recipes)
    data_train, data_validation = recipes[:int(cv_ratio*len(recipes))], recipes[int(cv_ratio*len(recipes)):]
    train_ds, validation_ds = dataloaders.NextTokenDataset(data_train, model_cfg.block_size), dataloaders.NextTokenDataset(data_validation, model_cfg.block_size), 
    train_dl, validation_dl = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=2), DataLoader(validation_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2)

    model = gpt.GPTLanguageModel(model_cfg, embedding_weights)
    model.to(train_cfg.device)

    if train_cfg.wandb: 
        wandb.init(
            project='recipe-generator-ingredient',
            config={ **model_cfg._asdict(), **train_cfg._asdict(), 'loss_note': 'ce_loss' }
        )
        wandb.watch(model, log_freq=train_cfg.save_steps)

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    loss_func = MaskedCrossEntropyLoss()
    adam_optimiser = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    trainer = Trainer(
        train_cfg,
        train_dl,
        validation_dl,
        model,
        adam_optimiser,
        loss_func,
        metric_funcs=[calculate_accuracy],
        sample_inference=partial(sample_inference, foods=foods),
        save_dir=Path('./outputs/ingredients/'),
        wandb=wandb if train_cfg.wandb else None
    )

    trainer.train()

def calculate_accuracy(trainer, batch):

    xb, yb, mask = batch

    def pad_after_end_token(preds):

        device=preds.device
        padded_tokens = preds.clone()

        for i, label in enumerate(preds):
            end_index = (label == 3).nonzero()
            mask = torch.ones(label.shape, device=device)
            if end_index.numel(): mask[end_index[0]+1:] = 0
            padded_tokens[i] = label * mask
        
        return padded_tokens

    preds = torch.argmax(trainer.model_output, 2) # B, N
    preds = pad_after_end_token(preds)
    
    match_results = torch.zeros(yb.shape, device=trainer.train_cfg.device) 
    for i in range(yb.shape[1]):
        label = yb[:,i]
        match = preds.eq(label.unsqueeze(1)).any(1) * ((label != 0) * (label != 3)) # B
        match_results[:,i] = match

    accuracy = (match_results).sum() / ((yb != 0) * (yb != 3)).sum()

    return accuracy

def sample_inference(trainer, model_input, foods):

    trainer.model.eval()

    def get_food_id(food):
        ids = np.argwhere(foods==food)
        return ids[0] if ids else 0

    base_foods = ['chicken', 'cream', 'eggplant', 'egg', 'strawberry']
    base_food_ids = [get_food_id(food) for food in base_foods]

    sample_results = np.empty((len(base_foods)*5, 15), dtype=foods.dtype)
    for i, food_id in enumerate(base_food_ids):
        context = (torch.ones((5,1), dtype=torch.long)*food_id).to(trainer.train_cfg.device)
        generations = trainer.model.generate(context, 14).to('cpu')
        sample_results[i*5:(i*5)+5] = foods[generations]

    sample_results = pd.DataFrame(sample_results, dtype='string')

    return sample_results


if __name__ == '__main__':
    main()