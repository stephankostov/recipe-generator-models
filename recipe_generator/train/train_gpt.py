from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import os
import json
from typing import NamedTuple
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import wandb

import recipe_generator.models.gpt as gpt
import recipe_generator.data.data as data
import recipe_generator.optimiser.optimiser as optimiser
from recipe_generator.loss.cross_entropy_mask import MaskedCrossEntropyLoss
from recipe_generator.utils.utils import set_seeds, nostdout

from recipe_generator.config.gpt import GPTConfig
from recipe_generator.config.train import TrainConfig

def main(recipes_file='../data/local/final/full/recipe_food_ids/0.npy',
        food_embeddings_file='../data/local/final/full/food_embeddings/0.npy',
        special_token_embeddings_file='../data/local/final/full/special_token_embeddings/0.npy',
        foods_file='../data/local/final/full/food_names/0.npy'):
    
    model_cfg = GPTConfig()
    train_cfg = TrainConfig()

    set_seeds(train_cfg.seed)

    recipes = np.load(recipes_file)
    food_embeddings = np.load(food_embeddings_file)
    special_token_embeddings = np.load(special_token_embeddings_file)
    foods = np.load(foods_file)
    print("Loaded data", food_embeddings.shape, special_token_embeddings.shape, recipes.shape, foods.shape)
    
    food_embeddings = torch.tensor(food_embeddings, dtype=torch.float)

    cv_ratio = 0.8
    np.random.shuffle(recipes)
    data_train, data_validation = recipes[:int(cv_ratio*len(recipes))], recipes[int(cv_ratio*len(recipes)):]
    train_ds, validation_ds = data.NextTokenDataset(data_train, model_cfg.block_size), data.NextTokenDataset(data_validation, model_cfg.block_size), 
    train_dl, validation_dl = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=2), DataLoader(validation_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2)

    model = gpt.GPTLanguageModel(model_cfg, food_embeddings, special_token_embeddings)
    model.to(train_cfg.device)

    if train_cfg.wandb: 
        wandb.init(
            project='recipe-generator-gpt-losstest',
            config={ **model_cfg._asdict(), **train_cfg._asdict(), 'loss_note': 'ce_loss' }
        )
        wandb.watch(model, log_freq=train_cfg.save_steps)

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    loss_func = MaskedCrossEntropyLoss()
    adam_optimiser = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    training_metrics = []
    global_step = 0

    for epoch in range(train_cfg.n_epochs):
        
        for i, batch in enumerate(tqdm(train_dl)):

            # loading data onto device
            batch = [x.to(train_cfg.device) for x in batch]
            xb, yb, mask = batch

            # training
            model.train()
            output = model(xb)

            adam_optimiser.zero_grad(set_to_none=True)
            loss = loss_func(output.transpose(1,2), yb, mask)
            loss.backward()
            adam_optimiser.step()
            
            global_step += 1

            # evaluation
            if i % train_cfg.save_steps == 0 or global_step >= train_cfg.max_steps:

                with torch.no_grad():

                    batch = next(iter(validation_dl))
                    batch = [x.to(train_cfg.device) for x in batch]

                    xb, yb, mask = batch

                    model.eval()
                    output = model(xb)

                    validation_loss = loss_func(output.transpose(1,2), yb, mask)

                    eval_metrics = {
                        'epoch': epoch, 'global_step': global_step, 
                        'train_loss': loss.item(), 'validation_loss': validation_loss.item(), 
                        'learning_rate': adam_optimiser.param_groups[0]['lr'],
                        'accuracy': calculate_accuracy(output, yb),
                        # 'input': [b.to('cpu') for b in batch],
                        # 'output': output.to('cpu'),
                    }
                    
                    if train_cfg.wandb: wandb.log(eval_metrics, step=global_step)
                    training_metrics.append(eval_metrics)
    
                if global_step >= train_cfg.max_steps: 
                    save(model, (xb), training_metrics, foods, epoch, train_cfg.device, train_cfg.wandb)
                    return
    
        save(model, (xb), training_metrics, foods, epoch, train_cfg.device, train_cfg.wandb)

def calculate_accuracy(model_output, labels):

    def pad_after_end_token(preds):

        device=preds.device
        padded_tokens = preds.clone()

        for i, label in enumerate(preds):
            end_index = (label == 3).nonzero()
            mask = torch.ones(label.shape, device=device)
            if end_index.numel(): mask[end_index[0]+1:] = 0
            padded_tokens[i] = label * mask
        
        return padded_tokens

    preds = torch.argmax(model_output, 2) # B, N
    preds = pad_after_end_token(preds)
    
    match_results = torch.zeros(labels.shape, device=labels.device) 
    for i in range(labels.shape[1]):
        label = labels[:,i]
        match = preds.eq(label.unsqueeze(1)).any(1) * ((label != 0) * (label != 3)) # B
        match_results[:,i] = match

    accuracy = (match_results).sum() / ((labels != 0) * (labels != 3)).sum()

    return accuracy

def save(model, model_input, training_metrics, token_map, epoch, device, wandb_save=False):

    with open('./outputs/gpt/train_metrics.pickle', 'wb') as f: pickle.dump(training_metrics, f)

    if wandb_save:
        with nostdout(): torch.onnx.export(model, model_input, './outputs/gpt/model.onnx')
        wandb.save('./outputs/gpt/model.onnx')

    sample_results = sample_generations(model, token_map, device)
    sample_results.to_string(f'./outputs/gpt/samples_{epoch}.txt')
    if wandb_save: wandb.save(f'./outputs/gpt/samples_{epoch}.txt')

    model.to('cpu')
    torch.save(model.state_dict(), './outputs/gpt/model.pt')
    model.to(device)


def sample_generations(model, foods, device):

    model.eval()

    def get_food_id(food):
        ids = np.argwhere(foods==food)
        return ids[0] if ids else 0

    base_foods = ['chicken', 'cream', 'eggplant', 'egg', 'strawberry']
    base_food_ids = [get_food_id(food) for food in base_foods]

    sample_results = np.empty((len(base_foods)*5, 15), dtype=foods.dtype)
    for i, food_id in enumerate(base_food_ids):
        context = (torch.ones((5,1), dtype=torch.long)*food_id).to(device)
        generations = model.generate(context, 14).to('cpu')
        sample_results[i*5:i+5] = foods[generations]

    sample_results = pd.DataFrame(sample_results, dtype='string')

    return sample_results


if __name__ == '__main__':
    main()