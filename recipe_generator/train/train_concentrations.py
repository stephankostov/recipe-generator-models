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
import torch.nn.functional as F

import wandb

import recipe_generator.models.gpt as gpt
import recipe_generator.data.data as data
import recipe_generator.optimiser.optimiser as optimiser
from recipe_generator.loss.cross_entropy_mask import MaskedCrossEntropyLoss
from recipe_generator.utils.utils import set_seeds, nostdout
from recipe_generator.models.bert import WeightConcentrationBERT
from recipe_generator.loss.loss import MaskedMSELoss

from recipe_generator.config.bert import BERTConfig
from recipe_generator.config.train import TrainConfig

def main(food_embeddings_file='../data/local/final/full/food_embeddings/0.npy',
        special_token_embeddings_file='../data/local/final/full/special_token_embeddings/0.npy',
        foods_file='../data/local/final/full/food_names/0.npy',
        recipe_foods_file='../data/local/final/full/recipes/foods.npy',
        recipe_weights_file='../data/local/final/full/recipes/weights.npy'):
    
    model_cfg = BERTConfig()
    train_cfg = TrainConfig()

    set_seeds(train_cfg.seed)

    foods = np.load(foods_file)
    recipe_foods = np.load(recipe_foods_file)
    recipe_weights = np.load(recipe_weights_file)
    food_embeddings = torch.tensor(np.load(food_embeddings_file), dtype=torch.float)
    special_token_embeddings = torch.tensor(np.load(special_token_embeddings_file), dtype=torch.float)
    print("Loaded data", food_embeddings.shape, special_token_embeddings.shape, recipe_foods.shape, foods.shape)
    
    cv_ratio = 0.8
    shuffle_idx = np.random.permutation(len(recipe_foods))
    recipe_foods, recipe_weights = recipe_foods[shuffle_idx], recipe_weights[shuffle_idx]
    train_idxs, validation_idxs = range(0, round(cv_ratio*len(recipe_foods))), range(round(cv_ratio*len(recipe_foods)), len(recipe_foods))
    train_ds, validation_ds = data.WeightsDataset(recipe_foods[train_idxs], recipe_weights[train_idxs], model_cfg.max_len), data.WeightsDataset(recipe_foods[validation_idxs], recipe_weights[validation_idxs], model_cfg.max_len), 
    train_dl, validation_dl = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2), DataLoader(validation_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2)

    model = WeightConcentrationBERT(model_cfg, food_embeddings, special_token_embeddings)
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
            loss = loss_func(output, yb, mask)
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

                    validation_loss = loss_func(output, yb, mask)

                    eval_metrics = {
                        'epoch': epoch, 'global_step': global_step, 
                        'train_loss': loss.item(), 'validation_loss': validation_loss.item(), 
                        'learning_rate': adam_optimiser.param_groups[0]['lr'],
                        'error': torch.sqrt(validation_loss)
                    }
                    
                    if train_cfg.wandb: wandb.log(eval_metrics, step=global_step)
                    training_metrics.append(eval_metrics)
    
                if global_step >= train_cfg.max_steps: 
                    save(model, (xb, yb), output, training_metrics, foods, epoch, train_cfg.device, train_cfg.wandb)
                    return
    
        save(model, (xb, yb), output, training_metrics, foods, epoch, train_cfg.device, train_cfg.wandb)

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

def save(model, model_input, model_output, training_metrics, token_map, epoch, device, wandb_save=False, artifact=None):

    with open('./outputs/gpt/train_metrics.pickle', 'wb') as f: pickle.dump(training_metrics, f)

    if wandb_save:
        with nostdout(): torch.onnx.export(model, model_input[0], './outputs/weights/model.onnx')
        wandb.save('./outputs/weights/model.onnx')

    validation_results = output_samples(model_input, model_output, token_map)
    validation_results.to_string(f'./outputs/weights/validation_results{epoch}.txt')
    if wandb_save:
        wandb.save(f'./outputs/weights/validation_results{epoch}.txt')

    # sample_results = sample_generations(model, token_map, device)
    # sample_results.to_string(f'./outputs/gpt/samples_{epoch}.txt')
    # if wandb_save: wandb.save(f'./outputs/gpt/samples_{epoch}.txt')

    model.to('cpu')
    torch.save(model.state_dict(), './outputs/weights/model.pt')
    model.to(device)


def output_samples(model_input, model_output, foods):

    input, target, output = (*model_input, F.softmax(model_output, dim=1))
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