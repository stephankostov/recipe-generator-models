from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import os
import json
from typing import NamedTuple
import pickle
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import models.gpt as gpt
import data.data as data
import optimiser.optimiser as optimiser
from utils.utils import set_seeds

class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    device: str = 'cuda'

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

def main(train_cfg='./config/train.json',
          model_cfg='./config/gpt_base.json',
          recipes_file='../data/local/final/full/recipe_food_ids/0.npy',
          food_vectors_file='../data/local/final/full/food_compounds/0.npy'):
    
    train_cfg = Config.from_json(train_cfg)
    model_cfg = gpt.Config.from_json(model_cfg)

    device = train_cfg.device if torch.cuda.is_available() else 'cpu'
    set_seeds(train_cfg.seed)

    food_vectors = np.load(food_vectors_file)
    recipes = np.load(recipes_file)

    print("Loaded data", food_vectors.shape, recipes.shape)

    # set special token indexes
    special_token_ids = ['pad','mask','unknown']
    
    food_vectors = torch.tensor(food_vectors, dtype=torch.float)

    cv_ratio = 0.8
    np.random.shuffle(recipes)
    data_train, data_validation = recipes[:int(cv_ratio*len(recipes))], recipes[int(cv_ratio*len(recipes)):]
    train_ds, validation_ds = data.NextTokenDataset(data_train), data.NextTokenDataset(data_validation), 
    train_dl, validation_dl = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=2), DataLoader(validation_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2)
    
    model = gpt.GPTLanguageModel(model_cfg, food_vectors)
    model.to(device)

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    loss_func = nn.CrossEntropyLoss()
    adam_optimiser = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    training_metrics = []
    global_step = 0

    for epoch in range(train_cfg.n_epochs):
        
        for i, batch in enumerate(tqdm(train_dl)):

            # loading data onto device
            batch = [x.to(device) for x in batch]
            xb, yb = batch

            # training
            model.train()
            output = model(xb)

            adam_optimiser.zero_grad(set_to_none=True)
            loss = loss_func(output.transpose(1,2), yb)
            loss.backward()
            adam_optimiser.step()
            
            global_step += 1

            # evaluation
            if i % train_cfg.save_steps == 0 or global_step >= train_cfg.total_steps:

                with torch.no_grad():

                    batch = next(iter(validation_dl))
                    batch = [x.to(device) for x in batch]

                    xb, yb = batch

                    model.eval()
                    output = model(xb)

                    validation_loss = loss_func(output.transpose(1,2), yb)

                    training_metrics.append({
                        'epoch': epoch, 'global_step': global_step, 
                        'train_loss': loss.item(), 'validation_loss': validation_loss.item(), 
                        'learning_rate': adam_optimiser.param_groups[0]['lr'],
                        'accuracy': calculate_accuracy(output, yb),
                        'perplexity': torch.exp(validation_loss.to('cpu')),
                        'input': [b.to('cpu') for b in batch],
                        'output': output.to('cpu'),
                    })
    
                if global_step >= train_cfg.total_steps: 
                    with open('./outputs/gpt/train_metrics.pickle', 'wb') as f:
                        pickle.dump(training_metrics, f)
                    return
    
    with open('./outputs/gpt/train_metrics.pickle', 'wb') as f:
        pickle.dump(training_metrics, f)

def calculate_accuracy(model_output, labels):
    # output: batch, n_tokens, n_predictions
    # labels: batch, n_predictions
    preds = torch.argmax(model_output, 2) # batch, n_predictions
    accuracy = torch.sum((preds==labels) * (labels != 0)) / torch.sum((labels != 0))

    return accuracy



if __name__ == '__main__':
    main()